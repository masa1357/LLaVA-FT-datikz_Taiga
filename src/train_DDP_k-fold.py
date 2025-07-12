# =====================================================================
# train_DDP_k-fold.py
# date: 2025/07/11
# description:
#   - LLMのLoRAチューニングを行う
#   - OOM対策として ZeRO Data Parallelism を採用
#   - k-foldを採用
# 対象モデル；
#   - elyza/Llama-3-ELYZA-JP-8B
# =====================================================================

# ================ 標準ライブラリ ================
import os
import sys
import re
import argparse
from logging import (
    getLogger,
    StreamHandler,
    Formatter,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL,
    Logger,
)
from functools import partial
from typing import Dict, List

# ================ サードパーティ ================
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset , ConcatDataset
from torchinfo import summary

from dotenv import load_dotenv  # .env 読み込み
import sacrebleu  # BLEU
import evaluate  # ROUGE / BERTScore / MoverScore ラッパ
# from moverscore_v2 import word_mover_score, get_idf_dict -> ローカルに移行


from transformers import Trainer, TrainingArguments, set_seed, GenerationConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold


# ================ プロジェクト内（ローカル） ================
from util import load_model, set_seed, set_logger
from gradepred_data_forkfold import GradePredictionDataset, ext_GPDataset, GradePredictionCollator
from accelerate import Accelerator
from moverscore import word_mover_score, get_idf_dict

# -------- environment setting --------
load_dotenv()

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TORCH_ENABLE_DISTRIBUTED"] = "1"
os.environ["TORCH_DTENSOR_SKIP_CHECK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ACCELERATE_DISABLE_FREE_MEMORY"] = (
    "1"  #  AssertionError: DeepSpeed backend not set, please initialize it using init_process_group()への対策
)

BYTES_PER_PARAM = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
}

if not hasattr(np, "float"):
    np.float = float

class LoggingWithStepCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_str = {
                "step": state.global_step,
                **logs  # loss, lr, etc.
            }
            


def evaluate(
    pred_result,
    eval_dataset,
    tokenizer,
    show_samples: int = 5,
    logger: Logger = getLogger("EvaluationLogger"),
) -> Dict[str, float]:

    # logger.info(f"pred_result elements\t:{pred_result._asdict().keys()}")
    logger.info("start evaluate!")
    if isinstance(pred_result, dict):
        logger.info(f"pred_result keys\t:{list(pred_result.keys())}")
        input_text = pred_result["input_sentence"]
        pred_text = pred_result["output_sentence"]
        label_text = pred_result["label_sentence"]
        # input_text = pred_result["input_sentence"]
    else:  # NamedTuple
        logger.info(f"pred_result elements\t:{pred_result._asdict().keys()}")
        input_text = pred_result.input_sentence
        pred_text = pred_result.output_sentence
        label_text = pred_result.label_sentence
        # input_text = pred_result.input_sentence

    logger.info(f"#pred={len(pred_text)}, #labels={len(label_text)}")

    if show_samples > 0:
        logger.info("✅️ Visualize sample answers")
        for i in range(show_samples):
            msg = "\n".join(
                [
                    "========================",
                    f"Sample {i}:",
                    f"Raw Questionnaires\t: \n{eval_dataset[i]['input_text']}",
                    f"Predict\t: {pred_text[i]}",
                    f"Target\t: {label_text[i]}",
                    "========================",
                ]
            )
            logger.info(msg)
            msg = ""

    # Metrics #1    : BLEU
    bleu = sacrebleu.corpus_bleu(pred_text, [label_text]).score

    # Metrics #2    : MoverScore
    idf_p = get_idf_dict(pred_text)
    idf_r = get_idf_dict(label_text)
    ms = word_mover_score(
        label_text,
        pred_text,
        idf_r,
        idf_p,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
        batch_size=16,
        device = torch.device("cpu")
    )
    moverscore = float(np.mean(ms))
    # moverscore = 1.0  # 仮置き値（実際の計算はコメントアウト）

    # Metrics #3    : Accuracy
    # -> target内には[A,B,C,D,F]のいずれかを含む文字列が入っている
    # predictionとtargetから最初に出現する["A", "B", "C", "D", "F"]を抽出して比較する
    def extract_grade(text: str) -> str:
        # 1. 文字列から[/INST]以前の部分を削除
        text = text.split("[/INST]")[-1]

        # 2. 文字列から成績を抽出
        # 「成績は、Xです」の X を正規表現で抜く
        m = re.search(r"成績は、([A-D]|F)です", text)
        if m:
            return m.group(1)
        else:
            for grade in ["A", "B", "C", "D", "F"]:
                if grade in text:
                    return grade
        return "F"  # デフォルトは F

    pred_grades = [extract_grade(text) for text in pred_text]
    label_grades = [extract_grade(text) for text in label_text]
    accuracy = np.mean(np.array(pred_grades) == np.array(label_grades)) * 100
    cf = confusion_matrix(
        label_grades, pred_grades, labels=["A", "B", "C", "D", "F"]
    )
    

    # 予測に成功しているケースをいくつか表示
    if show_samples > 0:
        logger.info("⭕ Visualize successful predictions")
        for i in range(len(pred_text)):
            if pred_grades[i] == label_grades[i]:
                msg = "\n".join(
                    [
                        "========================",
                        f"Sample {i}:",
                        f"Predict\t: {pred_text[i]} (Grade: {pred_grades[i]})",
                        f"Target\t: {label_text[i]} (Grade: {label_grades[i]})",
                        "========================",
                    ]
                )
                logger.info(msg)

    # 予測に失敗しているケースをいくつか表示
    if show_samples > 0:
        logger.info("❌ Visualize failed predictions")
        for i in range(len(pred_text)):
            if pred_grades[i] != label_grades[i]:
                msg = "\n".join(
                    [
                        "========================",
                        f"Sample {i}:",
                        f"Predict\t: {pred_text[i]} (Grade: {pred_grades[i]})",
                        f"Target\t: {label_text[i]} (Grade: {label_grades[i]})",
                        "========================",
                    ]
                )
                logger.info(msg)

    return {
        "bleu": round(bleu, 4),
        "moverscore": round(moverscore, 4),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": cf
    }


def custom_compute_metrics(res: EvalPrediction) -> Dict:

    return {}


# DummyFile: 何も書き込まないダミークラス
class DummyFile:
    def write(self, x):
        pass  # 何もしない

    def flush(self):
        pass  # 何もしない


def main():
    # ? logger設定
    print("set logger")
    logger = set_logger(level=DEBUG)

    acc = Accelerator()
    if not acc.is_main_process:
        logger.setLevel(CRITICAL)
        # ランク0以外のプロセスでは、sys.stdoutをDummyFileにリダイレクト
        sys.stdout = DummyFile()
        # sys.stderr = DummyFile()
    logger.info(f"logger set complite")

    # ================================================================
    # パラメータの取得
    # ================================================================

    parser = argparse.ArgumentParser(
        description="Fine-tune LLama with LoRA on Reflection dataset"
    )

    # モデルや出力に関する設定
    parser.add_argument(
        "--base_model",
        type=str,
        default="elyza/Llama-3-ELYZA-JP-8B",
        help="Base model ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/lora-elyza-reflection",
        help="Directory to save LoRA-tuned checkpoints",
    )

    # LoRAパラメータ
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    # データ・学習設定
    parser.add_argument("--max_words", type=int, default=4096, help="data max_words")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Global batch size (across all devices)",
    )
    parser.add_argument(
        "--micro_batch_size", type=int, default=1, help="Batch size per device"
    )

    # ログ出力設定
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "none"],
        help="Reporting backend for logging",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="lora-elyza-reflection_test",
        help="Run name for experiment tracking",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="NA",
        help="File name for logging (default: NA, no file logging)",
    )

    args = parser.parse_args()
    set_seed(42)

    # ================================================================
    # DDP (Distributed Data Parallel) の設定
    # ================================================================

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    # if ddp := world_size != 1:
    #     gradient_accumulation_steps = gradient_accumulation_steps // world_size
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError(
            "global_batch_size must be divisible by per_device_batch_size × WORLD_SIZE"
        )

    ddp = world_size != 1
    gradient_accumulation_steps = max(
        1, args.batch_size // args.micro_batch_size // world_size
    )

    if ddp:
        logger.info(f"DDP is enabled (ddp = {ddp}, world_size = {world_size})")
    else:
        logger.info(f"DDP is disabled (ddp = {ddp}, world_size = {world_size})")

    # ================================================================
    # モデル，データセットのロード
    # ================================================================

    model, tokenizer = load_model(args.base_model, if_ZeRO=True)
    summary(model)

    # データセットを読み込む
    dataset_path = "./data/"

    dataset = GradePredictionDataset(
        dataset_path=dataset_path,
        logger = logger
    )
    # ext_dataset = ext_GPDataset(
    #     dataset_path=dataset_path
    # )

    dataset.reset()

    #! 単文データの場合
    dataset.concat()

    #! 15 * 5データの場合
    # dataset.unzip()


    example_collator = GradePredictionCollator(
            tokenizer,
            max_tokens=args.max_words,
            include_target=True,
            logger=logger,
        )


    logger.info(
        f"custom_collate_fn initialized with processor: {type(tokenizer).__name__}"
    )

    # collatorの動作を確認
    example_loader = DataLoader(
        dataset, collate_fn=example_collator, batch_size=4, shuffle=True
    )
    batch = next(iter(example_loader))
    logger.info(
        f"Batch keys: {batch.keys()}"
    )  # -> Batch keys: dict_keys(['input_ids', 'attention_mask', 'labels'])
    for k, v in batch.items():
        logger.info(
            f"Batch key: {k}, value shape: {v.shape if isinstance(v, torch.Tensor) else type(v)}"
        )
    # batch内のinput_idsをでコードしてプロンプトを確認
    input_text = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

    logger.info(f"Input prompt: \n{input_text}")
    logger.debug(f"batch:\n{batch}")

    # example_loaderを削除
    del(example_loader)

    # ================================================================
    # 訓練前推論
    # !訓練前にモデルを使用すると，訓練できいバグが発生
    # -> 訓練の設定前に推論を行い，モデルを再度ロードし直すことで回避
    # ================================================================
    msg = "\n".join(
        [
            "=================================================",
            "🔄 Start Evaluation before training...",
            "=================================================",
        ]
    )
    logger.info(msg)

    pred_text = []
    input_text = []
    label_text = []
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=example_collator,
    )

    model, eval_loader = acc.prepare(model, eval_loader)  # DDP対応化

    # if acc.is_main_process:  # rank 0 だけ表示
    #     st = acc.state
    #     acc.print(
    #         f"Accelerate initialized ⇒ "
    #         f"world_size={st.num_processes}, "
    #         f"local_rank={st.local_process_index}, "
    #         f"device={st.device}"
    #     )
    model.eval()  # モデルを評価モードに設定

    # modelのdeviceを設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for batch in eval_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        with torch.no_grad():
            # 生成
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        target_ids = batch["target_ids"]
        # decode
        tgt_str = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

        input_text.extend(decoded_inputs)
        pred_text.extend(decoded_outputs)
        label_text.extend(tgt_str)

    # gather_for_metrics で全 rank 分を rank0 に集約
    input_text = acc.gather_for_metrics(input_text)
    pred_text = acc.gather_for_metrics(pred_text)
    label_text = acc.gather_for_metrics(label_text)

    # 先頭切り捨て
    input_text = input_text[0 : len(dataset)]
    pred_text = pred_text[0 : len(dataset)]
    label_text = label_text[0 : len(dataset)]

    pred_result = {
        "input_sentence": input_text,
        "output_sentence": pred_text,
        "label_sentence": label_text,
    }

    metrics = evaluate(
        pred_result, dataset, tokenizer, show_samples=5, logger=logger
    )
    logger.info("✅ Evaluation before training completed successfully!")
    logger.info(
        f"Metrics\t:\nMoverScore\t: {metrics['moverscore']}\nAccuracy\t: {metrics['accuracy']}\n"
    )
    msg = "\n".join(
        [
            "=================================================",
            "confusion_matrix:",
            f"{metrics['confusion_matrix']}",
            "=================================================",
        ]
    )
    logger.info(msg)
    logger.info(
        "\n====================================\n===================================="
    )

    # モデルの削除（一時保留）
    del model, tokenizer, eval_loader
    torch.cuda.empty_cache()  # GPUメモリを解放
    logger.info("🔄 Reloading model and tokenizer...")

    # 再度ロード
    model, tokenizer = load_model(args.base_model, if_ZeRO=True)

    logger.info("✅ Model and tokenizer reloaded successfully!")

    # ================================================================
    # 訓練用の設定
    # ================================================================

    # ① LoRA設定
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Self-Attention系
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP（FFN）系
        ],
    )

    # LoRA適用
    model.enable_input_require_grads()  #! 追加::入力テンソルに勾配を流せる状態を強制する安全スイッチ
    model = get_peft_model(model, peft_config)

    logger.info(f"Trainable parameters:")
    model.print_trainable_parameters()
    logger.info("✅ LoRA has been successfully applied to the model.")
    logger.debug(summary(model))

    # full_finetune_modules のモジュールを再度 trainable にする
    full_finetune_modules = ["embed_tokens", "lm_head"]

    logger.info("🔧 Applying LoRA and enabling full finetune modules...")

    # LoRA 適用済み（前段） -> ZeRO 3 でエラーが発生する可能性あり
    # for name, param in model.named_parameters():
    #     if any(module_name in name for module_name in full_finetune_modules):
    #         param.requires_grad = False
    #         param.data = param.data.to(torch.float16)

    logger.info("✅ LoRA has been applied.")
    logger.info(
        f"✅ The following modules are fully finetuned: {', '.join(full_finetune_modules)}"
    )

    model.gradient_checkpointing_enable()
    # if local_rank == 0:
    #     model.logger.info_trainable_parameters()
    #     summary(model, depth=2)

    logger.debug("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug(f" - {name}: {param.shape}, dtype: {param.dtype}")

    print("=" * 100)
    print("=" * 100)

    # ②TrainingArguments
    training_args = TrainingArguments(
        gradient_checkpointing=True,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        logging_dir="./logs",
        logging_steps=50,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        fp16_full_eval=True,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        remove_unused_columns=False,
        run_name=args.run_name,
        save_total_limit=args.epochs,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=False,
        label_names=["labels"],  # PEFT環境下では明示したほうが良いらしい？
        learning_rate=2e-6,
        disable_tqdm=True,  # tqdm消す
    )

    groups = np.array([dataset[i]["userid"] for i in range(len(dataset))])
    logger.info(len(groups))
    gkf = GroupKFold(n_splits=5)

    train_logger = set_logger(name="CollateTrain", level=INFO)
    eval_logger = set_logger(name="CollateEval", level=INFO)


    # ================================================================
    # 交差検証(k=5)
    # ================================================================
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(
        X=np.zeros(len(dataset)), groups=groups)):
        msg = "\n".join(
            [
                "=================================================",
                f"🔄 Start {fold}th fold...",
                "=================================================",
            ]
        )
        logger.info(msg)
        #  --- train ------------------------------------------
        train_dataset = Subset(dataset, tr_idx)
        logger.info(f"len(train_dataset): {len(train_dataset)}")
        # --- valid ---------------------------------------------
        eval_dataset   = Subset(dataset, va_idx)
        logger.info(f"len(eval_dataset): {len(eval_dataset)}")

        train_collator = GradePredictionCollator(
            tokenizer,
            max_tokens=args.max_words,
            include_target=True,
            logger=train_logger,
            question_filter=[1],
        )
        eval_collator = GradePredictionCollator(
            tokenizer,
            max_tokens=args.max_words,
            include_target=False,
            logger=eval_logger,
            question_filter=[1],
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=train_collator,
            # compute_metrics=custom_compute_metrics,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[LoggingWithStepCallback],
            # predict_with_generate=True,  # 推論時に generate を使用
        )

        logger.info("Trainer instance has been created.")
        logger.info(
            f"Trainer is set with model: {type(model).__name__}, train dataset size: {len(train_dataset)}, eval dataset size: {len(eval_dataset)}"
        )

        # ================================================================
        # 訓練
        # ================================================================
        print("=" * 100)
        print("=" * 100)

        # trainer.accelerator.end_training()  # ★ inference Accelerator を閉じる
        # trainer._created_accelerator = False  # ★ “作成済み” フラグをリセット
        model.train()
        logger.info("🔄 Start training...")
        trainer.data_collator = train_collator
        trainer.train()
        logger.info("✅ Model training has been completed successfully!")

        # モデルの保存
        file_path = args.output_dir + args.logfile
        if args.logfile != "NA":
            logger.info(f"Saving model to {file_path}")
            model.save_pretrained(file_path)
            tokenizer.save_pretrained(file_path)
        else:
            logger.info("No logfile specified, skipping model save.")

        # ================================================================
        # 訓練後推論
        # ================================================================
        print("=" * 100)
        print("=" * 100)

        logger.info("🔄 Start Evaluation after training...")
        trainer.data_collator = eval_collator

        # ---
        model.eval()
        pred_text = []
        input_text = []
        label_text = []
        loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=eval_collator,
        )
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            with torch.no_grad():
                # 生成
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=64,
                    do_sample=True,  # 推論時は通常Greedy
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            target_ids = batch["target_ids"]
            # decode
            tgt_str = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

            input_text.extend(decoded_inputs)
            pred_text.extend(decoded_outputs)
            label_text.extend(tgt_str)

        # 結果を格納（Trainer.predictの形式に寄せたい場合）
        pred_result = {
            "input_sentence": input_text,
            "output_sentence": pred_text,
            "label_sentence": label_text,
        }

        metrics = evaluate(
            pred_result, eval_dataset, tokenizer, show_samples=5, logger=logger
        )
        logger.info("✅ Evaluation after training completed successfully!")
        logger.info(
            f"Metrics\t:\nMoverScore\t: {metrics['moverscore']}\nAccuracy\t: {metrics['accuracy']}\n"
        )
        msg = "\n".join(
            [
                "=================================================",
                "confusion_matrix:",
                f"{metrics['confusion_matrix']}",
                "=================================================",
            ]
        )
        logger.info(msg)


if __name__ == "__main__":
    main()
