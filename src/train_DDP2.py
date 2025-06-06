# =====================================================================
# train.py
# date: 2025/06/04
# description:
#   - LLMのLoRAチューニングを行う
#   - OOM対策として ZeRO Data Parallelism を採用
# 対象モデル；
#   - elyza/Llama-3-ELYZA-JP-8B
# =====================================================================

# ================ 標準ライブラリ ================
import os
import argparse
from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO, WARNING, Logger
from functools import partial
from typing import Dict, List

# ================ サードパーティ ================
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from dotenv import load_dotenv  # .env 読み込み
import sacrebleu  # BLEU
import evaluate  # ROUGE / BERTScore / MoverScore ラッパ
from moverscore_v2 import word_mover_score, get_idf_dict

from transformers import Trainer, TrainingArguments, set_seed, GenerationConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers.trainer_utils import EvalPrediction, PredictionOutput

# ================ プロジェクト内（ローカル） ================
from util import load_model, set_seed, set_logger
from gradepred_data import GradePredictionDataset, GradePredictionCollator

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


def evaluate(
    pred_result: PredictionOutput,
    eval_dataset,
    tokenizer,
    show_samples: int = 5,
    logger: Logger = getLogger("EvaluationLogger"),
) -> Dict[str, float]:

    logger.info(f"pred_result elements\t:{pred_result._asdict().keys()}")

    pred_logits = pred_result.predictions  # (bs, seq, vocab) か np.object_
    logger.info(f"pred_logits shape\t: {pred_logits.shape}")
    logger.debug(f"pred_logits \t: \n{pred_logits}")
    if pred_logits.ndim == 3:  # logits パターン
        pred_ids = pred_logits.argmax(-1)  # (bs, seq)
    else:  # 既に ID が入っている場合
        logger.info("pred_logits is already in ID format, skipping argmax operation.")
        pred_ids = pred_logits

    # -100 → pad に置換（labels も同様に）
    pred_ids = pred_ids.tolist()  # list[list[int]]
    label_ids = pred_result.label_ids
    logger.info(f"label_ids shape\t: {label_ids.shape}")
    logger.debug(f"label_ids \t: \n{label_ids}")
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_ids = label_ids.tolist()

    pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_text = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if show_samples > 0:
        logger.info("✅️ Visualize sample answers")
        for i in range(min(show_samples, len(pred_text))):
            logger.info(f"Sample {i}\t: ")
            logger.info(f"Raw Questionnaires\t: \n{eval_dataset[i]['input_text']}")
            # print(f"Raw Questionnaires\t: \n{eval_dataset[i]['input_text']}")

            logger.info(f"Predict\t: \n{pred_text[i]}")
            logger.info(f"Target\t: \n{labels_text[i]}")

    # Metrics #1    : BLEU
    bleu = sacrebleu.corpus_bleu(pred_text, [labels_text]).score

    # Metrics #2    : MoverScore
    idf_p = get_idf_dict(pred_text)
    idf_r = get_idf_dict(labels_text)
    ms = word_mover_score(
        labels_text,
        pred_text,
        idf_r,
        idf_p,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
        batch_size=16,
    )
    moverscore = float(np.mean(ms))

    # Metrics #3    : Accuracy
    # -> target内には[A,B,C,D,F]のいずれかを含む文字列が入っている
    # predictionとtargetから最初に出現する["A", "B", "C", "D", "F"]を抽出して比較する
    def extract_grade(text: str) -> str:
        for grade in ["A", "B", "C", "D", "F"]:
            if grade in text:
                return grade
        return "F"  # デフォルトは F

    pred_grades = [extract_grade(text) for text in pred_text]
    label_grades = [extract_grade(text) for text in labels_text]
    accuracy = np.mean(np.array(pred_grades) == np.array(label_grades)) * 100

    return {
        "bleu": round(bleu, 4),
        "moverscore": round(moverscore, 4),
        "accuracy": round(accuracy, 4),
    }


def custom_compute_metrics(res: EvalPrediction) -> Dict:

    return {}


def main():
    # ? logger設定
    print("set logger")
    logger = set_logger(level=INFO)
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

    train_dataset = GradePredictionDataset(
        dataset_path=dataset_path,
        question_filter=[1],
        concatenate=True,
        mode="train",
    )
    eval_dataset = GradePredictionDataset(
        dataset_path=dataset_path,
        question_filter=[1],
        concatenate=True,
        mode="valid",
    )

    logger.info(f"len(train_dataset): {len(train_dataset)}")
    logger.info(f"len(eval_dataset): {len(eval_dataset)}")

    # 独自collatorの定義
    # custom_collate_fn = partial(
    #     collate_fn, tokenizer=tokenizer, max_tokens=args.max_words, testcase=True, question_filter=[1]
    # )
    train_logger = set_logger(name="CollateTrain", level=INFO)
    eval_logger = set_logger(name="CollateEval", level=DEBUG)

    train_collator = GradePredictionCollator(
        tokenizer,
        max_tokens=args.max_words,
        include_target=True,
        logger=train_logger,
    )
    eval_collator = GradePredictionCollator(
        tokenizer,
        max_tokens=args.max_words,
        include_target=False,
        logger=eval_logger,
    )

    logger.info(
        f"custom_collate_fn initialized with processor: {type(tokenizer).__name__}"
    )

    # collatorの動作を確認
    example_loader = DataLoader(
        train_dataset, collate_fn=train_collator, batch_size=4, shuffle=True
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
    logger.info(f"Input prompt: {input_text}")
    logger.debug(f"\n{batch}")

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

    logger.info("=" * 100)
    logger.info("=" * 100)

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
        # learning_rate=2e-4,
        learning_rate=2e-6,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=train_collator,
        # compute_metrics=custom_compute_metrics,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Trainer instance has been created.")
    logger.info(
        f"Trainer is set with model: {type(model).__name__}, train dataset size: {len(train_dataset)}, eval dataset size: {len(eval_dataset)}"
    )

    # ================================================================
    # 訓練前推論
    # ================================================================
    # ! モデルの出力config ! 不要
    # model.generation_config = GenerationConfig(
    #     return_dict_in_generate=False,
    #     output_scores=False,  # これも一緒に False にしておくと安全
    #     do_sample=False,  # 評価では通常 OFF
    #     max_new_tokens=128,  # 必要に応じて
    # )

    # logger.info(" Start Evaluation before training...")
    # trainer.data_collator = eval_collator
    # pred_result = trainer.predict(eval_dataset)

    # metrics = evaluate(pred_result, eval_dataset, tokenizer, logger=logger)
    # logger.info(
    #     f"Metrics\t:\nMoverScore\t: {metrics['moverscore']}\nAccuracy\t: {metrics['accuracy']}\n"
    # )

    # ================================================================
    # 訓練
    # ================================================================
    # trainer.accelerator.end_training()  # ★ inference Accelerator を閉じる
    # trainer._created_accelerator = False  # ★ “作成済み” フラグをリセット
    logger.info("🔄 Start training...")
    trainer.data_collator = train_collator
    trainer.train()
    logger.info("✅ Model training has been completed successfully!")

    # ================================================================
    # 訓練後推論
    # ================================================================
    logger.info(" Start Evaluation after training...")
    trainer.data_collator = eval_collator
    pred_result = trainer.predict(eval_dataset)

    # logger.debug(f"pred_result elements\t:{pred_result._asdict().keys()}")
    # pred_text = tokenizer.batch_decode(
    #     pred_result.predictions, skip_special_tokens=True
    # )
    # labels_text = tokenizer.batch_decode(
    #     pred_result.label_ids, skip_special_tokens=True
    # )

    # logger.info("✅️ Visualize sample answers")
    # for i in range(5):
    #     logger.info(f"Sample {i}\t: ")
    #     logger.info(f"Predict\t: {pred_text[i]}")
    #     logger.info(f"Target\t: {labels_text[i]}")

    metrics = evaluate(
        pred_result, eval_dataset, tokenizer, show_samples=5, logger=logger
    )
    logger.info(
        f"Metrics\t:\nMoverScore\t: {metrics['moverscore']}\nAccuracy\t: {metrics['accuracy']}\n"
    )


if __name__ == "__main__":
    main()
