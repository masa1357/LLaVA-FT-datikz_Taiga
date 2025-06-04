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
import logging
from functools import partial
from typing import Dict, List

# ================ サードパーティ ================
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from dotenv import load_dotenv                # .env 読み込み
import sacrebleu                              # BLEU
import evaluate                               # ROUGE / BERTScore / MoverScore ラッパ
from moverscore import word_mover_score, get_idf_dict

from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

# ================ プロジェクト内（ローカル） ================
from util import load_model, set_seed, set_logger
from gradepred_data import GradePredictionDataset, GradePredictionCollator

# -------- environment setting --------
load_dotenv()

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TORCH_ENABLE_DISTRIBUTED"] = "1"
os.environ["TORCH_DTENSOR_SKIP_CHECK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BYTES_PER_PARAM = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
}

def evaluate(pred_result: Dict[str, List[str]], test_dataset) -> Dict[str, float]:
    # 入力: pred_result["output_sentence"], test_dataset["labels"]
    preds: List[str] = pred_result["output_sentence"]
    refs: List[str] = test_dataset["labels"]  # reference sentences

    # Metrics #1    : BLEU (sacrebleu)
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score

    # Metrics #2    : MoverScore
    idf_dict_hyp = get_idf_dict(preds)
    idf_dict_ref = get_idf_dict(refs)
    moverscore_list = word_mover_score(
        refs, preds,
        idf_dict_ref, idf_dict_hyp,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
        batch_size=16
    )
    moverscore = sum(moverscore_list) / len(moverscore_list)

    return {
        "bleu": round(bleu, 4),
        "moverscore": round(moverscore, 4)
    }

def custom_compute_metrics(res: EvalPrediction) -> Dict:
    pred_ids  = res.predictions
    label_ids = res.label_ids

    # logits → ids (generate=False 時の保険) 
    if pred_ids.ndim == 3 : # (bs, seq, vocab)
        pred_ids = pred_ids.argmax(-1)

    # -100 を PAD に置換
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    preds  = tokenizer.batch_decode(pred_ids,   skip_special_tokens=True)
    labels = tokenizer.batch_decode(label_ids,  skip_special_tokens=True)

    postprocess = lambda seq: [s.strip() for s in seq]
    preds   = postprocess(preds)
    labels  = postprocess(labels)
    
    # ① n-gram 系
    bleu_refs = [[l] for l in labels]
    bleu_res  = bleu.compute(predictions=preds,   references=bleu_refs)                 # :contentReference[oaicite:5]{index=5}
    rouge_res = rouge.compute(predictions=preds,  references=labels, use_stemmer=True)  # :contentReference[oaicite:6]{index=6}
    
    # ② 埋め込み類似系    
    bert_res  = bertscore.compute(predictions=preds, references=labels, lang="ja")      # :contentReference[oaicite:7]{index=7}
    mover_res = mover.compute(predictions=preds,     references=labels)                 # :contentReference[oaicite:8]{index=8}

    return {
        "bleu":        bleu_res["bleu"],
        "rouge1":      rouge_res["rouge1"],
        "rouge2":      rouge_res["rouge2"],
        "rougeL":      rouge_res["rougeL"],
        "bertscore_f1": float(np.mean(bert_res["f1"])),
        "moverscore":   float(np.mean(mover_res["score"])),
    }

def main():
    #? logger設定
    logger = set_logger(level="DEBUG")

    #================================================================
    # パラメータの取得
    #================================================================

    parser = argparse.ArgumentParser(description="Fine-tune LLama with LoRA on Reflection dataset")

    # モデルや出力に関する設定
    parser.add_argument("--base_model",     type = str, default = "elyza/Llama-3-ELYZA-JP-8B",      help = "Base model ID")
    parser.add_argument("--output_dir",     type = str, default = "./outputs/lora-elyza-reflection",help = "Directory to save LoRA-tuned checkpoints")

    # LoRAパラメータ
    parser.add_argument("--lora_r",         type = int,    default = 8,   help = "LoRA rank")
    parser.add_argument("--lora_alpha",     type = int,    default = 16,  help = "LoRA alpha")
    parser.add_argument("--lora_dropout",   type = float,  default = 0.1, help = "LoRA dropout")

    # データ・学習設定
    parser.add_argument("--max_words",          type = int, default = 4096, help = "data max_words")
    parser.add_argument("--epochs",             type = int, default = 3,    help = "Number of training epochs")
    parser.add_argument("--batch_size",         type = int, default = 1,    help = "Global batch size (across all devices)")
    parser.add_argument("--micro_batch_size",   type = int, default=1,      help = "Batch size per device")

    # ログ出力設定
    parser.add_argument("--report_to",      type=str,   default="none", choices=["wandb", "tensorboard", "none"],   help="Reporting backend for logging")
    parser.add_argument("--run_name",       type=str,   default="lora-elyza-reflection_test",                       help="Run name for experiment tracking")

    args = parser.parse_args()
    set_seed(42)

    #================================================================
    # DDP (Distributed Data Parallel) の設定
    #================================================================   
    
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
        print(f"[info] DDP is enabled (ddp = {ddp}, world_size = {world_size})")
    else:
        print(f"[info] DDP is disabled (ddp = {ddp}, world_size = {world_size})")


    #================================================================
    # モデル，データセットのロード
    #================================================================

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

    logger.info("[info] len(train_dataset):", len(train_dataset))
    logger.info("[info] len(eval_dataset):", len(eval_dataset))

    # 独自collatorの定義
    # custom_collate_fn = partial(
    #     collate_fn, tokenizer=tokenizer, max_tokens=args.max_words, testcase=True, question_filter=[1]
    # )

    train_collator = GradePredictionCollator(
        tokenizer,
        max_tokens=args.max_words,
        include_target=True,
        logger=logging.getLogger("CollateTrain"),
    )
    eval_collator = GradePredictionCollator(
        tokenizer,
        max_tokens=args.max_words,
        include_target=False,
        logger=logging.getLogger("CollateEval"),
    )

    logger.info(
        f"[info] custom_collate_fn initialized with processor: {type(tokenizer).__name__}"
    )

    # collatorの動作を確認
    example_loader = DataLoader(train_dataset, collate_fn=custom_collate_fn, batch_size=4, shuffle=True)
    batch = next(iter(example_loader))
    for k,v in batch.items():
        logger.info(k, v.shape)
    logger.info(batch)


    #================================================================
    # 訓練用の設定
    #================================================================

    # ① LoRA設定
    peft_config = LoraConfig(
        r               = args.lora_r,
        lora_alpha      = args.lora_alpha,
        lora_dropout    = args.lora_dropout,
        bias            = "none",
        task_type       = TaskType.CAUSAL_LM,
        target_modules  = [
            # "q_proj",
            # "k_proj",
            # "v_proj",
            # "o_proj",         # Self-Attention系
            "gate_proj",
            "up_proj",
            "down_proj",        # MLP（FFN）系
        ],
    )

    # LoRA適用
    model.enable_input_require_grads()  #! 追加::入力テンソルに勾配を流せる状態を強制する安全スイッチ
    model = get_peft_model(model, peft_config)
    model.logger.info_trainable_parameters()
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

    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name, param.shape, param.dtype)

    logger.info("=" * 100)
    logger.info("=" * 100)

    # ②TrainingArguments
    training_args = TrainingArguments(
        gradient_checkpointing      = True,
        output_dir                  = args.output_dir,
        per_device_train_batch_size = args.micro_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,  # args.grad_accum,
        num_train_epochs            = args.epochs,
        warmup_ratio                = 0.03,
        logging_dir                 = "./logs",
        logging_steps               = 50,
        lr_scheduler_type           = "cosine",
        optim                       = "adamw_torch",
        save_strategy               = "epoch",
        eval_strategy               = "epoch",
        fp16                        = True,
        fp16_full_eval              = True,
        per_device_eval_batch_size  = 1,
        eval_accumulation_steps     = 1,
        remove_unused_columns       = False,
        run_name                    = args.run_name,
        save_total_limit            = args.epochs,
        ddp_find_unused_parameters  = False,
        load_best_model_at_end      = False,
        )


    trainer = Trainer(
        model           = model,
        tokenizer       = tokenizer,
        data_collator   = train_collator,
        compute_metrics = custom_compute_metrics,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
    )

    logger.info("[info] Trainer instance has been created.")
    logger.info(
        f"[info] Trainer is set with model: {type(model).__name__}, train dataset size: {len(train_dataset)}, eval dataset size: {len(eval_dataset)}"
    )

    #================================================================
    # 訓練前推論
    #================================================================
    logger.info(" Start Evaluation before training...")
    trainer.data_collator = eval_collator
    pred_result = trainer.predict(eval_dataset)
    logger.debug(f"pred_result elements\t:{pred_result.keys()}")
    pred_text = tokenizer.batch_decode(
        pred_result.predictions, skip_special_tokens=True
    )
    labels_text = tokenizer.batch_decode(
        pred_result.label_ids, skip_special_tokens=True
    )

    logger.info("✅️ Visualize sample answers")
    for i in range(5):
        logger.info(f"Sample {i}\t: ")
        logger.info(f"Predict\t: {pred_text[i]}")
        logger.info(f"Target\t: {labels_text[i]}")


    metrics = evaluate(pred_result, eval_dataset)
    logger.info(f"Metrics\t:\nMoverScore\t: {metrics['moverscore']}\n")
    
    #================================================================
    # 訓練
    #================================================================
    logger.info("🔄 Start training...")
    trainer.data_collator = train_collator
    trainer.train()
    logger.info("✅ Model training has been completed successfully!")

    #================================================================
    # 訓練後推論
    #================================================================
    logger.info(" Start Evaluation after training...")
    trainer.data_collator = eval_collator
    pred_result = trainer.predict(eval_dataset)
    logger.debug(f"pred_result elements\t:{pred_result.keys()}")
    pred_text = tokenizer.batch_decode(
        pred_result.predictions, skip_special_tokens=True
    )
    labels_text = tokenizer.batch_decode(
        pred_result.label_ids, skip_special_tokens=True
    )


    logger.info("✅️ Visualize sample answers")
    for i in range(5):
        logger.info(f"Sample {i}\t: ")
        logger.info(f"Predict\t: {pred_text[i]}")
        logger.info(f"Target\t: {labels_text[i]}")
        
    metrics = evaluate(pred_result, eval_dataset)
    logger.info(f"Metrics\t:\nMoverScore\t: {metrics['moverscore']}\n")
    



if __name__ == "__main__":
    main()