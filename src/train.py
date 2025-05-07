# =====================================================================
# train.py
# date: 2025/05/06
# description:
#   - LLMのLoRAチューニングを行う
# 対象モデル；
#   - elyza/Llama-3-ELYZA-JP-8B
# =====================================================================

# -------- import libraries --------
import os
import argparse
from dotenv import load_dotenv
from transformers import Trainer, TrainingArguments, set_seed
import torch
from torchinfo import summary
from functools import partial
from peft import LoraConfig, TaskType, get_peft_model
import json

from util import load_model
from gradepred_data import GradePredictionDataset, collate_fn

# -------- environment setting --------
load_dotenv()

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TORCH_ENABLE_DISTRIBUTED"] = "1"
os.environ["TORCH_DTENSOR_SKIP_CHECK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def evaluate_generate(
    model, tokenizer, dataset, collate_fn, batch_size: int = 4, device="cuda"
):
    from torch.utils.data import DataLoader
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            # gold ラベルは dataset 側から直接取得
            golds = [s["grades"] for s in dataset[total : total + len(ids)]]

            gen = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
            preds = [p.strip()[:1] for p in preds]   # 先頭 1 文字を抽出
            correct += sum(p == g for p, g in zip(preds, golds))
            total   += len(golds)

    acc = correct / total
    print(f"[eval] accuracy = {acc:.4f} ({correct}/{total})")
    return acc




# -------- main function --------
def main():
    import torch

    torch.compile = None
    if hasattr(torch, "compile"):
        torch.compile = lambda *args, **kwargs: args[0]

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
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size per device"
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

    # Set random seed for reproducibility
    set_seed(42)

    # DDP (Distributed Data Parallel) の設定
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    if ddp := world_size != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if ddp:
        print(f"[info] DDP is enabled (ddp = {ddp}, world_size = {world_size})")
    else:
        print(f"[info] DDP is disabled (ddp = {ddp}, world_size = {world_size})")

    # モデルとプロセッサのロード
    model, tokenizer = load_model(args.base_model)
    summary(model)

    # データセットを読み込む
    dataset_path = "../data/"


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

    print("[info] len(train_dataset):", len(train_dataset))
    print("[info] len(eval_dataset):", len(eval_dataset))

    custom_collate_fn = partial(
        collate_fn, tokenizer=tokenizer, max_tokens=args.max_words
    )
    print(
        f"[info] custom_collate_fn initialized with processor: {type(tokenizer).__name__}"
    )

    # LoRA設定
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
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print("✅ LoRA has been successfully applied to the model.")
    summary(model)

    # full_finetune_modules のモジュールを再度 trainable にする
    full_finetune_modules = ["embed_tokens", "lm_head"]

    print("🔧 Applying LoRA and enabling full finetune modules...")

    # LoRA 適用済み（前段）
    for name, param in model.named_parameters():
        if any(module_name in name for module_name in full_finetune_modules):
            param.requires_grad = True
            #! param.data = param.data.to(torch.float32)
            param.data = param.data.to(torch.float16)

    print("✅ LoRA has been applied.")
    print(
        f"✅ The following modules are fully finetuned: {', '.join(full_finetune_modules)}"
    )

    model.print_trainable_parameters()
    summary(model)

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.dtype)

    print("=" * 100)
    print("=" * 100)

    # TrainingArguments
    training_args = TrainingArguments(
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
        # dataloader_num_workers=4,
        remove_unused_columns=False,
        # report_to=None if args.report_to == "none" else args.report_to,
        run_name=args.run_name,
        save_total_limit=args.epochs,  # 2
        ddp_find_unused_parameters=True,
        # ddp_find_unused_parameters=False if ddp else None,
        load_best_model_at_end=False,
    )

    print("[info] Initialized TrainingArguments:")
    print(json.dumps(training_args.to_dict(), indent=2))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=custom_collate_fn,
    )

    print("[info] Trainer instance has been created.")
    print(
        f"[info] Trainer is set with model: {type(model).__name__}, train dataset size: {len(train_dataset)}, eval dataset size: {len(eval_dataset)}"
    )

    print("🔄 Starting training...")
    trainer.train()
    print("✅ Model training has been completed successfully!")

    ## rank 0のみが保存処理
    # if is_main_process():
    #    print("✅ Training finished. Starting model saving...")
    #    torch.distributed.barrier()  # 全rankの同期を明示的にとる
    #    merge_lora_and_save(model, processor, args.merged_output_dir)
    #    print("✅ Model saved successfully.")

    print("🎉 All steps completed successfully!")

    # evaluate
    # --- train() の直後に呼ぶ ---
    print("🔍  Running evaluation on validation split …")
    evaluate_generate(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        collate_fn=custom_collate_fn,
        device=model.device,
    )
    print("✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
