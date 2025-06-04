# =====================================================================
# train.py
# date: 2025/05/06
# description:
#   - LLMのLoRAチューニングを行う
#   - OOM対策として ZeRO Data Parallelism を採用
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
import numpy as np
from transformers import EvalPrediction
from util import load_model
from gradepred_data import GradePredictionDataset, collate_fn
from torch.nn.utils.rnn import pad_sequence
import evaluate
from torch.utils.data import DataLoader
# 使用VRAM数の推定

# from accelerate import estimate_memory

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

def evaluate_generate(
    model, tokenizer, dataset, collate_fn, batch_size: int = 4, device="cuda", log=False
):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, include_target=False),
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            # labels == -100 ならプロンプト、そうでないならターゲット
            # prompt_len = (labels == -100).sum(dim=1)
            prompt_len = (labels == -100).sum(dim=1).tolist()

            # # gold ラベルは dataset 側から直接取得
            # golds = [s["grades"] for s in dataset[total : total + len(ids)]]
            golds = batch["grades"]

            # # prompt の実長（パディングを除く）
            # prompt_lens = mask.sum(dim=1).tolist()

            # # バッチごとに可変長の prompt を生成へ
            # prompt_only = []
            # for seq, plen in zip(ids, prompt_len):
            #     prompt_only.append(seq[:plen])

            # prompt_only = pad_sequence(
            #     prompt_only, batch_first=True, padding_value=tokenizer.pad_token_id
            # )
            # attn_mask = (prompt_only != tokenizer.pad_token_id).long()

            #! ───── ② プロンプトだけを切り出して生成 ─────
            prompt_only = [seq[:p] for seq, p in zip(ids, prompt_len)]
            prompt_only = pad_sequence(
                prompt_only, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            attn_mask = (prompt_only != tokenizer.pad_token_id).long()
            #! ───── ② プロンプトだけを切り出して生成 ─────

            gen = model.generate(
                input_ids=prompt_only.to(device),
                attention_mask=attn_mask.to(device),
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                synced_gpus=False,
            )

            # ③ 生成部だけ取り出して decode
            # preds = []
            # for seq, plen in zip(gen, prompt_lens):
            #     gen_ids = seq[plen:]  # tensor(new_len)
            #     text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            #     # preds.append(text[:1])  # 先頭1文字で十分なら
            #     preds.append(text)

            #! ───── ③ 生成部分だけを decode ─────
            preds = [
                tokenizer.decode(seq[p:], skip_special_tokens=True).strip()[:50]
                for seq, p in zip(gen, prompt_len)
            ]
            #! ───── ③ 生成部分だけを decode ─────

            #! 出力を抽出するため，input_ids のlength以降を取得
            # preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
            # 入力文，出力文を確認
            # if log:
            #     print(
            #         "input_texts:",
            #         tokenizer.batch_decode(ids, skip_special_tokens=True),
            #     )
            #     print("generated_texts:", preds)
            #     print("true_labels:", golds)
            #     print()

            #! ───── ④ デバッグ出力（1 行ずつ対応させる） ─────
            for inp, pr, gd in zip(
                tokenizer.batch_decode(ids, skip_special_tokens=True),
                preds,
                golds,
            ):
                print(f"[input] {inp}\n[pred]  {pr}\n[gold]  {gd}\n")
            #! ───── ④ デバッグ出力（1 行ずつ対応させる） ─────

            #! ───── ⑤ 精度計算 ─────
            correct += sum(p.casefold() == g.casefold() for p, g in zip(preds, golds))
            total += len(golds)
            #! ───── ⑤ 精度計算 ─────

            # preds = [p.strip()[:1] for p in preds]  # 先頭 1 文字を抽出
            # correct += sum(p == g for p, g in zip(preds, golds))
            # correct += sum(p.casefold() == g.casefold() for p, g in zip(preds, golds))
            # total += len(golds)

    acc = correct / total
    print(f"[eval] accuracy = {acc:.4f} ({correct}/{total})")
    return acc


def compute_exact_match_and_f1(predictions, labels):
    """
    Args:
        predictions: torch.Tensor | np.ndarray
            - logits (batch, num_classes) か
            - 既に argmax 済みの (batch,) いずれか
        labels: 同上  (batch,)
    Returns:
        dict: {"exact_match": float, "precision": float, "recall": float, "f1": float}
    """

    # ---- 1. Tensor → ndarray、logits → argmax ----------------------------
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()

    if predictions.ndim > 1:                # (batch, num_classes) のとき
        predictions = predictions.argmax(-1)

    predictions = np.asarray(predictions).reshape(-1)  # (batch,)
    labels       = np.asarray(labels).reshape(-1)

    # ---- 2. メトリクス計算 ----------------------------------------------
    exact_match = float((predictions == labels).mean())

    # 単語分類が 1 クラス問題なら precision/recall = exact_match
    # 多クラスで macro F1 を取りたい場合は sklearn を使う方が楽
    precision = exact_match
    recall    = exact_match
    f1        = exact_match

    return {
        "exact_match": exact_match,
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
    }


# Trainer に渡す compute_metrics 関数
def compute_metrics_for_single_word_task(eval_pred: EvalPrediction):
    """
    Trainer の compute_metrics 引数に渡す関数。
    単一単語出力タスクの評価メトリクス（Exact Match, F1-score）を計算します。
    """
    predictions = eval_pred.predictions # ロジットまたは予測ID
    labels = eval_pred.label_ids      # 正解ラベルID

    # 上記で定義したカスタム関数を呼び出す
    metrics = compute_exact_match_and_f1(predictions, labels)

    return metrics


# -------- main function --------
def main():
    import torch

    # torchのバージョンを確認
    print(f"[info] torch version: {torch.__version__}")
    print(f"[info] torch cuda version: {torch.version.cuda}")

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

    # Set random seed for reproducibility
    set_seed(42)

    # DDP (Distributed Data Parallel) の設定
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

    # モデルとプロセッサのロード
    model, tokenizer = load_model(args.base_model, if_ZeRO=True)
    summary(model)

    # データセットを読み込む
    dataset_path = "./data/"

    train_dataset = GradePredictionDataset(
        dataset_path=dataset_path,
        question_filter=[1],
        concatenate=True,
        mode="train",
        testcase=True,
    )
    eval_dataset = GradePredictionDataset(
        dataset_path=dataset_path,
        question_filter=[1],
        concatenate=True,
        mode="valid",
        testcase=True,
    )

    print("[info] len(train_dataset):", len(train_dataset))
    print("[info] len(eval_dataset):", len(eval_dataset))

    custom_collate_fn = partial(
        collate_fn, tokenizer=tokenizer, max_tokens=args.max_words, testcase=True, question_filter=[1]
    )
    print(
        f"[info] custom_collate_fn initialized with processor: {type(tokenizer).__name__}"
    )

    # collate_fn の確認
    print("[info] Checking custom_collate_fn with a sample batch...")
    sample_batch = train_dataset[:2]  # 最初の2サンプルを取得
    collated_batch = custom_collate_fn(sample_batch)
    
    print("[info] Sample collated batch(decoded):")
    # ── ① 入力文の decode ─────────────────────────────
    decoded_inputs = tokenizer.batch_decode(
        collated_batch["input_ids"][:5],          # 先頭 5 個
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # ── ② labels の -100 を pad_token_id に置換してから decode ──
    labels_fixed = collated_batch["labels"][:5].clone()
    labels_fixed[labels_fixed == -100] = tokenizer.pad_token_id

    decoded_labels = tokenizer.batch_decode(
        labels_fixed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # ── ③ 表示 ───────────────────────────────────────
    for i, (inp, lab) in enumerate(zip(decoded_inputs, decoded_labels)):
        print(f"[{i}] input : {inp}")
        print(f"    label : {lab}")
        

    # LoRA設定
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            # "q_proj",
            # "k_proj",
            # "v_proj",
            # "o_proj",       # Self-Attention系
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP（FFN）系
        ],
    )

    # LoRA適用
    model.enable_input_require_grads()  #! 追加::入力テンソルに勾配を流せる状態を強制する安全スイッチ
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
            param.requires_grad = False  #! True -> 変更
            # param.data = param.data.to(torch.float32)
            param.data = param.data.to(torch.float16)

    print("✅ LoRA has been applied.")
    print(
        f"✅ The following modules are fully finetuned: {', '.join(full_finetune_modules)}"
    )

    model.print_trainable_parameters()
    summary(model)
    # 勾配チェックポイント＋rank 0 限定 summary
    model.gradient_checkpointing_enable()  #! 追加::model.gradient_checkpointing_enable() を LoRA 適用後に呼ぶと 20–30 % 追加節約
    # if local_rank == 0:
    #     model.print_trainable_parameters()
    #     summary(model, depth=2)

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.dtype)


    # TrainingArguments
    training_args = TrainingArguments(
        # ---
        gradient_checkpointing=True,
        # ---
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        logging_dir="./logs",
        logging_steps=50,
        lr_scheduler_type="cosine",
        optim="adamw_torch",  # "adamw_bnb_8bit",            # Adam 状態を 75% 圧縮  #! ZeROだと無効化されるっぽい
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        fp16_full_eval=True,  # eval も半精度(2023.05.18)
        #!
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        #!
        # dataloader_num_workers=4,
        remove_unused_columns=False,
        # report_to=None if args.report_to == "none" else args.report_to,
        run_name=args.run_name,
        save_total_limit=args.epochs,  # 2
        ddp_find_unused_parameters=False,  # True,  #! もしかしたら消した方がいいかも
        # ddp_find_unused_parameters=False if ddp else None,
        load_best_model_at_end=False,
        # deepspeed="ds_config_zero3.json",   #! 追加(ZeRO) -> ymalでCMDから指定したので再削除(2025.05.18)
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
        compute_metrics=compute_metrics_for_single_word_task,
    )

    print("[info] Trainer instance has been created.")
    print(
        f"[info] Trainer is set with model: {type(model).__name__}, train dataset size: {len(train_dataset)}, eval dataset size: {len(eval_dataset)}"
    )
    
    print("=" * 100)
    print("=" * 100)

    print("🔍  Running evaluation on vanilla model …")
    # before_train_results = trainer.evaluate()
    evaluate_generate(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        collate_fn=custom_collate_fn,
        device=model.device,
        batch_size=args.micro_batch_size,
        log=True
    )
    print("✅ Evaluation completed successfully!")
    # print("Evaluation result before training:", before_train_results)
    # torch.cuda.empty_cache() 
    print("=" * 100)
    print("=" * 100)

    print("🔄 Starting training...")
    trainer.train()
    print("✅ Model training has been completed successfully!")

    # rank 0のみが保存処理
    if is_main_process():
       print("✅ Training finished. Starting model saving...")
       torch.distributed.barrier()  # 全rankの同期を明示的にとる
       merge_lora_and_save(model, processor, args.merged_output_dir)
       print("✅ Model saved successfully.")

    print("🎉 All steps completed successfully!")

    # evaluate
    # --- train() の直後に呼ぶ ---
    print("🔍  Running evaluation on validation split …")
    # evaluate_generate(
    #     model=model,
    #     tokenizer=tokenizer,
    #     dataset=eval_dataset,
    #     collate_fn=custom_collate_fn,
    #     device=model.device,
    #     batch_size=args.micro_batch_size,
    #     log=True,
    # )
    
    
    # eval_results = trainer.evaluate()
    # print("✅ Evaluation completed successfully!")
    # print("Evaluation result:", eval_results)
    evaluate_generate(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        collate_fn=custom_collate_fn,
        device=model.device,
        batch_size=args.micro_batch_size,
        log=True
    )



if __name__ == "__main__":
    main()
