



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

def train():
    pass

def evaluate():
    pass

def custom_compute_metrics(res: EvalPrediction) -> Dict:
    pred = res.predictions.argmax(axis=1)
    target = res.label_ids
    pass


def main():
    # torchのバージョンを確認
    print(f"[info] torch version: {torch.__version__}")
    print(f"[info] torch cuda version: {torch.version.cuda}")

    #================================================================
    # パラメータの取得
    #================================================================

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

    # 独自collatorの定義
    custom_collate_fn = partial(
        collate_fn, tokenizer=tokenizer, max_tokens=args.max_words, testcase=True, question_filter=[1]
    )
    print(
        f"[info] custom_collate_fn initialized with processor: {type(tokenizer).__name__}"
    )

    # collatorの動作を確認
    example_loader = DataLoader(train_dataset, collate_fn=custom_collate_fn, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    for k,v in batch.items():
        print(k, v.shape)
    print(batch)


    #================================================================
    # 訓練用の設定
    #================================================================

    # ① LoRA設定
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

    model.gradient_checkpointing_enable()  #! 追加::model.gradient_checkpointing_enable() を LoRA 適用後に呼ぶと 20–30 % 追加節約
    # if local_rank == 0:
    #     model.print_trainable_parameters()
    #     summary(model, depth=2)

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.dtype)

    print("=" * 100)
    print("=" * 100)

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
        model=model,
        tokenizer=tokenizer,
        data_collator=custom_collate_fn,
        compute_metrics=custom_compute_metrics,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )




if __name__ == "__main__":
    main()
