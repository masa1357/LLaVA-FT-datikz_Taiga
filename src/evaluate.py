# =====================================================================
# evaluate.py
# date: 2025/06/15
# description:
#   - モデルの評価を行う
#   - trainと異なり，ファイルからモデルを読み込んで評価することで，バグを防止
# 対象モデル；
#   - elyza/Llama-3-ELYZA-JP-8B
# =====================================================================


def load_model():
    pass

def evaluate():
    pass

def main():
    # ? logger設定
    print("set logger")
    logger = set_logger(level=INFO)

    acc = Accelerator()
    if not acc.is_main_process:
        logger.setLevel(ERROR)
    logger.info(f"logger set complite")

    # ================================================================
    # パラメータの取得
    # ================================================================

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

    model, tokenizer = load_model(args.base_model_path, if_ZeRO=True)
    summary(model)
    tokenizer.padding_side = "left"

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

    train_logger = set_logger(name="CollateTrain", level=INFO)
    eval_logger = set_logger(name="CollateEval", level=INFO)

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
    # 評価用の設定
    # ================================================================


if __name__ == "__main__":
    main()
