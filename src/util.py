# =====================================================================
# util.py
# date: 2025/05/06
# description:
#   - 雑多な機能をまとめる
# 機能一覧；
# =====================================================================
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(base_model: str = "elyza/Llama-3-ELYZA-JP-8B", use_fast: bool = True, dtype=torch.float16):
    """
    モデルとプロセッサを読み込む（DDP対応）

    Parameters:
        base_model (str): モデル名またはローカルパス
        use_fast (bool): Fast tokenizer/image processor を使うかどうか
        dtype: モデルの重みのデータ型（例：torch.float16）

    Returns:
        model (AutoModelForCausalLM): 読み込まれたモデル
        processor (AutoProcessor): 読み込まれたプロセッサ
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    print(f"[info] Loading model from '{base_model}' to device {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map={"": device}
    )

    print(f"[info] Loading processor (use_fast={use_fast})...")
    # processor = AutoProcessor.from_pretrained(base_model, use_fast=use_fast)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=use_fast)

    if tokenizer.pad_token_id is None:
        #! どっちがいいかは要検討
        # tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = tokenizer.pad_token_id

        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    print("✅ Model and processor loaded successfully!")
    return model, tokenizer
