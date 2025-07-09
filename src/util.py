# =====================================================================
# util.py
# date: 2025/05/06
# description:
#   - 雑多な機能をまとめる
# 機能一覧；
# =====================================================================
import os
import pprint, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from logging import (
    getLogger,
    StreamHandler,
    Formatter,
    INFO,
    DEBUG,
    ERROR,
    WARNING,
    FileHandler,
)
import numpy as np
import random
import time
from contextlib import contextmanager
import io
import sys

BYTES_PER_PARAM = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
}


def set_logger(name: str = __name__, level=INFO):
    """
    loggerの定義
    """
    logger = getLogger(name)
    logger.setLevel(level)

    # [INFO] ハンドラが既に追加されているかをチェック
    if not logger.hasHandlers():
        # ? 出力されるログの表示内容を定義
        formatter = Formatter(
            "%(asctime)s : %(name)s : %(levelname)s : %(lineno)s : %(message)s"
        )

        stream_handler = StreamHandler()
        # stream_handler.setStream(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))

        # ? 標準出力のhandlerをセット
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info("Test_message")

    return logger


# ? seedの固定
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 複数GPU対応
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ? 時間計測関数
@contextmanager
def timer(name: str):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.2f} s")


def load_model(
    base_model: str = "elyza/Llama-3-ELYZA-JP-8B",
    use_fast: bool = True,
    dtype=torch.float16,
    if_ZeRO: bool = False,
):
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
    torch.cuda.set_device(local_rank)  # <追加> rankごとのGPUを固定
    device = f"cuda:{local_rank}"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"[info] Loading processor (use_fast={use_fast})...")
    # processor = AutoProcessor.from_pretrained(base_model, use_fast=use_fast)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=use_fast,  # padding_side="left"
    )
    tokenizer.padding_side = "left"
    #! pad_token が無い場合だけ追加
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    #! 語彙サイズを合わせる
    model.resize_token_embeddings(len(tokenizer))

    # if tokenizer.pad_token_id is None:
    #     # どっちがいいかは要検討
    #     # tokenizer.pad_token = tokenizer.eos_token
    #     # model.config.pad_token_id = tokenizer.pad_token_id

    #     tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    #     model.resize_token_embeddings(len(tokenizer))

    print("✅ Model and processor loaded successfully!")
    return model, tokenizer


def estimate_vram_cost(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    dtype=torch.float16,
    optim_factor: float = 4,  # AdamW (fp32) なら 4 (= 1重み + 1勾配 + 2状態)
    act_factor: float = 1.3,  # 活性の再現用オーバーヘッド (経験則)
) -> dict:
    """
    LLM 1 GPU あたりの VRAM 使用量 (推定) を返す

    Returns
    -------
    dict : {
        "params_MB": int
        "grads_MB" : int
        "optimizer_MB": …,
        "activations_MB": …,
        "total_MB": …,
    }
    """
    bytes_per_param = BYTES_PER_PARAM[dtype]
    # ① パラメータ数
    param_count = sum(p.numel() for p in model.parameters())
    params_MB = param_count * bytes_per_param / (1024**2)

    # ② 勾配（パラメータと同サイズ／同 dtype）
    grads_MB = params_MB

    # ③ Optimizer 状態 (AdamW = fp32 × 2 倍)
    optimizer_MB = params_MB * (optim_factor - 2)  # 勾配+重みは除外済

    # ④ アクティベーション (おおよそ batch*seq*hidden*4bytes×layers×係数)
    hidden = model.config.hidden_size
    layers = model.config.num_hidden_layers
    acts_numel = batch_size * seq_len * hidden * layers * act_factor
    activations_MB = acts_numel * bytes_per_param / (1024**2)

    total_MB = params_MB + grads_MB + optimizer_MB + activations_MB
    return {
        "params_MB": params_MB,
        "grads_MB": grads_MB,
        "optimizer_MB": optimizer_MB,
        "activations_MB": activations_MB,
        "total_MB": total_MB,
    }
