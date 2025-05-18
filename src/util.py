# =====================================================================
# util.py
# date: 2025/05/06
# description:
#   - 雑多な機能をまとめる
# 機能一覧；
# =====================================================================
import os
import pprint, torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import deepspeed
from accelerate.utils import compute_module_sizes, init_empty_weights

def load_model(base_model: str = "elyza/Llama-3-ELYZA-JP-8B", use_fast: bool = True, dtype=torch.float16, if_ZeRO: bool = False):
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
    torch.cuda.set_device(local_rank) #!<追加> ★ rankごとのGPUを固定
    device = f"cuda:{local_rank}"

    # モデルの使用メモリを算出
    cfg  = AutoConfig.from_pretrained("elyza/Llama-3-ELYZA-JP-8B")

    with init_empty_weights():                     # まだ重みをダウンロードしない
        m = AutoModelForCausalLM.from_config(cfg)

    sizes = compute_module_sizes(m, dtype="float16")   # 各サブモジュールのバイト数
    total_MB = sum(s["param_bytes"] for s in sizes.values()) / 2**20
    pprint.pprint(sizes["model.embed_tokens"])         # 個別層のサイズ
    print(f"TOTAL: {total_MB:.1f} MB")

    #! --- 追加 ---
    # if if_ZeRO:
    #     print(f"[info] Loading model from '{base_model}' to device {device} with ZeRO optimization...")
    #     with deepspeed.zero.Init(
    #                 enabled=True, remote_device=f"cuda:{local_rank}", pin_memory=True
    #     ):
    #         model = AutoModelForCausalLM.from_pretrained(
    #             base_model,
    #             torch_dtype=dtype,
    #             low_cpu_mem_usage=True,      # ★ metaテンソルをやめる
    #             device_map={"": "meta"} #{"": f"cuda:{local_rank}"},      # ★ rankごとのGPUを固定
    #             # trust_remote_code=True,
    #             )
    # #! --- 追加 ---
    # else:
    #     print(f"[info] Loading model from '{base_model}' to device {device}...")
    #     model = AutoModelForCausalLM.from_pretrained(
    #         base_model, torch_dtype=dtype, device_map={"": device}
    #     )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,         # meta テンソルで OK
        device_map={"": "meta"},        # ←★ 重要：Accelerate が GPU に割当
        trust_remote_code=True,
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
