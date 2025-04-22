import os
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from peft import PeftModel
from transformers import PreTrainedModel

#def load_model(base_model: str, use_fast: bool = True, dtype=torch.float16, device_map="auto"):
#    """
#    Parameters:
#        base_model (str): モデル名 or パス
#        use_fast (bool): Fast image processor を使うかどうか
#        dtype (torch.dtype): モデルの重みのデータ型（例：torch.float16）
#        device_map (str or dict): デバイスの割り当て方式（例："auto"）
#
#    Returns:
#        model (LlavaForConditionalGeneration): 読み込まれたモデル
#        processor (AutoProcessor): 読み込まれたプロセッサ
#    """
#    print(f"[info] Loading model from '{base_model}'...")
#    model = LlavaForConditionalGeneration.from_pretrained(
#        base_model,
#        torch_dtype=dtype,
#        device_map=device_map
#    )
#
#    print(f"[info] Loading processor (use_fast={use_fast})...")
#    processor = AutoProcessor.from_pretrained(base_model, use_fast=use_fast)
#
#    print("✅ Model and processor loaded successfully!")
#    return model, processor

def load_model(base_model: str, use_fast: bool = True, dtype = torch.float16):
    """
    モデルとプロセッサを読み込む（DDP対応）

    Parameters:
        base_model (str): モデル名またはローカルパス
        use_fast (bool): Fast tokenizer/image processor を使うかどうか
        dtype: モデルの重みのデータ型（例：torch.float16）

    Returns:
        model (LlavaForConditionalGeneration): 読み込まれたモデル
        processor (AutoProcessor): 読み込まれたプロセッサ
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    print(f"[info] Loading model from '{base_model}' to device {device}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map={"": device}
    )

    print(f"[info] Loading processor (use_fast={use_fast})...")
    processor = AutoProcessor.from_pretrained(base_model, use_fast=use_fast)

    print("✅ Model and processor loaded successfully!")
    return model, processor


def merge_and_save(base_model_name, lora_checkpoint, save_path):
    # ベースモデルとProcessorの読み込み
    base_model, processor = load_model(base_model_name)

    # LoRA重みの読み込みとマージ
    print("🔄 Loading and merging LoRA into base model...")
    model = PeftModel.from_pretrained(base_model, lora_checkpoint)
    model = model.merge_and_unload()

    # 保存先ディレクトリを作成
    os.makedirs(save_path, exist_ok=True)

    # モデルとプロセッサの保存
    print("💾 Saving merged model and processor to:", save_path)
    model.save_pretrained(save_path, safe_serialization=True)
    processor.save_pretrained(save_path)
    print("✅ Merged model and processor saved successfully!")

#def merge_lora_and_save(model, processor, save_dir):
#    if isinstance(model, PeftModel):
#        print("🔄 Merging LoRA weights into base model...")
#        model = model.merge_and_unload()  # LoRAをマージして戻す
#        base_model = model.base_model#.model  # 実体を取り出す
#    else:
#        base_model = model
#
#    os.makedirs(save_dir, exist_ok=True)
#    print(f"✅ Saving full model to: {save_dir}")
#    base_model.save_pretrained(save_dir, safe_serialization=True)
#    processor.save_pretrained(save_dir)
#    print("✅ Full model and processor saved.")

#def merge_lora_and_save(model, processor, save_dir):
#    # DDPで動作している場合、rank 0 のみで実行
#    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
#        return
#
#    if isinstance(model, PeftModel):
#        print("🔄 Merging LoRA weights into base model...")
#
#        model = model.to("cuda")
#        model.eval()
#
#        # LoRAのマージ
#        model = model.merge_and_unload()
#
#        # 元のベースモデルを取得
#        base_model = model.base_model.model if hasattr(model.base_model, "model") else model.base_model
#    else:
#        base_model = model
#
#    # 保存先ディレクトリ作成
#    os.makedirs(save_dir, exist_ok=True)
#    print(f"💾 Saving full model to: {save_dir}")
#
#    # モデルとプロセッサを保存
#    base_model.save_pretrained(save_dir, safe_serialization=True)
#    processor.save_pretrained(save_dir)
#
#    print("✅ Full model and processor saved.")


def merge_lora_and_save(model, processor, save_dir):
    # DDPで動作している場合、rank 0 のみで実行
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    if isinstance(model, PeftModel):
        print("🔄 Merging LoRA weights into base model...")

        model = model.to("cuda")
        model.eval()

        model = model.merge_and_unload()
        base_model = model.base_model.model if hasattr(model.base_model, "model") else model.base_model
    else:
        base_model = model
    
    print(f"[info] merge_lora_and_save  >> base_model type: {type(base_model)}")

    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Saving full model to: {save_dir}")

    # --- モデル保存 ---
    if isinstance(base_model, PreTrainedModel):
        base_model.save_pretrained(save_dir, safe_serialization=True)
    else:
        torch.save(base_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    # --- Processor保存 ---
    processor.save_pretrained(save_dir)

    print("✅ Full model and processor saved.")


def generate_captions(model, processor, image, device, max_new_tokens=128):
    chat = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Please describe this image."}
        ]
    }]

    inputs = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)
    #inputs["images"] = images.to(device, dtype=torch.float16)

    #for k in ["input_ids", "attention_mask"]:
    #    inputs[k] = inputs[k].to(device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)