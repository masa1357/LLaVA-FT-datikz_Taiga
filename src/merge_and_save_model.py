import os
import argparse
from transformers import AutoProcessor
from peft import PeftModel

from src.utils import merge_and_save

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model and save the result.")
    parser.add_argument(
        "--base_model_name", type=str, required=True,
        help="Base model name or path (e.g., 'llava-hf/llava-1.5-7b-hf')"
    )
    parser.add_argument(
        "--lora_checkpoint", type=str, required=True,
        help="Path to LoRA checkpoint directory (containing adapter_model.safetensors)"
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Directory to save the merged model and processor"
    )

    args = parser.parse_args()
    merge_and_save(args.base_model_name, args.lora_checkpoint, args.save_path)

if __name__ == "__main__":
    main()