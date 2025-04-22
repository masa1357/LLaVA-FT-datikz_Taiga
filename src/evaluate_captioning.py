import os
import argparse
from tqdm import tqdm
import torch
import evaluate

import json

from PIL import Image

from datikz_data import DatikzCaptionDataset, collate_fn
from utils import load_model, generate_captions
from torch.utils.data import DataLoader
from functools import partial

from torchinfo import summary

from compute_cider import compute_cider_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to merged model directory")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_json", type=str, default="./results/debug.json", help="Path to save or load generated results")
    args = parser.parse_args()

    if os.path.exists(args.output_json):
        print(f"✅[info] Loading saved captions from {args.output_json}")
        with open(args.output_json, "r") as f:
            results = json.load(f)
        generated_captions = results["generated_captions"]
        references = results["references"]
    else:
        print(f"🔄[info] {args.output_json} not found. Generating captions now...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = load_model(args.model_path)
        model = model.to(device)
        model.eval()
    
        summary(model)

        print("[info] Generating captions...")
        test_dataset = DatikzCaptionDataset(split="test")
        generated_captions = []
        references = []

        for data in tqdm(test_dataset):
            image = data["image"]
            label = data["caption"]

            preds = generate_captions(model, processor, image, device)

            for pred in preds:
                generated_captions.append(pred)
                references.append([label])  # nested for multi-reference compatibility

        # 保存
        with open(args.output_json, "w") as f:
            json.dump({
                "generated_captions": generated_captions,
                "references": references
            }, f, indent=2, ensure_ascii=False)
        print(f"[info] Captions saved to {args.output_json}")


    # --- Metric Loading using evaluate ---
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    #cider = evaluate.load("cider")
    meteor = evaluate.load("meteor")

    # --- Metric Computation ---
    bleu_score = bleu.compute(predictions=generated_captions, references=references)
    rouge_score = rouge.compute(predictions=generated_captions, references=[ref[0] for ref in references])
    meteor_score = meteor.compute(predictions=generated_captions, references=[ref[0] for ref in references])
    #cider_score = cider.compute(predictions=generated_captions, references=references)
    #cider_score = compute_cider_score(generated_captions, references)

    # --- Output ---
    print("\n=== Evaluation Results ===")
    print(f"BLEU: {bleu_score['bleu']:.4f}")
    print(f"  BLEU-1: {bleu_score['precisions'][0]:.4f}")
    print(f"  BLEU-2: {bleu_score['precisions'][1]:.4f}")
    print(f"  BLEU-3: {bleu_score['precisions'][2]:.4f}")
    print(f"  BLEU-4: {bleu_score['precisions'][3]:.4f}")

    print("ROUGE:")
    for key, val in rouge_score.items():
        print(f"  {key}: {val:.4f}")
    print(f"METEOR: {meteor_score['meteor']:.4f}")
    #print(f"CIDEr: {cider_score:.4f}")


if __name__ == "__main__":
    main()
