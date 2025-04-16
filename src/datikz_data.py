import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


class DatikzCaptionDataset(Dataset):
    def __init__(self, split="train"):
        raw_dataset = load_dataset("HuggingFaceM4/datikz", split="train")
        self.dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)[split]
        print(f"✅ DatikzCaptionDataset initialized with {len(self.dataset)} samples for split '{split}'")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        caption = item["caption"].replace("<image>", "")  # 念のため除去
    
        return {
            "image": image,
            "caption": caption
        }
    
def collate_fn(batch, processor, max_words):
    conversations = []

    for item in batch:
        words = item["caption"].split()
        if len(words) > max_words:
            truncated_caption = " ".join(words[:max_words])
        else:
            truncated_caption = item["caption"]

        conversations.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": "Please describe this image."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": truncated_caption}
                ]
            }
        ])

    # batched apply_chat_template（画像含む）
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        #truncation=True,
        #max_length=max_length, 
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    # ラベル作成（pad部分を -100 に）
    labels = input_ids.clone()
    labels[input_ids == processor.tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values
    }

if __name__=='__main__':
    from functools import partial

    from torch.utils.data import DataLoader
    from transformers import AutoProcessor

    train_dataset = DatikzCaptionDataset(split="train")
    test_dataset = DatikzCaptionDataset(split="test")
    print("[info] len(train_dataset) : ", len(train_dataset))
    print("[info] len(test_dataset) : ", len(test_dataset))

    print(f"[debug] type(train_dataset[0]['image']) : f{type(train_dataset[0]['image'])}")
    print(f"[debug] train_dataset[0]['caption'] : f{train_dataset[0]['caption']}")

    # ========================================================================
    base_model = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=True)

    custom_collate_fn = partial(collate_fn, processor=processor, max_words=256)
    print(f"[info] custom_collate_fn initialized with processor: {type(processor).__name__}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    batch = next(iter(train_loader))
    for key, value in batch.items():
        print(f"[debug] {key}: shape = {value.shape}")

    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    # tqdmを使ってcaptionのトークン数を取得
    tokenizer = processor.tokenizer
    token_lengths = [
        len(tokenizer(example["caption"])["input_ids"])
        for example in tqdm(train_dataset, desc="Counting token lengths")
    ]
    
    # 統計量の計算
    avg_length = np.mean(token_lengths)
    min_length = np.min(token_lengths)
    max_length = np.max(token_lengths)
    
    # 結果を表示
    print(f"[debug] Caption length stats:")
    print(f"[debug]   Mean: {avg_length:.2f} characters")
    print(f"[debug]   Min : {min_length} characters")
    print(f"[debug]   Max : {max_length} characters")
    
    # ヒストグラムの描画
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=30, edgecolor='black')
    plt.title("Distribution of Caption Lengths (Character Count)")
    plt.xlabel("Number of Characters")
    plt.ylabel("Number of Captions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('caption_length.png')
    plt.show()

    