"""
Cosmos-Reason2-2B QLoRA SFT Training — H100 version
For H100 NVL (30GB available) with construction safety dataset (71407)

Usage:
  CUDA_VISIBLE_DEVICES=1 python3.11 train_h100.py
  CUDA_VISIBLE_DEVICES=1 python3.11 train_h100.py --max-samples 2000
"""

import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "nvidia/Cosmos-Reason2-2B"

# Training hyperparams — tuned for 30GB VRAM
NUM_EPOCHS = 2
BATCH_SIZE = 2          # H100 can handle larger batch
GRAD_ACCUM = 4          # effective batch = 2 * 4 = 8
LEARNING_RATE = 1e-4    # slightly lower for larger dataset
WARMUP_RATIO = 0.05
LORA_R = 32
LORA_ALPHA = 32
MAX_SEQ_LEN = 2048      # shorter for efficiency with large dataset
IMAGE_MIN_PIXELS = 128 * 28 * 28
IMAGE_MAX_PIXELS = 512 * 28 * 28


# ============================================================
# Dataset
# ============================================================
class ConstructionSafetyDataset(Dataset):
    def __init__(self, data_path: str, image_dir: str, processor, max_samples: int = None):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if max_samples:
            raw_data = raw_data[:max_samples]

        # Build image lookup
        image_dir = Path(image_dir)
        self.image_map = {}
        if image_dir.exists():
            for p in image_dir.rglob("*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.image_map[p.stem] = str(p)

        self.samples = []
        self.processor = processor
        skipped = 0

        for item in raw_data:
            image_file = item.get("image", "")
            image_stem = Path(image_file).stem
            image_path = self.image_map.get(image_stem)

            # If no image, still train on text-only (labels-only mode)
            self.samples.append({
                "image_path": image_path,
                "conversations": item["conversations"],
                "event_type": item.get("event_type", "unknown"),
            })

        has_images = sum(1 for s in self.samples if s["image_path"])
        print(f"  Dataset: {len(self.samples)} samples ({has_images} with images, {skipped} skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        convs = item["conversations"]

        # Extract system, user, assistant messages
        system_text = ""
        user_text = ""
        assistant_text = ""
        for msg in convs:
            if msg["from"] == "system":
                system_text = msg["value"]
            elif msg["from"] == "human":
                user_text = msg["value"]
                for tag in ("<image>\n", "<image>"):
                    user_text = user_text.replace(tag, "")
            elif msg["from"] == "gpt":
                assistant_text = msg["value"]

        # Build messages
        messages = []
        if system_text:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})

        if item["image_path"] and os.path.exists(item["image_path"]):
            user_content = [
                {"type": "image", "image": item["image_path"]},
                {"type": "text", "text": user_text},
            ]
            images = [Image.open(item["image_path"]).convert("RGB")]
        else:
            user_content = [{"type": "text", "text": user_text}]
            images = None

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        if images:
            inputs = self.processor(text=[text], images=images, padding=False, return_tensors="pt")
        else:
            inputs = self.processor(text=[text], padding=False, return_tensors="pt")

        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.squeeze(0)
                if v.dim() == 1 and v.size(0) > MAX_SEQ_LEN:
                    v = v[:MAX_SEQ_LEN]
                result[k] = v

        labels = result["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        result["labels"] = labels

        return result


# ============================================================
# Collator
# ============================================================
def collate_fn(batch):
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [item[key] for item in batch if item.get(key) is not None]
        if not values:
            continue

        if not isinstance(values[0], torch.Tensor):
            collated[key] = values
            continue

        if values[0].dim() == 1:
            max_len = max(v.size(0) for v in values)
            pad_value = -100 if key == "labels" else 0
            padded = []
            for v in values:
                if v.size(0) < max_len:
                    v = torch.nn.functional.pad(v, (0, max_len - v.size(0)), value=pad_value)
                padded.append(v)
            collated[key] = torch.stack(padded)
        else:
            try:
                collated[key] = torch.cat(values, dim=0)
            except RuntimeError:
                collated[key] = values[0]

    return collated


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="fine_tuning_construction.json")
    parser.add_argument("--image-dir", type=str, default="extracted_images")
    parser.add_argument("--output", type=str, default="cosmos-reason2-2b-construction-lora")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print("=" * 60)
    print("Cosmos-Reason2-2B QLoRA Training (H100)")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Data:   {args.data}")
    print(f"  Images: {args.image_dir}")
    print(f"  Output: {args.output}")
    print(f"  GPU:    {torch.cuda.get_device_name(local_rank)}")
    print(f"  VRAM:   {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    # 1. Processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        min_pixels=IMAGE_MIN_PIXELS,
        max_pixels=IMAGE_MAX_PIXELS,
    )

    # 2. Dataset
    print("\nPreparing dataset...")
    train_dataset = ConstructionSafetyDataset(
        args.data, args.image_dir, processor, max_samples=args.max_samples
    )

    # 3. Model with QLoRA
    print("\nLoading model (4-bit quantization)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Training args
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
    )

    # 5. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    reserved = round(torch.cuda.max_memory_reserved(local_rank) / 1024**3, 2)
    print(f"\nMemory reserved after model load: {reserved} GB")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    stats = trainer.train()

    # 6. Save
    print("\nSaving LoRA adapters...")
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)

    print(f"\nTraining complete!")
    print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s ({stats.metrics['train_runtime']/60:.1f} min)")
    print(f"  Loss: {stats.metrics.get('train_loss', 'N/A')}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
