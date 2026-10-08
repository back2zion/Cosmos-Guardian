"""
Cosmos-Reason2-2B QLoRA SFT Training — RTX 3090 DDP version
For 2x RTX 3090 (24GB each) with construction safety dataset (71407)

Usage:
  DDP 2 GPUs:  torchrun --nproc_per_node 2 train_construction.py
  Single GPU:  CUDA_VISIBLE_DEVICES=0 python train_construction.py
  Resume:      torchrun --nproc_per_node 2 train_construction.py --resume
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

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA = SCRIPT_DIR / "fine_tuning_construction.json"
DEFAULT_IMAGE_DIR = SCRIPT_DIR.parent / "data" / "construction" / "images"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "outputs" / "cosmos-reason2-2b-construction-lora"

# Training hyperparams — tuned for DDP on 2x RTX 3090 (24GB each)
NUM_EPOCHS = 3
BATCH_SIZE = 1          # per GPU — effective batch = 1 * 2 GPUs * 4 accum = 8
GRAD_ACCUM = 4          # effective batch = BATCH_SIZE * num_gpus * GRAD_ACCUM
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.05
LORA_R = 32
LORA_ALPHA = 32
MAX_SEQ_LEN = 2048
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

            if not image_path:
                skipped += 1
                continue

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
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print("=" * 60)
    print("Cosmos-Reason2-2B QLoRA Training (RTX 3090 DDP)")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Data:   {args.data}")
    print(f"  Images: {args.image_dir}")
    print(f"  Output: {args.output}")
    print(f"  Rank:   {local_rank} / {world_size}")
    print(f"  GPU:    {torch.cuda.get_device_name(local_rank)} ({torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.1f} GB)")
    print(f"  Resume: {args.resume}")
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

    if len(train_dataset) == 0:
        print("ERROR: No samples loaded. Check data path and image directory.")
        return

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
        device_map={"": local_rank},      # each DDP rank gets its own GPU
        attn_implementation="sdpa",       # RTX 3090 compatible (no flash_attention_2)
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
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    reserved = round(torch.cuda.max_memory_reserved(local_rank) / 1024**3, 2)
    print(f"\n  GPU{local_rank} memory reserved: {reserved} GB")

    # 6. Train (with optional resume)
    print("\n" + "=" * 60)
    resume_from = None
    if args.resume:
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"Resuming from: {resume_from}")
        else:
            print("No checkpoint found, starting fresh.")

    print("Starting training...")
    print("=" * 60)
    stats = trainer.train(resume_from_checkpoint=resume_from)

    # 7. Save
    print("\nSaving LoRA adapters...")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    print(f"\nTraining complete!")
    print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s ({stats.metrics['train_runtime']/60:.1f} min)")
    print(f"  Loss: {stats.metrics.get('train_loss', 'N/A')}")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    main()
