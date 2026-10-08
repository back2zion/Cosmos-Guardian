"""
Cosmos-Reason2-2B QLoRA SFT Training Script
For RTX 3090 (24GB) — supports both images and videos

Usage:
  Single GPU:  CUDA_VISIBLE_DEVICES=1 python train_cosmos_reason2.py
  Both GPUs:   accelerate launch --num_processes 2 train_cosmos_reason2.py
"""

import json
import os
from pathlib import Path

import av
import numpy as np
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
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # NVIDIA_Cosmos_Cookoff/

MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
DATA_PATH = SCRIPT_DIR / "fine_tuning_data.json"
MEDIA_ROOT = PROJECT_ROOT / "Sample"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "cosmos-reason2-2b-safety-lora"

# Training hyperparams
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8  # effective batch = BATCH_SIZE * GRAD_ACCUM
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
LORA_R = 32
LORA_ALPHA = 32
MAX_SEQ_LEN = 4096
VIDEO_MAX_FRAMES = 4   # sample frames from each video (memory-friendly)
# Vision token budget — limit image resolution to control token count
# 28x28 pixels = 1 vision token; 512*28*28 ≈ 400K pixels → ~512 tokens per image
IMAGE_MIN_PIXELS = 128 * 28 * 28   # ~100K pixels
IMAGE_MAX_PIXELS = 512 * 28 * 28   # ~400K pixels


# ============================================================
# Build media lookup
# ============================================================
def build_media_map(root: Path) -> dict[str, Path]:
    media_map = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".mp4", ".mkv", ".avi", ".mov"):
            media_map[p.name] = p
    return media_map


# ============================================================
# Video frame extraction using pyav
# ============================================================
def extract_video_frames(video_path: str, max_frames: int = 8) -> list[Image.Image]:
    """Extract evenly-spaced frames from video using pyav."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames or 300  # fallback if metadata missing

    n_frames = min(max_frames, total_frames)
    indices = set(np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist())

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_image().convert("RGB"))
        if len(frames) >= n_frames:
            break

    container.close()
    return frames if frames else [Image.new("RGB", (224, 224))]  # fallback


# ============================================================
# Custom Dataset — supports images + videos
# ============================================================
class SafetyReasoningDataset(Dataset):
    def __init__(self, data_path: Path, media_map: dict[str, Path], processor):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.samples = []
        self.processor = processor
        skipped = 0

        for item in raw_data:
            media_file = item.get("video", "")
            media_path = media_map.get(media_file)
            if not media_path or not media_path.exists():
                skipped += 1
                continue

            is_video = media_path.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov")
            self.samples.append({
                "media_path": str(media_path),
                "is_video": is_video,
                "conversations": item["conversations"],
            })

        n_img = sum(1 for s in self.samples if not s["is_video"])
        n_vid = sum(1 for s in self.samples if s["is_video"])
        print(f"  Dataset: {len(self.samples)} samples ({n_img} images, {n_vid} videos, {skipped} skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        media_path = item["media_path"]
        is_video = item["is_video"]
        convs = item["conversations"]

        # Clean text
        user_text = convs[0]["value"]
        for tag in ("<video>\n", "<image>\n", "<video>", "<image>"):
            user_text = user_text.replace(tag, "")
        assistant_text = convs[1]["value"]

        if is_video:
            # Extract frames and treat as multiple images
            frames = extract_video_frames(media_path, VIDEO_MAX_FRAMES)
            # Build message with multiple image placeholders
            image_contents = [{"type": "image", "image": f"frame_{i}"} for i in range(len(frames))]
            user_content = image_contents + [{"type": "text", "text": user_text}]
            images = frames
        else:
            # Single image
            image = Image.open(media_path).convert("RGB")
            user_content = [
                {"type": "image", "image": media_path},
                {"type": "text", "text": user_text},
            ]
            images = [image]

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize with images only (no video kwarg — videos are pre-extracted as frames)
        inputs = self.processor(
            text=[text],
            images=images,
            padding=False,
            return_tensors="pt",
        )

        # Squeeze batch dim
        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.squeeze(0)
                # Truncate 1D tensors (input_ids, attention_mask) to MAX_SEQ_LEN
                if v.dim() == 1 and v.size(0) > MAX_SEQ_LEN:
                    v = v[:MAX_SEQ_LEN]
                result[k] = v

        # Labels = input_ids, pad tokens masked as -100
        labels = result["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        result["labels"] = labels

        return result


# ============================================================
# Custom data collator for variable-length vision tokens
# ============================================================
def collate_fn(batch):
    """Collator that handles variable-size tensors."""
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [item[key] for item in batch if item.get(key) is not None]
        if not values:
            continue

        if not isinstance(values[0], torch.Tensor):
            collated[key] = values
            continue

        # 1D tensors (input_ids, attention_mask, labels) — pad to max length
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
            # Multi-dim tensors (pixel_values, grid_thw) — concatenate
            try:
                collated[key] = torch.cat(values, dim=0)
            except RuntimeError:
                collated[key] = values[0]

    return collated


# ============================================================
# Main
# ============================================================
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print("=" * 60)
    print("Cosmos-Reason2-2B QLoRA SFT Training")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Data:   {DATA_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  GPUs:   {torch.cuda.device_count()}")
    print(f"  Rank:   {local_rank}")
    print("=" * 60)

    # 1. Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        min_pixels=IMAGE_MIN_PIXELS,
        max_pixels=IMAGE_MAX_PIXELS,
    )

    # 2. Build media map & dataset
    print("\nPreparing dataset...")
    media_map = build_media_map(MEDIA_ROOT)
    print(f"  Found {len(media_map)} media files")
    train_dataset = SafetyReasoningDataset(DATA_PATH, media_map, processor)

    # 3. Load model with QLoRA
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
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # 4. Prepare for LoRA
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Training args
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    # Memory report
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(local_rank)
        reserved = round(torch.cuda.max_memory_reserved(local_rank) / 1024**3, 2)
        print(f"\nGPU{local_rank}: {gpu.name} ({round(gpu.total_memory / 1024**3, 1)} GB)")
        print(f"Memory reserved after model load: {reserved} GB")

    # 7. Train!
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    stats = trainer.train()

    # 8. Save
    print("\nSaving LoRA adapters...")
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    print(f"\nTraining complete!")
    print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s ({stats.metrics['train_runtime']/60:.1f} min)")
    print(f"  Loss: {stats.metrics.get('train_loss', 'N/A')}")
    print(f"  Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
