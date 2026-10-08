
import os
import glob
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "training_config_3090.json")

# Load config
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODEL_ID = config["training_params"]["model_id"]
DATA_PATH = os.path.join(SCRIPT_DIR, config["data"]["json_path"])
IMAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, config["data"]["image_root"]))

# Scan for media files
print(f"Scanning media files in {IMAGE_ROOT}...")
MEDIA_MAP = {}
for root, dirs, files in os.walk(IMAGE_ROOT):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.mkv', '.avi', '.mov')):
            MEDIA_MAP[filename] = os.path.join(root, filename)
print(f"Found {len(MEDIA_MAP)} media files.")

class CosmosDataset(Dataset):
    def __init__(self, data_path, processor):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Resolve Media
        video_filename = item.get("video")
        media_path = MEDIA_MAP.get(video_filename) if video_filename else None
        
        messages = []
        visuals = []
        
        # 2. Build Messages
        # Support Qwen2-VL chat format
        # If media exists, it acts as a user input in the first turn
        if media_path:
            is_video = media_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))
            media_content = {"type": "video", "video": media_path} if is_video else {"type": "image", "image": media_path}
            visuals.append(media_path)
            
            # Find first user message and prepend media
            first_turn_handled = False
            for turn in item["conversations"]:
                role = "user" if turn["from"] == "human" else "assistant"
                content_text = turn["value"].replace("<video>\n", "").replace("<image>\n", "").replace("<video>", "").replace("<image>", "")
                
                if role == "user" and not first_turn_handled:
                    messages.append({
                        "role": "user",
                        "content": [media_content, {"type": "text", "text": content_text}]
                    })
                    first_turn_handled = True
                else:
                    messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": content_text}]
                    })
        else:
            # Text only (fallback)
            for turn in item["conversations"]:
                role = "user" if turn["from"] == "human" else "assistant"
                content_text = turn["value"]
                messages.append({
                    "role": role, 
                    "content": [{"type": "text", "text": content_text}]
                })

        # 3. Apply Template
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # 4. Prepare inputs
        # Filter visuals into images and videos lists
        images = [v for v in visuals if v.lower().endswith(('.png', '.jpg', '.jpeg'))]
        videos = [v for v in visuals if v.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))]
        
        inputs = self.processor(
            text=[text_prompt],
            images=images if images else None,
            videos=videos if videos else None,
            padding=True,
            return_tensors="pt"
        )
        
        # Unpack batch dimension since we are returning a single item
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs["pixel_values"][0] if "pixel_values" in inputs else None,
            "image_grid_thw": inputs["image_grid_thw"][0] if "image_grid_thw" in inputs else None,
            "pixel_values_videos": inputs["pixel_values_videos"][0] if "pixel_values_videos" in inputs else None,
            "video_grid_thw": inputs["video_grid_thw"][0] if "video_grid_thw" in inputs else None,
            "labels": inputs["input_ids"][0]
        }

def train():
    # Load Processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Load Dataset
    train_dataset = CosmosDataset(DATA_PATH, processor)
    
    # Load Model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # PEFT Config
    lora_config = LoraConfig(
        r=config["training_params"]["lora_r"],
        lora_alpha=config["training_params"]["lora_alpha"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM", 
        modules_to_save=None
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training Args
    training_args = TrainingArguments(
        output_dir="./outputs/cosmos_guardian_checkpoints",
        per_device_train_batch_size=config["training_params"]["batch_size"],
        gradient_accumulation_steps=config["training_params"]["gradient_accumulation_steps"],
        learning_rate=config["training_params"]["learning_rate"],
        num_train_epochs=config["training_params"]["epochs"],
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        report_to="none",
        remove_unused_columns=False
    )
    
    data_collator = DataCollatorForSeq2Seq(
        processor.tokenizer,
        pad_to_multiple_of=8,
        padding=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    print("Starting Training on 2x RTX 3090...")
    trainer.train()
    
    trainer.save_model("./outputs/cosmos_guardian_final")
    processor.save_pretrained("./outputs/cosmos_guardian_final")
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()
