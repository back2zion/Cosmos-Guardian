"""
Cosmos Guardian Benchmark: Zero-shot vs Fine-tuned (LoRA)
Compares base Cosmos-Reason2-2B against LoRA-adapted model
on the AI Hub industrial safety dataset.

Metrics:
  - Hazard Detection Rate: Does the model detect any hazard?
  - Event Classification Accuracy: Does hazard type match ground truth?
  - Reasoning Keyword Score: length and physics-keyword density of reasoning text.
    This is a rough proxy, not a measure of reasoning correctness.
  - Inference Latency: Time per sample

Usage:
  CUDA_VISIBLE_DEVICES=0 python benchmark.py
  CUDA_VISIBLE_DEVICES=0 python benchmark.py --max-samples 20
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
import transformers
from PIL import Image

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SAMPLE_DIR = PROJECT_ROOT / "Sample"
LABEL_DIR = SAMPLE_DIR / "02.라벨링데이터"
MEDIA_DIR = SAMPLE_DIR / "01.원천데이터"
ADAPTER_PATH = SCRIPT_DIR.parent / "outputs" / "cosmos-reason2-2b-safety-lora"
RESULTS_DIR = SCRIPT_DIR.parent / "outputs" / "benchmark_results"
MODEL_NAME = "nvidia/Cosmos-Reason2-2B"

# Physics keywords for reasoning quality scoring
PHYSICS_KEYWORDS = [
    "kinetic", "energy", "momentum", "force", "velocity", "inertia",
    "compression", "impact", "trajectory", "gravity", "friction",
    "acceleration", "mass", "torque", "pressure", "entrapment",
    "clearance", "collision", "crushing", "shear",
]

# Event type synonyms for fuzzy matching
EVENT_SYNONYMS = {
    "jam": ["jam", "entrap", "crush", "compress", "pinch", "caught", "stuck", "squeeze"],
    "fall": ["fall", "drop", "gravity", "height", "slip", "trip"],
    "hit": ["hit", "strike", "impact", "collision", "projectile", "thrown"],
    "rule-violation": ["ppe", "violation", "helmet", "vest", "protective", "compliance", "equipment"],
}


# ============================================================
# Collect test samples from labels
# ============================================================
def load_training_ids() -> set[str]:
    """IDs used in the fine-tuning JSON files, so evaluation can exclude them."""
    ids = set()
    for name in ("fine_tuning_data.json", "fine_tuning_construction.json"):
        path = SCRIPT_DIR / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                ids.update(str(item.get("id", "")) for item in json.load(f))
    ids.discard("")
    return ids


def collect_samples(max_samples: int = None, exclude_ids: set[str] = frozenset()) -> list[dict]:
    samples = []
    for label_path in sorted(LABEL_DIR.rglob("*.json")):
        if label_path.stem in exclude_ids:
            continue
        try:
            with open(label_path, "r", encoding="utf-8-sig") as f:
                label = json.load(f)
        except Exception:
            continue

        meta = label.get("meta_information", {})
        event = meta.get("event", "unknown")
        filename = meta.get("file_name", "")

        # Find matching media
        media_matches = list(MEDIA_DIR.rglob(filename))
        if not media_matches:
            continue

        samples.append({
            "id": label_path.stem,
            "media_path": str(media_matches[0]),
            "event": event,
            "category": meta.get("category", ""),
            "facility": label.get("object_information_facility", {}).get("facility_name", ""),
            "environment": meta.get("environment", ""),
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples


# ============================================================
# Load model (base or with LoRA)
# ============================================================
def load_model(adapter_path: str = None):
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=compute_dtype,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  LoRA adapter loaded from {adapter_path}")

    processor = transformers.AutoProcessor.from_pretrained(MODEL_NAME)
    processor.image_processor.min_pixels = 224 * 224
    processor.image_processor.max_pixels = 448 * 448

    return model, processor


# ============================================================
# Run inference on a single sample
# ============================================================
def run_inference(model, processor, media_path: str, event: str) -> dict:
    is_video = Path(media_path).suffix.lower() in (".mp4", ".avi", ".mov")

    # Build content
    if is_video:
        import cv2
        cap = cv2.VideoCapture(media_path)
        frames = []
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = [int(i * total / 4) for i in range(4)]
            for i in range(total):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in indices:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb).resize((336, 336)))
            cap.release()
        media_content = {"type": "video", "video": frames, "fps": 1.0}
    else:
        media_content = {"type": "image", "image": media_path}

    prompt = (
        "Analyze the physical safety of this industrial environment. "
        "Focus on the PHYSICAL INTERACTION between machines and humans. "
        "Return a JSON with hazards_detected (type, severity, description, reasoning), "
        "overall_safety_score (0-100), and recommendation."
    )

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": (
                "You are a physical safety reasoning agent. "
                "Detect hazards and explain them with physics-based reasoning."
            )}],
        },
        {
            "role": "user",
            "content": [media_content, {"type": "text", "text": prompt}],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.1,
            use_cache=True,
        )
    latency = time.time() - start

    output_text = processor.tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return {"text": output_text, "latency": latency}


# ============================================================
# Scoring functions
# ============================================================
def parse_json_output(text: str) -> dict:
    """Try to extract JSON from model output."""
    start = text.find("{")
    if start == -1:
        return {}
    clean = text[start:]
    # Fix unclosed brackets
    open_b = clean.count("{") - clean.count("}")
    open_s = clean.count("[") - clean.count("]")
    if open_b > 0:
        clean += "}" * open_b
    if open_s > 0:
        clean += "]" * open_s
    try:
        return json.loads(clean)
    except Exception:
        # Try up to last }
        end = clean.rfind("}")
        if end != -1:
            try:
                return json.loads(clean[:end + 1])
            except Exception:
                pass
    return {}


def extract_hazards(parsed: dict) -> list[dict]:
    """Extract hazard list from various JSON schemas the models might produce."""
    # Try multiple possible keys
    for key in ["hazards_detected", "details", "hazards", "findings", "risks"]:
        val = parsed.get(key, [])
        if isinstance(val, list) and len(val) > 0:
            return val
    return []


def extract_all_text(parsed: dict) -> str:
    """Extract all text content from parsed JSON for keyword matching."""
    texts = []
    for key, val in parsed.items():
        if isinstance(val, str):
            texts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    texts.extend(str(v) for v in item.values())
                elif isinstance(item, str):
                    texts.append(item)
    return " ".join(texts).lower()


def score_hazard_detection(parsed: dict) -> bool:
    """Did the model detect any hazard?"""
    hazards = extract_hazards(parsed)
    if hazards:
        return True
    # Also check if there's an "analysis" field mentioning risk
    all_text = extract_all_text(parsed)
    risk_words = ["risk", "hazard", "danger", "unsafe", "violation", "collision", "entrap"]
    return any(w in all_text for w in risk_words)


def score_event_match(parsed: dict, gt_event: str) -> bool:
    """Does any detected hazard type match the ground truth event?"""
    synonyms = EVENT_SYNONYMS.get(gt_event, [gt_event])
    all_text = extract_all_text(parsed)
    for syn in synonyms:
        if syn in all_text:
            return True
    return False


def score_reasoning_quality(parsed: dict) -> float:
    """Keyword score 0-100 from reasoning length and physics keywords.

    A proxy only: it rewards long, keyword-rich text and does not check correctness.
    """
    hazards = extract_hazards(parsed)
    all_text = extract_all_text(parsed)

    if not hazards and not all_text:
        return 0.0

    if hazards:
        # Score per-hazard reasoning
        total_score = 0.0
        for h in hazards:
            reasoning = h.get("reasoning", "") + " " + h.get("description", "")
            length_score = min(len(reasoning) / 200.0, 1.0) * 50
            reasoning_lower = reasoning.lower()
            keyword_hits = sum(1 for kw in PHYSICS_KEYWORDS if kw in reasoning_lower)
            keyword_score = min(keyword_hits / 5.0, 1.0) * 50
            total_score += length_score + keyword_score
        return round(total_score / len(hazards), 1)
    else:
        # Score overall text
        length_score = min(len(all_text) / 500.0, 1.0) * 50
        keyword_hits = sum(1 for kw in PHYSICS_KEYWORDS if kw in all_text)
        keyword_score = min(keyword_hits / 5.0, 1.0) * 50
        return round(length_score + keyword_score, 1)


def score_safety_score_quality(parsed: dict) -> bool:
    """Did the model provide a valid safety score?"""
    for key in ["overall_safety_score", "safety_score", "score", "risk_score"]:
        score = parsed.get(key)
        if score is not None:
            try:
                s = int(score)
                return 0 <= s <= 100
            except (ValueError, TypeError):
                continue
    return False


# ============================================================
# Main benchmark
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Cosmos Guardian Benchmark")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also evaluate samples that appear in the fine-tuning data (results will be inflated)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect samples
    print("Collecting test samples...")
    exclude = set() if args.include_train else load_training_ids()
    if exclude:
        print(f"  Excluding {len(exclude)} IDs that appear in the fine-tuning data")
    samples = collect_samples(args.max_samples, exclude)
    print(f"  Found {len(samples)} held-out samples")
    if not samples:
        print("No samples found. Check Sample/ directory.")
        return

    # Event distribution
    events = {}
    for s in samples:
        events[s["event"]] = events.get(s["event"], 0) + 1
    print(f"  Events: {events}")

    results = {"base": [], "lora": []}

    for mode in ["base", "lora"]:
        adapter = str(ADAPTER_PATH) if mode == "lora" else None
        label = "LoRA Fine-tuned" if mode == "lora" else "Zero-shot (Base)"

        print(f"\n{'='*60}")
        print(f"  Running: {label}")
        print(f"{'='*60}")

        model, processor = load_model(adapter_path=adapter)

        for i, sample in enumerate(samples):
            sid = sample["id"]
            print(f"  [{i+1}/{len(samples)}] {sid} (event={sample['event']})", end=" ")

            try:
                output = run_inference(model, processor, sample["media_path"], sample["event"])
                parsed = parse_json_output(output["text"])

                detected = score_hazard_detection(parsed)
                matched = score_event_match(parsed, sample["event"])
                reasoning_q = score_reasoning_quality(parsed)
                valid_score = score_safety_score_quality(parsed)

                result = {
                    "id": sid,
                    "event": sample["event"],
                    "facility": sample["facility"],
                    "hazard_detected": detected,
                    "event_matched": matched,
                    "reasoning_quality": reasoning_q,
                    "valid_safety_score": valid_score,
                    "latency": round(output["latency"], 2),
                    "output_length": len(output["text"]),
                    "raw_output": output["text"],
                }
                results[mode].append(result)
                print(f"-> det={detected} match={matched} quality={reasoning_q} latency={output['latency']:.1f}s")

            except Exception as e:
                print(f"-> ERROR: {e}")
                results[mode].append({
                    "id": sid, "event": sample["event"],
                    "hazard_detected": False, "event_matched": False,
                    "reasoning_quality": 0, "valid_safety_score": False,
                    "latency": 0, "error": str(e),
                })

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # ============================================================
    # Compute aggregate metrics
    # ============================================================
    print(f"\n{'='*60}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*60}\n")

    summary = {}
    for mode in ["base", "lora"]:
        r = results[mode]
        n = len(r)
        if n == 0:
            continue

        detection_rate = sum(1 for x in r if x["hazard_detected"]) / n * 100
        match_rate = sum(1 for x in r if x["event_matched"]) / n * 100
        avg_reasoning = sum(x["reasoning_quality"] for x in r) / n
        valid_scores = sum(1 for x in r if x["valid_safety_score"]) / n * 100
        avg_latency = sum(x["latency"] for x in r) / n
        avg_output_len = sum(x["output_length"] for x in r) / n

        label = "LoRA Fine-tuned" if mode == "lora" else "Zero-shot (Base)"
        summary[mode] = {
            "label": label,
            "samples": n,
            "hazard_detection_rate": round(detection_rate, 1),
            "event_classification_acc": round(match_rate, 1),
            "avg_reasoning_quality": round(avg_reasoning, 1),
            "valid_safety_score_rate": round(valid_scores, 1),
            "avg_latency_sec": round(avg_latency, 2),
            "avg_output_length": round(avg_output_len),
        }

    # Print comparison table
    header = f"{'Metric':<30} {'Zero-shot':>12} {'Fine-tuned':>12} {'Delta':>10}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("Hazard Detection Rate (%)", "hazard_detection_rate"),
        ("Event Classification Acc (%)", "event_classification_acc"),
        ("Reasoning Keyword Score (0-100)", "avg_reasoning_quality"),
        ("Valid Safety Score Rate (%)", "valid_safety_score_rate"),
        ("Avg Latency (sec)", "avg_latency_sec"),
        ("Avg Output Length (chars)", "avg_output_length"),
    ]

    for display_name, key in metrics:
        base_val = summary.get("base", {}).get(key, 0)
        lora_val = summary.get("lora", {}).get(key, 0)
        delta = lora_val - base_val
        sign = "+" if delta > 0 else ""
        print(f"{display_name:<30} {base_val:>12} {lora_val:>12} {sign}{delta:>9.1f}")

    # Save results
    output_file = RESULTS_DIR / "benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "setup": {
                    "n_samples": len(samples),
                    "excluded_training_ids": len(exclude),
                    "includes_training_samples": args.include_train,
                },
                "details": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nFull results saved to: {output_file}")

    # Save markdown table for README
    md_file = RESULTS_DIR / "benchmark_table.md"
    with open(md_file, "w") as f:
        f.write("## Benchmark: Zero-shot vs Fine-tuned (LoRA)\n\n")
        f.write(f"**Dataset:** AI Hub 078 Industrial Safety ({summary.get('base', {}).get('samples', 0)} samples)\n\n")
        f.write("| Metric | Zero-shot | Fine-tuned | Delta |\n")
        f.write("|--------|-----------|------------|-------|\n")
        for display_name, key in metrics:
            base_val = summary.get("base", {}).get(key, 0)
            lora_val = summary.get("lora", {}).get(key, 0)
            delta = lora_val - base_val
            sign = "+" if delta > 0 else ""
            f.write(f"| {display_name} | {base_val} | {lora_val} | {sign}{delta:.1f} |\n")
    print(f"Markdown table saved to: {md_file}")


if __name__ == "__main__":
    main()
