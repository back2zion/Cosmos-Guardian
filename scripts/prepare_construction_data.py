"""
Extract and convert AIHub 227 construction safety data for Cosmos Reason2 fine-tuning.
Extracts images + labels from ZIPs, then converts to training format.

Usage:
  python prepare_construction_data.py --max-per-type 200
  python prepare_construction_data.py --max-per-type 500 --workers 4
"""

import argparse
import json
import os
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ============================================================
# Paths
# ============================================================
DATA_ROOT = Path("/mnt/c/projects/227.건설 현장 위험 상태 판단 데이터/01-1.정식개방데이터")
TRAIN_IMG_DIR = DATA_ROOT / "Training" / "01.원천데이터"
TRAIN_LABEL_DIR = DATA_ROOT / "Training" / "02.라벨링데이터"

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "construction"
EXTRACTED_IMAGES = OUTPUT_DIR / "images"
EXTRACTED_LABELS = OUTPUT_DIR / "labels"
OUTPUT_JSON = SCRIPT_DIR / "fine_tuning_construction.json"

# ============================================================
# Type mappings (same as construction_to_cosmos.py)
# ============================================================
TYPE_MAP = {
    "추락": "fall",
    "낙하": "falling_object",
    "협착": "entrapment",
    "전도": "overturn",
    "화재": "fire",
}

TYPE_DESCRIPTIONS = {
    "fall": "Fall from height",
    "falling_object": "Falling object / dropped material",
    "entrapment": "Entrapment / caught-in-between",
    "overturn": "Equipment overturn / collapse",
    "fire": "Fire / thermal hazard",
}

# ============================================================
# Physics-based reasoning templates
# ============================================================
REASONING_TEMPLATES = {
    "fall": [
        "The worker is positioned at an elevated surface without adequate fall protection. "
        "Gravitational potential energy (E = mgh) means a fall from this height would result in "
        "significant kinetic energy at impact. At approximately {height:.0f}m, impact velocity would reach "
        "v = sqrt(2gh) ≈ {velocity:.1f} m/s, generating forces far exceeding human tissue tolerance.",
        "Height differential creates gravitational risk. Without proper harness attachment points, "
        "the worker's center of mass is unsupported. Any shift in balance transfers potential energy "
        "to kinetic energy, with deceleration trauma upon ground contact.",
        "Elevated work position without guardrails or personal fall arrest system. "
        "The acceleration due to gravity (9.81 m/s²) would cause rapid velocity increase during a fall, "
        "with the resulting impact force (F = ma) likely causing severe injury.",
    ],
    "falling_object": [
        "Unsecured materials at elevation pose a dropped-object hazard. "
        "An object falling from height converts potential energy to kinetic energy: "
        "E = mgh. Even a small tool (0.5 kg) dropped from {height:.0f}m reaches "
        "v = {velocity:.1f} m/s at impact, delivering dangerous momentum transfer to anyone below.",
        "Objects positioned without toe boards or netting can become projectiles under gravity. "
        "The impact force F = mv/Δt depends on the deceleration time — a hard helmet provides "
        "~10ms of cushioning, but forces can still exceed skull fracture thresholds.",
        "Material storage at height without containment barriers. Wind load or vibration could "
        "dislodge items, creating ballistic trajectories. Terminal velocity depends on drag coefficient, "
        "but construction materials typically reach dangerous speeds within 2-3 meters of free fall.",
    ],
    "entrapment": [
        "Worker proximity to moving mechanical parts creates a caught-in/between hazard. "
        "The crushing force of hydraulic or mechanical systems far exceeds human tissue tolerance. "
        "Compression forces can cause traumatic injury with pressures as low as 50 psi on soft tissue.",
        "Rotating or reciprocating equipment creates entanglement and entrapment zones. "
        "The angular momentum of the machinery means it cannot stop instantaneously — "
        "stored kinetic energy (E = ½Iω²) continues the motion even after power cutoff.",
        "Pinch point identified between moving and stationary components. "
        "The mechanical advantage of the system amplifies input force, "
        "creating compression pressures that can crush bone and tissue.",
    ],
    "overturn": [
        "Equipment stability is compromised — the center of gravity has shifted beyond the support base. "
        "Overturning moment M = F × d exceeds the restoring moment provided by the equipment's weight "
        "and base geometry. Ground conditions or load distribution may have altered the stability triangle.",
        "The equipment is operating on uneven terrain or with an asymmetric load, "
        "shifting the center of mass outside the stability envelope. "
        "Once tipping begins, gravitational torque accelerates the overturn.",
        "Structural instability detected — lateral forces or wind loads are creating "
        "a tipping moment that approaches the equipment's rollover threshold.",
    ],
    "fire": [
        "Thermal hazard conditions present — heat source proximity to combustible materials "
        "creates ignition risk. Fire propagation follows the fire triangle principle: "
        "fuel, oxygen, and heat source are all present in this scenario.",
        "Potential ignition sources detected near flammable materials. "
        "Radiant heat transfer (Q = σεAT⁴) from hot work operations can raise "
        "nearby material temperatures above their flash point.",
        "Welding/cutting operations without proper fire watch protocol. "
        "Sparks at temperatures exceeding 1500°C can ignite materials with "
        "flash points below 100°C.",
    ],
}

SAFE_REASONING = {
    "fall": "Workers are operating at height with proper fall protection systems visible — "
            "harnesses, guardrails, and secured platforms are in place.",
    "falling_object": "Material handling at height follows proper protocols — "
                      "toe boards, debris netting, and secured storage prevent dropped objects.",
    "entrapment": "Machine guarding and lockout/tagout procedures appear to be in effect. "
                  "Workers maintain safe distance from moving parts.",
    "overturn": "Equipment is operating within its stability envelope on level ground. "
                "Load is properly centered and within rated capacity.",
    "fire": "Hot work area is properly prepared — fire-resistant blankets cover nearby combustibles, "
            "fire extinguisher is accessible.",
}

RECOMMENDATIONS = {
    "fall": "Install guardrails and ensure all workers use personal fall arrest systems. "
            "Verify anchor points meet OSHA 5000-lb requirement.",
    "falling_object": "Secure all materials at height with toe boards and debris netting. "
                      "Establish and enforce hard-hat zones below overhead work.",
    "entrapment": "Implement lockout/tagout procedures. Install machine guards on all pinch points.",
    "overturn": "Verify ground conditions and load charts before operation. "
                "Deploy outriggers on level ground. Do not exceed rated capacity.",
    "fire": "Remove combustibles from hot work area (35-ft radius). "
            "Assign fire watch with extinguisher. Obtain hot work permit.",
}


def parse_zip_type(filename: str):
    """Extract accident type and normality from ZIP filename."""
    # e.g. TL_5대사고유형_낙하_비정상_N-11.zip
    for kr, en in TYPE_MAP.items():
        if kr in filename:
            is_hazardous = "비정상" in filename or "_N-" in filename
            return en, is_hazardous
    return None, None


def generate_conversation(label_data: dict, image_filename: str, event_type: str, is_hazardous: bool) -> dict:
    """Generate a training conversation from label data."""
    learning = label_data.get("Learning_Data_Info.", {})
    annotations = learning.get("Annotations", [])
    num_objects = len(annotations)

    height = random.uniform(3, 12)
    velocity = (2 * 9.81 * height) ** 0.5

    if is_hazardous:
        reasoning_pool = REASONING_TEMPLATES.get(event_type, REASONING_TEMPLATES["fall"])
        reasoning = random.choice(reasoning_pool).format(height=height, velocity=velocity)
        severity = random.choice(["High", "Critical"])
        safety_score = random.randint(10, 35)
        hazard_type = TYPE_DESCRIPTIONS.get(event_type, event_type)

        response = {
            "hazards_detected": [{
                "type": hazard_type,
                "severity": severity,
                "description": f"Construction site hazard: {hazard_type.lower()} risk identified. "
                               f"{num_objects} relevant objects/zones annotated in scene.",
                "reasoning": reasoning,
            }],
            "overall_safety_score": safety_score,
            "recommendation": RECOMMENDATIONS.get(event_type, "Review safety protocols."),
        }
    else:
        reasoning = SAFE_REASONING.get(event_type, "Safe working conditions observed.")
        safety_score = random.randint(70, 95)
        response = {
            "hazards_detected": [],
            "overall_safety_score": safety_score,
            "safety_observation": f"Construction site appears to follow proper "
                                  f"{TYPE_DESCRIPTIONS.get(event_type, '')} prevention protocols. {reasoning}",
            "recommendation": "Continue current safety practices. Maintain vigilance and regular safety audits.",
        }

    system_msg = (
        "You are a physical safety reasoning agent specialized in construction site hazard analysis. "
        "Detect hazards and explain them with physics-based reasoning. "
        "Focus on the five major construction accident types: falls, falling objects, "
        "entrapment, equipment overturn, and fire."
    )

    user_msg = (
        "Analyze the physical safety of this construction site image. "
        "Focus on identifying any of the 5 major construction accident types: "
        "fall from height, falling objects, entrapment/caught-in, equipment overturn, or fire hazard. "
        "Return a JSON with hazards_detected (type, severity, description, reasoning), "
        "overall_safety_score (0-100), and recommendation."
    )

    return {
        "id": learning.get("Json_Data_ID", ""),
        "image": image_filename,
        "event_type": event_type,
        "is_hazardous": is_hazardous,
        "conversations": [
            {"from": "system", "value": system_msg},
            {"from": "human", "value": f"<image>\n{user_msg}"},
            {"from": "gpt", "value": json.dumps(response, indent=2, ensure_ascii=False)},
        ],
    }


def extract_from_zip_pair(img_zip_path: str, label_zip_path: str, max_samples: int,
                          event_type: str, is_hazardous: bool) -> list:
    """Extract matching images and labels from a ZIP pair."""
    samples = []

    try:
        with zipfile.ZipFile(label_zip_path, 'r') as lz:
            label_names = [n for n in lz.namelist() if n.endswith('.json')]
            random.shuffle(label_names)
            label_names = label_names[:max_samples]

            # Build set of needed image stems
            needed_stems = set()
            label_map = {}
            for ln in label_names:
                stem = Path(ln).stem
                needed_stems.add(stem)
                with lz.open(ln) as f:
                    try:
                        label_map[stem] = json.loads(f.read().decode('utf-8-sig'))
                    except Exception:
                        continue

        with zipfile.ZipFile(img_zip_path, 'r') as iz:
            img_names = {Path(n).stem: n for n in iz.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))}

            for stem in needed_stems:
                if stem not in img_names or stem not in label_map:
                    continue

                # Extract image
                img_name = img_names[stem]
                out_img = EXTRACTED_IMAGES / f"{stem}.jpg"
                if not out_img.exists():
                    iz.extract(img_name, str(EXTRACTED_IMAGES))
                    # Move from nested path to flat
                    extracted = EXTRACTED_IMAGES / img_name.lstrip('/')
                    if extracted.exists() and extracted != out_img:
                        extracted.rename(out_img)

                # Generate conversation
                conv = generate_conversation(label_map[stem], f"{stem}.jpg", event_type, is_hazardous)
                samples.append(conv)

    except Exception as e:
        print(f"  Error processing {Path(img_zip_path).name}: {e}")

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-type", type=int, default=200,
                        help="Max samples per accident type per normality (default: 200)")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print("Construction Safety Data Preparation")
    print(f"  Source: {DATA_ROOT}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Max per type: {args.max_per_type}")
    print("=" * 60)

    EXTRACTED_IMAGES.mkdir(parents=True, exist_ok=True)
    EXTRACTED_LABELS.mkdir(parents=True, exist_ok=True)

    # Find all ZIP pairs
    img_zips = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.zip')])
    label_zips = sorted([f for f in os.listdir(TRAIN_LABEL_DIR) if f.endswith('.zip')])

    # Match by type pattern
    # TS_5대사고유형_낙하_비정상_N-11.zip → TL_5대사고유형_낙하_비정상_N-11.zip
    type_budgets = {}  # (event_type, is_hazardous) → remaining budget
    all_samples = []

    for img_zip_name in img_zips:
        label_zip_name = img_zip_name.replace("TS_", "TL_")
        label_zip_path = os.path.join(TRAIN_LABEL_DIR, label_zip_name)
        img_zip_path = os.path.join(TRAIN_IMG_DIR, img_zip_name)

        if not os.path.exists(label_zip_path):
            continue

        event_type, is_hazardous = parse_zip_type(img_zip_name)
        if event_type is None:
            continue

        key = (event_type, is_hazardous)
        if key not in type_budgets:
            type_budgets[key] = args.max_per_type

        remaining = type_budgets[key]
        if remaining <= 0:
            continue

        # Calculate per-zip budget (distribute across ~10 zips per type)
        per_zip = max(remaining // 5, 20)  # at least 20 per zip

        print(f"Processing {img_zip_name} ({event_type}, {'hazard' if is_hazardous else 'safe'}) — budget: {per_zip}")
        samples = extract_from_zip_pair(img_zip_path, label_zip_path, per_zip, event_type, is_hazardous)
        print(f"  Extracted {len(samples)} samples")

        all_samples.extend(samples)
        type_budgets[key] -= len(samples)

    # Save training JSON
    random.shuffle(all_samples)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Saved to: {OUTPUT_JSON}")

    # Distribution
    dist = {}
    for s in all_samples:
        key = f"{s['event_type']}_{'hazard' if s['is_hazardous'] else 'safe'}"
        dist[key] = dist.get(key, 0) + 1
    print("\nDistribution:")
    for k, v in sorted(dist.items()):
        print(f"  {k}: {v}")

    # Check images
    n_images = len(list(EXTRACTED_IMAGES.glob("*.jpg")))
    print(f"\nExtracted images: {n_images}")
    print(f"Images dir: {EXTRACTED_IMAGES}")


if __name__ == "__main__":
    main()
