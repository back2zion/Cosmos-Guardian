"""
Convert AIHub 71407 (건설 현장 위험 상태 판단 데이터) to Cosmos Reason 2 fine-tuning format.

Input: JPG images + JSON labels with bounding box/polygon annotations
Output: JSONL training data with physics-based safety reasoning conversations

Usage:
  python construction_to_cosmos.py --data-dir /path/to/71407 --output fine_tuning_construction.json
  python construction_to_cosmos.py --data-dir /path/to/71407 --output fine_tuning_construction.json --max-per-type 500
"""

import argparse
import json
import os
import random
from pathlib import Path

# ============================================================
# Accident type mappings
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

# Situation: N = abnormal (hazard present), Y = normal (safe)
SITUATION_MAP = {
    "N": "hazardous",
    "Y": "safe",
}

# ============================================================
# Physics-based reasoning templates per accident type
# ============================================================
REASONING_TEMPLATES = {
    "fall": [
        "The worker is positioned at an elevated surface without adequate fall protection. "
        "Gravitational potential energy (E = mgh) means a fall from this height would result in "
        "significant kinetic energy at impact. At approximately {height}m, impact velocity would reach "
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
        "E = mgh. Even a small tool (0.5 kg) dropped from {height}m reaches "
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
        "creating compression pressures that can crush bone and tissue. "
        "Clearance is insufficient for safe hand/body positioning.",
    ],
    "overturn": [
        "Equipment stability is compromised — the center of gravity has shifted beyond the support base. "
        "Overturning moment M = F × d exceeds the restoring moment provided by the equipment's weight "
        "and base geometry. Ground conditions or load distribution may have altered the stability triangle.",
        "The equipment is operating on uneven terrain or with an asymmetric load, "
        "shifting the center of mass outside the stability envelope. "
        "Once tipping begins, gravitational torque accelerates the overturn — "
        "the rotational kinetic energy at impact can be devastating.",
        "Structural instability detected — lateral forces or wind loads are creating "
        "a tipping moment that approaches the equipment's rollover threshold. "
        "The moment of inertia resists initial rotation, but once initiated, "
        "the collapse progresses rapidly under gravitational acceleration.",
    ],
    "fire": [
        "Thermal hazard conditions present — heat source proximity to combustible materials "
        "creates ignition risk. Fire propagation follows the fire triangle principle: "
        "fuel, oxygen, and heat source are all present in this scenario.",
        "Potential ignition sources detected near flammable materials. "
        "Radiant heat transfer (Q = σεAT⁴) from hot work operations can raise "
        "nearby material temperatures above their flash point. "
        "Convective heat currents may spread fire to adjacent combustibles.",
        "Welding/cutting operations without proper fire watch protocol. "
        "Sparks at temperatures exceeding 1500°C can ignite materials with "
        "flash points below 100°C. The thermal energy transfer rate "
        "depends on material conductivity and exposure duration.",
    ],
}

SAFE_REASONING = {
    "fall": "Workers are operating at height with proper fall protection systems visible — "
            "harnesses, guardrails, and secured platforms are in place. "
            "The fall arrest system can absorb kinetic energy during a potential fall.",
    "falling_object": "Material handling at height follows proper protocols — "
                      "toe boards, debris netting, and secured storage prevent dropped objects. "
                      "The exclusion zone below is properly maintained.",
    "entrapment": "Machine guarding and lockout/tagout procedures appear to be in effect. "
                  "Workers maintain safe distance from moving parts, and physical barriers "
                  "prevent inadvertent contact with pinch points.",
    "overturn": "Equipment is operating within its stability envelope on level ground. "
                "Load is properly centered and within rated capacity. "
                "Outriggers or stabilizers are deployed where applicable.",
    "fire": "Hot work area is properly prepared — fire-resistant blankets cover nearby combustibles, "
            "fire extinguisher is accessible, and fire watch protocols are being followed.",
}

# ============================================================
# Generate conversation from label data
# ============================================================
def generate_conversation(label_data: dict, image_filename: str) -> dict:
    raw = label_data.get("Raw_Data_Info.", {})
    learning = label_data.get("Learning_Data_Info.", {})

    type_kr = raw.get("Type_Description", "")
    event_type = TYPE_MAP.get(type_kr, "unknown")
    situation_id = raw.get("Situation_ID", "")
    is_hazardous = situation_id.startswith("N")

    annotations = learning.get("Annotations", [])
    num_objects = len(annotations)

    # Generate height for physics calculations
    height = random.uniform(3, 12)
    velocity = (2 * 9.81 * height) ** 0.5

    # Build hazard description
    if is_hazardous:
        reasoning_pool = REASONING_TEMPLATES.get(event_type, REASONING_TEMPLATES["fall"])
        reasoning = random.choice(reasoning_pool).format(height=height, velocity=velocity)
        severity = random.choice(["High", "Critical"])
        safety_score = random.randint(10, 35)
        hazard_type = TYPE_DESCRIPTIONS.get(event_type, event_type)

        response = {
            "hazards_detected": [
                {
                    "type": hazard_type,
                    "severity": severity,
                    "description": f"Construction site hazard: {hazard_type.lower()} risk identified. "
                                   f"{num_objects} relevant objects/zones annotated in scene.",
                    "reasoning": reasoning,
                }
            ],
            "overall_safety_score": safety_score,
            "recommendation": get_recommendation(event_type),
        }
    else:
        reasoning = SAFE_REASONING.get(event_type, "Safe working conditions observed.")
        safety_score = random.randint(70, 95)

        response = {
            "hazards_detected": [],
            "overall_safety_score": safety_score,
            "safety_observation": f"Construction site appears to follow proper {TYPE_DESCRIPTIONS.get(event_type, '')} "
                                  f"prevention protocols. {reasoning}",
            "recommendation": "Continue current safety practices. Maintain vigilance and regular safety audits.",
        }

    # Format as conversation
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


def get_recommendation(event_type: str) -> str:
    recs = {
        "fall": "Install guardrails and ensure all workers use personal fall arrest systems. "
                "Verify anchor points meet OSHA 5000-lb requirement.",
        "falling_object": "Secure all materials at height with toe boards and debris netting. "
                          "Establish and enforce hard-hat zones below overhead work.",
        "entrapment": "Implement lockout/tagout procedures. Install machine guards on all pinch points. "
                      "Maintain minimum clearance distances per equipment specifications.",
        "overturn": "Verify ground conditions and load charts before operation. "
                    "Deploy outriggers on level ground. Do not exceed rated capacity.",
        "fire": "Remove combustibles from hot work area (35-ft radius). "
                "Assign fire watch with extinguisher. Obtain hot work permit.",
    }
    return recs.get(event_type, "Review and strengthen safety protocols.")


# ============================================================
# Main conversion
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Convert construction safety data to Cosmos format")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing labels/ and images/ subdirectories")
    parser.add_argument("--output", type=str, default="fine_tuning_construction.json",
                        help="Output JSON file path")
    parser.add_argument("--max-per-type", type=int, default=None,
                        help="Max samples per accident type (for balanced dataset)")
    parser.add_argument("--labels-only", action="store_true",
                        help="Process labels even without matching images")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    label_dir = data_dir / "labels"
    image_dir = data_dir / "images"

    # Find all extracted label JSONs
    label_files = []
    for d in [label_dir, data_dir]:
        label_files.extend(d.rglob("*.json"))

    if not label_files:
        print("No label JSON files found. Extract label ZIPs first.")
        print(f"  Looked in: {label_dir} and {data_dir}")
        return

    print(f"Found {len(label_files)} label files")

    # Find available images
    image_files = set()
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
        for img in image_dir.rglob(ext):
            image_files.add(img.stem)
    print(f"Found {len(image_files)} images")

    # Process labels
    samples = []
    type_counts = {}
    skipped = 0

    for label_path in sorted(label_files):
        try:
            with open(label_path, "r", encoding="utf-8-sig") as f:
                label = json.load(f)
        except Exception:
            continue

        # Skip non-label JSON files (e.g., previously generated training data)
        if not isinstance(label, dict):
            continue

        raw = label.get("Raw_Data_Info.", {})
        learning = label.get("Learning_Data_Info.", {})
        json_id = learning.get("Json_Data_ID", label_path.stem)

        # Check if matching image exists (or labels-only mode)
        if not args.labels_only and json_id not in image_files:
            skipped += 1
            continue

        type_kr = raw.get("Type_Description", "")
        event_type = TYPE_MAP.get(type_kr, "unknown")

        # Apply per-type limit
        if args.max_per_type:
            type_counts.setdefault(event_type, 0)
            if type_counts[event_type] >= args.max_per_type:
                continue
            type_counts[event_type] += 1

        image_filename = f"{json_id}.jpg"
        conv = generate_conversation(label, image_filename)
        samples.append(conv)

    print(f"\nConverted {len(samples)} samples (skipped {skipped} without images)")

    # Print distribution
    dist = {}
    for s in samples:
        key = f"{s['event_type']}_{'hazard' if s['is_hazardous'] else 'safe'}"
        dist[key] = dist.get(key, 0) + 1
    print("Distribution:")
    for k, v in sorted(dist.items()):
        print(f"  {k}: {v}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
