import json
import os
from pathlib import Path

def generate_reasoning_trace(category, event, facility):
    """
    Generates a sophisticated physical reasoning narrative based on AI Hub tags.
    This creates the 'Chain of Thought' that Cosmos needs for 1st place.
    """
    scenarios = {
        "jam": {
            "desc": f"a critical mechanical entrapment hazard involving the {facility}",
            "phys": f"The spatial convergence between the worker's extremities and the {facility}'s moving components indicates a violation of the safe clearance envelope. Kinetic energy transfer is imminent as the machine's automated cycle lacks reactive braking triggers for foreign object intrusion.",
            "pred": f"Irreversible mechanical crushing or limb entrapment within the {facility}'s pinch points."
        },
        "fall": {
            "desc": "a high-altitude gravitational hazard (falling)",
            "phys": "The subject's center of mass is positioned beyond the stable support base without secondary fall-arrest systems. Gravitational potential energy (mgh) is poised to convert into uncontrolled kinetic energy.",
            "pred": "Severe impact trauma upon collision with the lower structural plane."
        },
        "hit": {
            "desc": f"a dynamic momentum transfer hazard from an object near the {facility}",
            "phys": "Vector analysis shows the moving object's trajectory intersecting with the human occupancy zone. The relative velocity indicates a momentum (p=mv) that exceeds human tissue tolerance.",
            "pred": "Blunt force trauma or secondary projectile impact."
        },
        "rule-violation": {
            "desc": "a PPE (Personal Protective Equipment) non-compliance event",
            "phys": "The absence of required structural shielding (e.g., helmet, vest) increases the vulnerability to incidental energy discharges (impact, electrical, or thermal).",
            "pred": "Increased severity of injury due to lack of passive mitigation layers."
        }
    }
    
    # Default scenario
    scene = scenarios.get(event, {
        "desc": f"a complex safety violation involving {event} near the {facility}",
        "phys": "The spatial relationship between the worker and the environment's mechanical hazards violates the established safety buffer protocols.",
        "pred": "Potential for an industrial accident due to unregulated physical interaction."
    })
    
    return f"The scene exhibits {scene['desc']}. Physical analysis: {scene['phys']} Dynamic prediction: {scene['pred']}"

def convert_aihub_to_fine_tuning(label_dir, output_file):
    training_data = []
    
    # Recursively find all JSON files
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".json") and not file.endswith("_meta.json"): # Skip image-specific small jsons if any
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)
                    
                    meta = data.get("meta_information", {})
                    human = data.get("object_information_human", {})
                    facility = data.get("object_information_facility", {}).get("facility_name", "industrial equipment")
                    
                    category = meta.get("category", "safety")
                    event = meta.get("event", "unknown")
                    video_file = meta.get("file_name", "video.mp4")
                    
                    reasoning = generate_reasoning_trace(category, event, facility)
                    
                    # Create a data point for Fine-tuning or Prompting
                    training_data.append({
                        "id": file.replace(".json", ""),
                        "video": video_file,
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"<video>\nAnalyze the physical safety of this industrial environment focusing on {event}."
                            },
                            {
                                "from": "gpt",
                                "value": f"Risk Identified: {event.upper()} RISK detected.\n\nDescription: A {human.get('gender', 'person')} is working near a {facility}.\n\nReasoning Trace: {reasoning}"
                            }
                        ]
                    })
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(training_data)} samples to {output_file}")

if __name__ == "__main__":
    # Path to the AI Hub Sample Label directory
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    LABEL_DIR = str(script_dir.parent.parent / "Sample" / "02.라벨링데이터")
    OUTPUT_FILE = str(script_dir / "fine_tuning_data.json")
    
    convert_aihub_to_fine_tuning(LABEL_DIR, OUTPUT_FILE)
