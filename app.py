import argparse
import json
from pathlib import Path
try:
    from cosmos_guardian.core.agent import CosmosGuardianAgent
except ImportError:
    from core.agent import CosmosGuardianAgent

def main():
    parser = argparse.ArgumentParser(description="Cosmos Guardian: AI Safety Reasoning Agent")
    parser.add_argument("--media", type=str, required=True, help="Path to video or image file")
    parser.add_argument("--context", type=str, default="Industrial Warehouse Safety", help="Safety context for analysis")
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason2-2B", help="Hugging Face model ID")
    
    args = parser.parse_args()

    # Expand paths
    media_path = str(Path(args.media).absolute())
    
    if not Path(media_path).exists():
        print(f"Error: Media file not found at {media_path}")
        return

    agent = CosmosGuardianAgent(model_id=args.model)
    
    print("-" * 50)
    print(f"Target Media: {media_path}")
    print(f"Context: {args.context}")
    print("-" * 50)

    result = None
    for update in agent.analyze_media(media_path, safety_context=args.context):
        if update["stage"] == "complete":
            result = update["result"]
        elif "detail" in update:
            print(f"[{update['stage']}] {update['detail']}")

    print("\n--- ANALYSIS RESULT ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-" * 50)

if __name__ == "__main__":
    main()
