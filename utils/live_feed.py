import cv2
import time
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.agent import CosmosGuardianAgent  # noqa: E402

def capture_and_analyze(agent, source=0, interval=10, duration=3):
    """
    Captures video clips and analyzes them.
    source: webcam index or stream URL
    interval: seconds between analyses
    duration: duration of each clip in seconds
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Live feed started. Source: {source} ({width}x{height} @ {fps}fps)")
    print(f"Analyzing {duration}s clips every {interval}s...")

    temp_video = "temp_live_clip.mp4"

    try:
        while True:
            print("\nCapturing fresh clip...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
            
            start_time = time.time()
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                # Display local preview
                cv2.imshow("Cosmos Guardian - Live Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

            out.release()
            
            print("Analyzing clip...")
            result = {}
            for update in agent.analyze_media(temp_video, safety_context="Live Monitoring"):
                if update["stage"] == "complete":
                    result = update["result"]
            
            print("\nGuard Result:")
            for hazard in result.get("hazards_detected", []):
                print(f"[{hazard.get('severity')}] {hazard.get('type')}: {hazard.get('description')}")
                print(f"Reasoning: {hazard.get('reasoning')}")
            
            print(f"Overall Safety Score: {result.get('overall_safety_score')}")
            
            # Wait for next interval
            print(f"Waiting {interval}s for next check...")
            wait_start = time.time()
            while (time.time() - wait_start) < interval:
                ret, frame = cap.read() # Keep reading to flush buffer
                if not ret: break
                cv2.imshow("Cosmos Guardian - Live Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if Path(temp_video).exists():
            Path(temp_video).unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Webcam index or RTSP/HTTP stream URL")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--duration", type=int, default=3)
    args = parser.parse_args()

    # Convert source to int if it's a digit
    src = int(args.source) if args.source.isdigit() else args.source
    
    agent = CosmosGuardianAgent()
    capture_and_analyze(agent, source=src, interval=args.interval, duration=args.duration)
