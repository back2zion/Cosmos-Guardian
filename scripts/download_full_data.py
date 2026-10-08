from huggingface_hub import snapshot_download
from pathlib import Path
import os
import time

def download_full_dataset(output_dir: str = "data/full"):
    print("Initiating Full Dataset Download (1.48TB Target)...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Starting download stream from Hugging Face Hub...")
        # Note: We use max_workers for parallelism if supported, but snapshot_download handles it internally.
        # We'll rely on HF transfer acceleration if available.
        # We verify file integrity but skip symlinks to get actual files.
        
        start_time = time.time()
        
        snapshot_download(
            repo_id="pjramg/Safe_Unsafe_Test",
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True 
        )
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Full Dataset Download Complete! Total time: {duration/3600:.2f} hours")
        
    except Exception as e:
        print(f"Full download interrupted: {e}")

if __name__ == "__main__":
    download_full_dataset()
