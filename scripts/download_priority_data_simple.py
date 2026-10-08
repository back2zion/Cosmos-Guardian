from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path
import os
import shutil

def download_priority_data_simple(output_dir: str = "data/priority"):
    print("Initiating Direct Stream Download: Smart Manufacturing Safety Data Subset...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Connected to Hugging Face Hub. downloading raw media files...")
        
        # Download the dataset repository content to a cache dir, 
        # allowing patterns to select only video files if possible.
        # For simplicity and speed, we grab the 'data' folder or files.
        # Since we don't know exact structure, we start with a snapshot allowing specific extensions
        
        # We'll use allow_patterns to get videos
        # Adjust pattern based on typical dataset structure
        download_path = snapshot_download(
            repo_id="pjramg/Safe_Unsafe_Test",
            repo_type="dataset",
            allow_patterns=["*.mp4", "*.avi", "*.mov", "*.json", "*.parquet"], 
            local_dir=output_dir,
            local_dir_use_symlinks=False # Get actual files
        )
        
        print(f"Priority Data Downloaded to {download_path}")
        print("Ready for training injection.")
        
    except Exception as e:
        print(f"Direct stream download failed: {e}")

if __name__ == "__main__":
    download_priority_data_simple()
