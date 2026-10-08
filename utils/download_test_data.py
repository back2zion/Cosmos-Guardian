import fiftyone as fo
import fiftyone.utils.huggingface as fouh
from pathlib import Path

def download_sample_data(output_dir: str = "./assets/test_data"):
    print("Loading Safe_Unsafe_Test dataset from Hugging Face...")
    try:
        dataset = fo.load_dataset("pjramg/Safe_Unsafe_Test")
    except:
        dataset = fouh.load_from_hub("pjramg/Safe_Unsafe_Test")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting sample videos to {output_dir}...")
    # Just take 5 samples for testing
    samples = dataset.limit(5)
    
    for i, sample in enumerate(samples):
        source_path = Path(sample.filepath)
        label = sample.label if hasattr(sample, 'label') else "unknown"
        target_path = output_path / f"sample_{i}_{label}.mp4"
        
        # Copy file
        import shutil
        shutil.copy(source_path, target_path)
        print(f"Saved: {target_path}")

if __name__ == "__main__":
    download_sample_data()
