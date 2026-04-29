"""Download processed safetensors from Hugging Face Hub into processed/.

Usage:
    python human_data_processing/download_from_hf.py
"""

from huggingface_hub import snapshot_download
from pathlib import Path

REPO_ID = "jaxaht/lbf-human-data"
OUTPUT_DIR = Path(__file__).parent / "processed"

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(OUTPUT_DIR),
    )
    print(f"Downloaded to {OUTPUT_DIR}")
