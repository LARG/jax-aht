#!/usr/bin/env python3
"""Upload a local agent directory to a HuggingFace dataset repository.

Example:
    python upload_to_hf.py \
        --source_dir results/overcooked-v1/coord_ring/ppo_br_s5 \
        --dataset jaxaht/eval-teammates-br \
        --path_in_dataset overcooked_v1_coord_ring
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

# Set your HuggingFace access token here as a fallback.
# Generate one at https://huggingface.co/settings/tokens (role: write).
HF_TOKEN = None

# Files and directories to exclude from the upload.
IGNORE_PATTERNS = [
    "run.log",
    "**/run.log",
    "wandb",
    "**/wandb/**",
    ".hydra",
    "**/.hydra/**",
]


def upload(source_dir: str, dataset: str, path_in_dataset: str, api_key: str) -> None:
    source_path = Path(source_dir).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    api = HfApi(token=api_key)

    print(f"Uploading '{source_path}' -> '{dataset}/{path_in_dataset}/'")
    print(f"Ignoring patterns: {IGNORE_PATTERNS}")

    api.upload_folder(
        folder_path=str(source_path),
        repo_id=dataset,
        repo_type="dataset",
        path_in_repo=path_in_dataset,
        ignore_patterns=IGNORE_PATTERNS,
    )

    print(f"\nDone. View at: https://huggingface.co/datasets/{dataset}/tree/main/{path_in_dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a directory to a HuggingFace dataset.")
    parser.add_argument(
        "--source_dir", required=True,
        help="Local directory whose contents to upload (e.g. results/overcooked-v1/coord_ring/ppo_br_s5)",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="HuggingFace dataset repo id (e.g. jaxaht/eval-teammates-br)",
    )
    parser.add_argument(
        "--path_in_dataset", required=True,
        help="Destination path inside the dataset repo (e.g. overcooked_v1_coord_ring)",
    )
    parser.add_argument(
        "--api_key", default=None,
        help="HuggingFace access token. Falls back to the HF_TOKEN constant in this file.",
    )
    args = parser.parse_args()

    token = args.api_key or HF_TOKEN
    if not token:
        raise ValueError(
            "No HuggingFace API key found. Either pass --api_key or set HF_TOKEN at the top of this file."
        )

    upload(
        source_dir=args.source_dir,
        dataset=args.dataset,
        path_in_dataset=args.path_in_dataset,
        api_key=token,
    )
