#!/usr/bin/env python3
"""Download a path from a HuggingFace dataset repository into a local directory.

The contents of `path_in_dataset` are placed directly inside `dest_dir`,
mirroring the reverse of upload_to_hf.py.

Example:
    python download_from_hf.py \
        --dataset jaxaht/eval-teammates-br \
        --path_in_dataset overcooked_v1_coord_ring \
        --dest_dir results/overcooked-v1/coord_ring/ppo_br_s5
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

# Set your HuggingFace access token here as a fallback.
# A read-scoped token is sufficient for private datasets.
# Generate one at https://huggingface.co/settings/tokens (role: read).
HF_TOKEN = None


def download(dataset: str, path_in_dataset: str, dest_dir: str, api_key: str) -> None:
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{dataset}/{path_in_dataset}/' -> '{dest_path.resolve()}'")

    snapshot_download(
        repo_id=dataset,
        repo_type="dataset",
        allow_patterns=f"{path_in_dataset}/**",
        local_dir=str(dest_path),
        token=api_key,
    )

    # snapshot_download preserves the repo path structure, so files land at
    # dest_dir/path_in_dataset/... Move them up one level so dest_dir contains
    # the files directly (matching what was uploaded).
    nested = dest_path / path_in_dataset
    if nested.exists() and nested.is_dir():
        for item in nested.iterdir():
            item.rename(dest_path / item.name)
        nested.rmdir()

    print(f"\nDone. Files saved to: {dest_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a path from a HuggingFace dataset.")
    parser.add_argument(
        "--dataset", required=True,
        help="HuggingFace dataset repo id (e.g. jaxaht/eval-teammates-br)",
    )
    parser.add_argument(
        "--path_in_dataset", required=True,
        help="Path inside the dataset repo to download (e.g. overcooked_v1_coord_ring)",
    )
    parser.add_argument(
        "--dest_dir", required=True,
        help="Local directory to download into (e.g. results/overcooked-v1/coord_ring/ppo_br_s5)",
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

    download(
        dataset=args.dataset,
        path_in_dataset=args.path_in_dataset,
        dest_dir=args.dest_dir,
        api_key=token,
    )
