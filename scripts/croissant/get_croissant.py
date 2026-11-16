#!/usr/bin/env python3
"""Download the Croissant metadata file for a HuggingFace dataset.

The output file is named after the dataset (slashes replaced with underscores)
and saved in the same directory as this script.

Example:
    python get_croissant.py \
        --dataset jaxaht/eval-teammates-br \
        --api_key hf_xxxx
"""

import argparse
import json
from pathlib import Path

import requests

# Set your HuggingFace access token here as a fallback.
# Generate one at https://huggingface.co/settings/tokens (role: read).
HF_TOKEN = None

HF_CROISSANT_URL = "https://huggingface.co/api/datasets/{dataset}/croissant"


def get_croissant(dataset: str, api_key: str) -> None:
    url = HF_CROISSANT_URL.format(dataset=dataset)
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Fetching croissant metadata from: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()

    filename = dataset.replace("/", "_") + ".croissant.json"
    output_path = Path(__file__).parent / filename

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a HuggingFace dataset's Croissant metadata.")
    parser.add_argument(
        "--dataset", required=True,
        help="HuggingFace dataset repo id (e.g. jaxaht/eval-teammates-br)",
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

    get_croissant(dataset=args.dataset, api_key=token)
