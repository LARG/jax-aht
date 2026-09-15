"""Download the evaluation and validation teammate data from Hugging Face.

By default only files that are missing locally are downloaded; pass --force to
re-download everything.
"""

import argparse
import os
import shutil
import tempfile
import zipfile

from huggingface_hub import hf_hub_download, list_repo_files

DEFAULT_REPO_ID = "jaxaht/eval-teammates"
VAL_REPO_ID = "jaxaht/val-teammates"

# name -> (kind, remote name, local destination, repo id)
DATA_FILES = {
    "best_returns_teammates": (
        "zip",
        "best_heldout_returns.zip",
        "results/",
        DEFAULT_REPO_ID,
    ),
    "lbf_teammates": ("dir", "lbf_7x7", "eval_teammates/", DEFAULT_REPO_ID),
    "lbf_12x12_teammates": ("dir", "lbf_12x12", "eval_teammates/", DEFAULT_REPO_ID),
    "overcooked-v1_teammates": (
        "dir",
        "overcooked-v1",
        "eval_teammates/",
        DEFAULT_REPO_ID,
    ),
    "hanabi_teammates": ("dir", "hanabi", "eval_teammates/", DEFAULT_REPO_ID),
    "mini_hanabi_teammates": ("dir", "mini_hanabi", "eval_teammates/", DEFAULT_REPO_ID),
    "lbf_val_teammates": ("dir", "lbf", "val_teammates/", VAL_REPO_ID),
    "lbf_12x12_val_teammates": ("dir", "lbf_12x12", "val_teammates/", VAL_REPO_ID),
    "overcooked-v1_val_teammates": (
        "dir",
        "overcooked-v1",
        "val_teammates/",
        VAL_REPO_ID,
    ),
    "mini_hanabi_val_teammates": ("dir", "mini_hanabi", "val_teammates/", VAL_REPO_ID),
}


def _move_tree(source_dir: str, destination_dir: str, force: bool) -> tuple[int, int]:
    """Move every file under source_dir into destination_dir, preserving structure.

    Existing files are left alone unless force is set. Returns (moved, skipped).
    """
    moved = skipped = 0
    for root, _, filenames in os.walk(source_dir):
        for name in filenames:
            src = os.path.join(root, name)
            dst = os.path.join(destination_dir, os.path.relpath(src, source_dir))

            if os.path.isfile(dst) and not force:
                skipped += 1
                continue

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved += 1
    return moved, skipped


def download_and_unzip_hf_file(
    repo_id: str, filename: str, destination_dir: str, force: bool = False
) -> None:
    """Download a zip archive from a HF dataset repo and extract it into destination_dir.

    If the archive contains a single top-level folder, its contents are placed
    directly in destination_dir rather than nested one level deeper.

    Raises on download or extraction failure.
    """
    print(f"Starting download & extraction: {repo_id}/{filename} -> {destination_dir}")
    os.makedirs(destination_dir, exist_ok=True)

    archive_path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", force_download=force
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_dir)

        # Unwrap a single top-level folder, if that is how the archive is laid out.
        source_dir = temp_dir
        entries = os.listdir(temp_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(temp_dir, entries[0])):
            source_dir = os.path.join(temp_dir, entries[0])

        moved, skipped = _move_tree(source_dir, destination_dir, force)

    if skipped:
        print(f"Skipped {skipped} file(s) already present in {destination_dir}.")
    if moved:
        print(f"Moved {moved} file(s) to {destination_dir}.")
    elif skipped:
        print(f"Nothing to do: {filename} is already fully extracted.")
    else:
        print(f"Warning: {filename} contained no files.")


def download_hf_directory(
    repo_id: str, remote_dir: str, destination_dir: str, force: bool = False
) -> None:
    """Download a directory from a HF dataset repo, preserving its structure.

    remote_dir becomes a subdirectory of destination_dir. Files already present
    locally are skipped unless force is set.

    Raises on download failure.
    """
    destination = os.path.join(destination_dir, remote_dir)
    print(f"Starting download: {repo_id}/{remote_dir} -> {destination}")
    os.makedirs(destination_dir, exist_ok=True)

    repo_files = [
        f
        for f in list_repo_files(repo_id=repo_id, repo_type="dataset")
        if f == remote_dir or f.startswith(f"{remote_dir}/")
    ]
    if not repo_files:
        print(f"Warning: no files found under {repo_id}/{remote_dir}.")
        return

    if force:
        wanted = repo_files
    else:
        wanted = [
            f
            for f in repo_files
            if not os.path.exists(os.path.join(destination_dir, f))
        ]

    skipped = len(repo_files) - len(wanted)
    if skipped:
        print(f"Skipping {skipped} file(s) already present in {destination_dir}.")
    if not wanted:
        print(f"Nothing to do: {remote_dir} is already fully downloaded.")
        return

    print(f"Downloading {len(wanted)} file(s) from {repo_id}/{remote_dir}...")
    for name in wanted:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=name,
            local_dir=destination_dir,
            force_download=force,
        )
    print(f"Successfully downloaded {remote_dir} to {destination_dir}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download evaluation/validation teammate data from Hugging Face. "
        "By default, files already present locally are skipped."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-download everything, overwriting files that are already present locally.",
    )
    args = parser.parse_args()

    failed = []
    for name, (kind, remote_name, destination_dir, repo_id) in DATA_FILES.items():
        download = (
            download_and_unzip_hf_file if kind == "zip" else download_hf_directory
        )
        try:
            download(repo_id, remote_name, destination_dir, force=args.force)
        except Exception as e:  # noqa: BLE001 - report and continue to the next dataset
            print(f"Download failed for {name}: {type(e).__name__}: {e}")
            failed.append(name)
        else:
            print(f"Download completed successfully for {name}.")

    if failed:
        print(f"\n{len(failed)} download(s) failed: {', '.join(failed)}")
        return 1

    print("\nAll downloads completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
