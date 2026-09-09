"""Check that teammate checkpoint paths in the config files exist on Hugging Face.

The heldout and validation settings point at checkpoints hosted in our dataset
repos. Nothing in the repo notices when a config path and the uploaded data drift
apart, so this test lists each dataset repo once and checks every configured path
against that listing. It needs network access, so it is opt-in: run it with

    pytest -m hf_data --run-hf-data

or let the weekly check-teammate-paths workflow run it.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "evaluation/configs"

# Local directory a path is relative to -> dataset repos that may hold it. Best
# responses live in their own repo, except for BRDiv's, which are stored inside the
# teammate run they respond to (via ckpt_key: final_params_br), so eval_teammates
# paths have to be looked up in both.
DIR_TO_REPOS = {
    "eval_teammates": ("jaxaht/eval-teammates", "jaxaht/eval-teammates-br"),
    "val_teammates": ("jaxaht/val-teammates",),
}
PATH_KEYS = ("path", "weight_file")


def iter_configured_paths(config):
    """Yield every checkpoint path in a loaded config, however deeply nested."""
    if isinstance(config, dict):
        for key, value in config.items():
            if key in PATH_KEYS and isinstance(value, str) and value:
                yield value
            else:
                yield from iter_configured_paths(value)
    elif isinstance(config, list):
        for item in config:
            yield from iter_configured_paths(item)


def collect_paths():
    """Map each configured checkpoint path to the config files that reference it."""
    paths = {}
    for config_file in sorted(CONFIG_DIR.glob("global_*.yaml")):
        config = yaml.safe_load(config_file.read_text())
        for path in iter_configured_paths(config):
            paths.setdefault(path, set()).add(config_file.name)
    return paths


@pytest.mark.hf_data
def test_configured_teammate_paths_exist_on_hf(request):
    if not request.config.getoption("--run-hf-data"):
        pytest.skip("needs network access to Hugging Face; pass --run-hf-data to run")

    from huggingface_hub import list_repo_files

    paths = collect_paths()
    assert paths, f"no checkpoint paths found in {CONFIG_DIR}/global_*.yaml"

    # One listing per repo is far cheaper than an existence check per path.
    repo_files = {
        repo_id: set(list_repo_files(repo_id=repo_id, repo_type="dataset"))
        for repo_id in sorted({r for repos in DIR_TO_REPOS.values() for r in repos})
    }

    missing = []
    for path, config_files in sorted(paths.items()):
        local_dir, _, remote_path = path.partition("/")
        repo_ids = DIR_TO_REPOS.get(local_dir)
        sources = ", ".join(sorted(config_files))

        if repo_ids is None:
            missing.append(
                f"{path} ({sources}): unknown top-level directory {local_dir!r}"
            )
            continue

        prefix = remote_path.rstrip("/")
        found = any(
            f == prefix or f.startswith(f"{prefix}/")
            for repo_id in repo_ids
            for f in repo_files[repo_id]
        )
        if not found:
            missing.append(f"{path} ({sources}): not found in {' or '.join(repo_ids)}")

    assert not missing, "\n".join(
        [f"{len(missing)} of {len(paths)} configured paths are missing:", *missing]
    )
