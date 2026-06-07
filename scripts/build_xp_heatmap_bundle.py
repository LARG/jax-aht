#!/usr/bin/env python3
"""Build a shareable bundle of XP heatmap artifacts.

The script expects a directory of XP matrix CSVs. It copies those CSVs,
renders one paper-style PDF heatmap for each BR-column-normalized matrix CSV,
checks that normalized matrices have maximum value 1.0, writes a manifest, and
optionally creates a zip archive.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from evaluation.plot_xp_csv_heatmap import generate_heatmap_from_csv, parse_matrix


DEFAULT_TITLES = {
    "lbf_7x7_nolevels": "LBF 7x7 No Levels",
    "lbf_12x12": "LBF 12x12",
    "overcooked_asymm_advantages": "Asymmetric Advantages",
    "overcooked_coord_ring": "Coordination Ring",
    "overcooked_counter_circuit": "Counter Circuit",
    "overcooked_cramped_room": "Cramped Room",
    "overcooked_forced_coord": "Forced Coordination",
}


def task_name_from_csv(csv_path: Path) -> str:
    return csv_path.name.split("__", 1)[0]


def is_normalized_matrix_csv(csv_path: Path) -> bool:
    return csv_path.name.endswith("_brcolmax_normalized=True.csv")


def validate_unit_max(csv_path: Path, tolerance: float) -> float:
    _, _, values = parse_matrix(csv_path)
    max_value = float(np.nanmax(values))
    if not np.isclose(max_value, 1.0, atol=tolerance):
        raise ValueError(f"{csv_path} max is {max_value:.6f}, expected 1.0")
    return max_value


def copy_csvs(csv_dir: Path, out_csv_dir: Path) -> list[Path]:
    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    out_csv_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for csv_path in csv_paths:
        dst = out_csv_dir / csv_path.name
        shutil.copy2(csv_path, dst)
        copied.append(dst)
    return copied


def build_bundle(
    csv_dir: Path,
    out_dir: Path,
    *,
    expected_pdf_count: int | None,
    max_tolerance: float,
    make_zip: bool,
) -> Path | None:
    resolved_csv_dir = csv_dir.resolve()
    resolved_out_dir = out_dir.resolve()
    if resolved_csv_dir == resolved_out_dir or resolved_csv_dir.is_relative_to(resolved_out_dir):
        raise ValueError("--csv-dir must not be inside --out-dir because the output directory is recreated")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_csv_dir = out_dir / "csvs"
    out_pdf_dir = out_dir / "pdfs"
    out_pdf_dir.mkdir(parents=True)

    copied_csvs = copy_csvs(csv_dir, out_csv_dir)
    normalized_csvs = [csv_path for csv_path in copied_csvs if is_normalized_matrix_csv(csv_path)]
    if expected_pdf_count is not None and len(normalized_csvs) != expected_pdf_count:
        raise ValueError(f"Expected {expected_pdf_count} normalized CSVs, found {len(normalized_csvs)}")

    manifest: dict[str, object] = {
        "source_csv_dir": str(csv_dir),
        "normalization": "BR column-max normalization; each normalized matrix max is checked as 1.0.",
        "colormap": "paper-style red-yellow-green.",
        "package_contents": "XP PDF heatmaps plus matching CSVs.",
        "outputs": {},
    }
    outputs: dict[str, object] = {}

    for csv_path in normalized_csvs:
        task = task_name_from_csv(csv_path)
        pdf_path = out_pdf_dir / f"{task}__xp_return_brcolmax_heatmap.pdf"
        max_value = validate_unit_max(csv_path, max_tolerance)
        generate_heatmap_from_csv(
            csv_path,
            title=DEFAULT_TITLES.get(task, task.replace("_", " ").title()),
            out_path=pdf_path,
            cbar_label="Mean return",
            vmin=0.0,
            vmax=1.0,
        )
        outputs[task] = {
            "csv": str(csv_path),
            "pdf": str(pdf_path),
            "max_after_normalization": max_value,
        }

    manifest["outputs"] = outputs
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if not make_zip:
        return None

    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.name != ".DS_Store":
                archive.write(path, path.relative_to(out_dir.parent))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", required=True, type=Path, help="Directory containing XP CSVs.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output bundle directory.")
    parser.add_argument(
        "--expected-pdf-count",
        type=int,
        default=None,
        help="Fail if this many normalized heatmap PDFs are not generated.",
    )
    parser.add_argument(
        "--max-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for checking each normalized matrix max is 1.0.",
    )
    parser.add_argument("--no-zip", action="store_true", help="Do not create a zip archive.")
    args = parser.parse_args()

    zip_path = build_bundle(
        args.csv_dir,
        args.out_dir,
        expected_pdf_count=args.expected_pdf_count,
        max_tolerance=args.max_tolerance,
        make_zip=not args.no_zip,
    )
    print(args.out_dir)
    if zip_path is not None:
        print(zip_path)


if __name__ == "__main__":
    main()
