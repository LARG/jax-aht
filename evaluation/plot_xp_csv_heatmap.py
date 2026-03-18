"""Generate a heatmap from an XP matrix CSV exported by evaluation/run.py.

The CSV cells are formatted like "0.50 (0.45, 0.55)". This script extracts the
mean value from each cell and renders a readable heatmap.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


MEAN_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)")


def parse_matrix(csv_path: Path):
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0][1:]
    row_labels = []
    values = []

    for row in rows[1:]:
        row_labels.append(row[0])
        parsed = []
        for cell in row[1:]:
            match = MEAN_RE.match(cell)
            if not match:
                raise ValueError(f"Could not parse mean from cell: {cell!r}")
            parsed.append(float(match.group(1)))
        values.append(parsed)

    return row_labels, header, np.array(values, dtype=float)


def plot_heatmap(
    row_labels,
    col_labels,
    values,
    title: str,
    out_path: Path,
    annotate: bool = True,
):
    plt.figure(figsize=(max(10, len(col_labels) * 0.8), max(8, len(row_labels) * 0.6)))
    sns.heatmap(
        values,
        cmap="viridis",
        annot=annotate,
        fmt=".2f",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Mean value"},
    )
    plt.title(title)
    plt.xlabel("Best-response policies")
    plt.ylabel("Held-out teammates")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def generate_heatmap_from_csv(
    csv_path: Path,
    title: str | None = None,
    out_path: Path | None = None,
    annotate: bool = True,
) -> Path:
    row_labels, col_labels, values = parse_matrix(csv_path)
    resolved_out_path = out_path or csv_path.with_suffix(".png")
    resolved_title = title or csv_path.stem
    plot_heatmap(row_labels, col_labels, values, resolved_title, resolved_out_path, annotate)
    return resolved_out_path


def iter_csvs(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    for csv_path in sorted(input_path.glob("*.csv")):
        yield csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a CSV file or a directory containing XP CSV files.",
    )
    parser.add_argument("--title", default="XP Matrix Heatmap")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--no-annot",
        action="store_true",
        help="Disable per-cell numeric annotations.",
    )
    args = parser.parse_args()

    input_path = args.input_path
    annotate = not args.no_annot

    if input_path.is_file():
        out_path = generate_heatmap_from_csv(
            input_path,
            title=args.title,
            out_path=args.out,
            annotate=annotate,
        )
        print(out_path)
        return

    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input path must be a CSV file or directory: {input_path}")

    generated = 0
    for csv_path in iter_csvs(input_path):
        out_path = generate_heatmap_from_csv(
            csv_path,
            title=csv_path.stem,
            out_path=None,
            annotate=annotate,
        )
        print(out_path)
        generated += 1

    if generated == 0:
        raise ValueError(f"No CSV files found in directory: {input_path}")


if __name__ == "__main__":
    main()
