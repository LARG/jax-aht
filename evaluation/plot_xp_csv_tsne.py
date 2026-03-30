"""Generate t-SNE visualizations from XP matrix CSV outputs.

Input CSV format is the matrix CSV produced by evaluation/run.py where each
cell is formatted as "mean (ci_lower, ci_upper)".
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.manifold import TSNE
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "plot_xp_csv_tsne.py requires scikit-learn. Install with `pip install scikit-learn`."
    ) from exc


MEAN_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)")

AGENT_TYPE_COLORS = {
    "heuristic": "#1f77b4",
    "ippo": "#ff7f0e",
    "comedi": "#2ca02c",
    "lbrdiv": "#d62728",
    "brdiv": "#9467bd",
    "other": "#7f7f7f",
}


def infer_agent_type(label: str) -> str:
    label_l = label.lower()

    if "ippo" in label_l:
        return "ippo"

    if "comedi" in label_l:
        return "comedi"

    if "lbrdiv" in label_l:
        return "lbrdiv"

    if "brdiv" in label_l:
        return "brdiv"

    heuristic_markers = (
        "seq_agent",
        "independent_agent",
        "onion_agent",
        "plate_agent",
        "heuristic",
    )
    if any(marker in label_l for marker in heuristic_markers):
        return "heuristic"

    return "other"


def parse_matrix(csv_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"CSV must contain a header and at least one row: {csv_path}")

    col_labels = rows[0][1:]
    row_labels = []
    values = []

    for row in rows[1:]:
        row_labels.append(row[0])
        parsed_row = []
        for cell in row[1:]:
            match = MEAN_RE.match(cell)
            if not match:
                raise ValueError(f"Could not parse mean from cell {cell!r} in {csv_path}")
            parsed_row.append(float(match.group(1)))
        values.append(parsed_row)

    return row_labels, col_labels, np.asarray(values, dtype=float)


def _choose_perplexity(n_samples: int, requested: float | None) -> float:
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for t-SNE.")

    max_valid = max(2.0, float(n_samples - 1))
    if requested is not None:
        if requested >= n_samples:
            raise ValueError(
                f"perplexity ({requested}) must be less than number of samples ({n_samples})."
            )
        return float(requested)

    return min(max_valid, max(5.0, float(n_samples) / 3.0))


def run_tsne(features: np.ndarray, perplexity: float | None, seed: int) -> np.ndarray:
    n_samples = features.shape[0]
    chosen_perplexity = _choose_perplexity(n_samples, perplexity)
    tsne = TSNE(
        n_components=2,
        perplexity=chosen_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(features)


def plot_embedding(coords: np.ndarray, labels: list[str], title: str, out_path: Path):
    agent_types = [infer_agent_type(label) for label in labels]
    point_colors = [AGENT_TYPE_COLORS.get(agent_type, AGENT_TYPE_COLORS["other"]) for agent_type in agent_types]

    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=52, c=point_colors)

    for i, label in enumerate(labels):
        plt.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.9)

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    legend_order = ["heuristic", "ippo", "comedi", "lbrdiv", "brdiv", "other"]
    present_types = []
    for agent_type in legend_order:
        if agent_type in agent_types:
            present_types.append(agent_type)
    if present_types:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=agent_type,
                markerfacecolor=AGENT_TYPE_COLORS[agent_type],
                markersize=8,
            )
            for agent_type in present_types
        ]
        plt.legend(handles=handles, title="Agent type", loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def process_csv(
    csv_path: Path,
    out_dir: Path | None,
    perplexity: float | None,
    seed: int,
) -> tuple[Path, Path]:
    row_labels, col_labels, matrix = parse_matrix(csv_path)

    row_coords = run_tsne(matrix, perplexity=perplexity, seed=seed)
    col_coords = run_tsne(matrix.T, perplexity=perplexity, seed=seed)

    target_dir = out_dir if out_dir is not None else csv_path.parent
    row_out = target_dir / f"{csv_path.stem}_rows_tsne.png"
    col_out = target_dir / f"{csv_path.stem}_cols_tsne.png"

    plot_embedding(
        row_coords,
        row_labels,
        f"t-SNE (rows): {csv_path.stem}",
        row_out,
    )
    plot_embedding(
        col_coords,
        col_labels,
        f"t-SNE (columns): {csv_path.stem}",
        col_out,
    )

    return row_out, col_out


def iter_input_csvs(input_path: Path):
    if input_path.is_file():
        yield input_path
        return
    for csv_path in sorted(input_path.glob("*.csv")):
        if csv_path.stem.endswith("_tidy"):
            continue
        yield csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to one XP matrix CSV or directory containing CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write output PNGs. Defaults to CSV directory.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=None,
        help="t-SNE perplexity. Must be < number of samples.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_path = args.input_path
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    generated = 0
    for csv_path in iter_input_csvs(input_path):
        row_out, col_out = process_csv(
            csv_path,
            out_dir=args.out_dir,
            perplexity=args.perplexity,
            seed=args.seed,
        )
        print(row_out)
        print(col_out)
        generated += 1

    if generated == 0:
        raise ValueError(f"No matrix CSV files found in: {input_path}")


if __name__ == "__main__":
    main()