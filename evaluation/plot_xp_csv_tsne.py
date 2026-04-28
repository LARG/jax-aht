"""Generate t-SNE visualizations from XP matrix CSV outputs.

Input CSV format is the matrix CSV produced by evaluation/run.py where each
cell is formatted as "mean (ci_lower, ci_upper)".
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Literal
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


def plot_embedding(
    coords: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
    *,
    show_point_labels: bool = True,
    show_density: bool = False,
    point_size: float = 52,
    point_alpha: float = 0.95,
    figsize: tuple[float, float] = (10.0, 8.0),
    dpi: int = 220,
    title_fontsize: int = 14,
    axis_fontsize: int = 12,
    tick_fontsize: int = 10,
    legend_fontsize: int = 10,
    legend_title_fontsize: int = 11,
):
    agent_types = [infer_agent_type(label) for label in labels]
    point_colors = [
        AGENT_TYPE_COLORS.get(agent_type, AGENT_TYPE_COLORS["other"]) for agent_type in agent_types
    ]

    plt.figure(figsize=figsize)

    # Light density layer gives cluster structure without hiding category colors.
    if show_density:
        plt.hexbin(
            coords[:, 0],
            coords[:, 1],
            gridsize=28,
            cmap="Greys",
            bins="log",
            mincnt=1,
            alpha=0.25,
            linewidths=0,
            zorder=1,
        )

    plt.scatter(coords[:, 0], coords[:, 1], s=point_size, c=point_colors, alpha=point_alpha, zorder=2)

    if show_point_labels:
        for i, label in enumerate(labels):
            plt.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.9)

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel("t-SNE 1", fontsize=axis_fontsize)
    plt.ylabel("t-SNE 2", fontsize=axis_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

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
        plt.legend(
            handles=handles,
            title="Agent type",
            loc="best",
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def process_csv(
    csv_path: Path,
    out_dir: Path | None,
    perplexity: float | None,
    seed: int,
    embedding: Literal["both", "rows", "cols"],
    show_point_labels: bool,
    show_density: bool,
    point_size: float,
    point_alpha: float,
    figsize: tuple[float, float],
    dpi: int,
    title_fontsize: int,
    axis_fontsize: int,
    tick_fontsize: int,
    legend_fontsize: int,
    legend_title_fontsize: int,
) -> tuple[Path | None, Path | None]:
    row_labels, col_labels, matrix = parse_matrix(csv_path)

    row_coords = run_tsne(matrix, perplexity=perplexity, seed=seed) if embedding in {"both", "rows"} else None
    col_coords = run_tsne(matrix.T, perplexity=perplexity, seed=seed) if embedding in {"both", "cols"} else None

    target_dir = out_dir if out_dir is not None else csv_path.parent
    row_out = target_dir / f"{csv_path.stem}_rows_tsne.png"
    col_out = target_dir / f"{csv_path.stem}_cols_tsne.png"

    if row_coords is not None:
        plot_embedding(
            row_coords,
            row_labels,
            f"t-SNE (rows): {csv_path.stem}",
            row_out,
            show_point_labels=show_point_labels,
            show_density=show_density,
            point_size=point_size,
            point_alpha=point_alpha,
            figsize=figsize,
            dpi=dpi,
            title_fontsize=title_fontsize,
            axis_fontsize=axis_fontsize,
            tick_fontsize=tick_fontsize,
            legend_fontsize=legend_fontsize,
            legend_title_fontsize=legend_title_fontsize,
        )

    if col_coords is not None:
        plot_embedding(
            col_coords,
            col_labels,
            f"t-SNE (columns): {csv_path.stem}",
            col_out,
            show_point_labels=show_point_labels,
            show_density=show_density,
            point_size=point_size,
            point_alpha=point_alpha,
            figsize=figsize,
            dpi=dpi,
            title_fontsize=title_fontsize,
            axis_fontsize=axis_fontsize,
            tick_fontsize=tick_fontsize,
            legend_fontsize=legend_fontsize,
            legend_title_fontsize=legend_title_fontsize,
        )

    return (row_out if row_coords is not None else None), (col_out if col_coords is not None else None)


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
    parser.add_argument(
        "--embedding",
        choices=("both", "rows", "cols"),
        default="both",
        help="Which embedding(s) to render.",
    )
    parser.add_argument(
        "--hide-point-labels",
        action="store_true",
        help="Do not annotate every point label on the plot.",
    )
    parser.add_argument(
        "--show-density",
        action="store_true",
        help="Overlay a light hexbin density map behind points.",
    )
    parser.add_argument("--point-size", type=float, default=52.0)
    parser.add_argument("--point-alpha", type=float, default=0.95)
    parser.add_argument("--fig-width", type=float, default=10.0)
    parser.add_argument("--fig-height", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--title-fontsize", type=int, default=14)
    parser.add_argument("--axis-fontsize", type=int, default=12)
    parser.add_argument("--tick-fontsize", type=int, default=10)
    parser.add_argument("--legend-fontsize", type=int, default=10)
    parser.add_argument("--legend-title-fontsize", type=int, default=11)
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Preset for publication figures: cols only, no per-point labels, larger text and higher DPI.",
    )
    args = parser.parse_args()

    input_path = args.input_path
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    embedding = args.embedding
    show_point_labels = not args.hide_point_labels
    show_density = args.show_density
    point_size = args.point_size
    point_alpha = args.point_alpha
    fig_width = args.fig_width
    fig_height = args.fig_height
    dpi = args.dpi
    title_fontsize = args.title_fontsize
    axis_fontsize = args.axis_fontsize
    tick_fontsize = args.tick_fontsize
    legend_fontsize = args.legend_fontsize
    legend_title_fontsize = args.legend_title_fontsize

    if args.publication:
        embedding = "cols"
        show_point_labels = False
        show_density = True
        point_size = 68.0
        dpi = max(dpi, 320)
        fig_width = max(fig_width, 12.0)
        fig_height = max(fig_height, 9.0)
        title_fontsize = max(title_fontsize, 18)
        axis_fontsize = max(axis_fontsize, 16)
        tick_fontsize = max(tick_fontsize, 13)
        legend_fontsize = max(legend_fontsize, 12)
        legend_title_fontsize = max(legend_title_fontsize, 13)

    generated = 0
    for csv_path in iter_input_csvs(input_path):
        row_out, col_out = process_csv(
            csv_path,
            out_dir=args.out_dir,
            perplexity=args.perplexity,
            seed=args.seed,
            embedding=embedding,
            show_point_labels=show_point_labels,
            show_density=show_density,
            point_size=point_size,
            point_alpha=point_alpha,
            figsize=(fig_width, fig_height),
            dpi=dpi,
            title_fontsize=title_fontsize,
            axis_fontsize=axis_fontsize,
            tick_fontsize=tick_fontsize,
            legend_fontsize=legend_fontsize,
            legend_title_fontsize=legend_title_fontsize,
        )
        if row_out is not None:
            print(row_out)
            generated += 1
        if col_out is not None:
            print(col_out)
            generated += 1

    if generated == 0:
        raise ValueError(f"No matrix CSV files found in: {input_path}")


if __name__ == "__main__":
    main()