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
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


MEAN_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)")
PAPER_CMAP = LinearSegmentedColormap.from_list(
    "jax_aht_paper_rdylgn",
    ["#A50026", "#FFFFBF", "#006837"],
)


def tuple_text(*parts: str) -> str:
    return "(" + ", ".join(parts) + ")"


def format_row_label(label: str) -> str:
    label = label.strip()
    match = re.fullmatch(r"ippo_mlp(?:_s2c\d+)? \(([^)]+)\)", label)
    if match:
        return f"IPPO {tuple_text(*[p.strip() for p in match.group(1).split(',')])}"
    match = re.fullmatch(r"ippo_mlp_cc \(([^)]+)\)", label)
    if match:
        return f"IPPO-cc ({match.group(1)})"
    match = re.fullmatch(r"ippo_mlp_pass \(([^)]+)\)", label)
    if match:
        return f"IPPO-pass ({match.group(1)})"
    match = re.fullmatch(r"brdiv-conf(\d*) \(([^)]+)\)", label)
    if match:
        suffix = match.group(1)
        name = f"BRDiv{suffix}" if suffix else "BRDiv"
        return f"{name} ({match.group(2)})"
    match = re.fullmatch(r"lbrdiv-conf \(([^)]+)\)", label)
    if match:
        return f"lBRDiv {tuple_text(*[p.strip() for p in match.group(1).split(',')])}"
    match = re.fullmatch(r"comedi \(([^)]+)\)", label)
    if match:
        return f"CoMeDi {tuple_text(*[p.strip() for p in match.group(1).split(',')])}"
    for prefix, formatted in (
        ("independent_agent_", "Ind-"),
        ("onion_agent_", "Onion-"),
        ("plate_agent_", "Plate-"),
    ):
        if label.startswith(prefix):
            return formatted + label.removeprefix(prefix).replace("_", ".")
    label_map = {
        "seq_agent_lexi": "Seq-Lexi",
        "seq_agent_rlexi": "Seq-RLexi",
        "seq_agent_col": "Seq-Col",
        "seq_agent_rcol": "Seq-RCol",
        "seq_agent_nearest": "Seq-Nearest",
        "seq_agent_farthest": "Seq-Farthest",
        "entitled_agent": "Entitled",
        "greedy_closest_teammate": "Greedy-closest",
        "greedy_lowest_level": "Greedy-lowest",
        "greedy_highest_level": "Greedy-highest",
        "human_proxy": "Human",
    }
    return label_map.get(label, label)


def format_col_core(core: str) -> tuple[str, bool]:
    match = re.fullmatch(r"ippo_mlp_(\d+)", core)
    if match:
        return f"IPPO-{match.group(1)}", True
    match = re.fullmatch(r"ippo_mlp_s2c\d+_(\d+)_(\d+)", core)
    if match:
        return f"IPPO {tuple_text(match.group(1), match.group(2))}", False
    match = re.fullmatch(r"ippo_mlp_cc_(\d+)", core)
    if match:
        return f"IPPO-cc-{match.group(1)}", True
    match = re.fullmatch(r"ippo_mlp_pass_(\d+)", core)
    if match:
        return f"IPPO-pass-{match.group(1)}", True
    match = re.fullmatch(r"brdiv_conf(\d*)_(\d+)", core)
    if match:
        suffix = match.group(1)
        name = f"BRDiv{suffix}" if suffix else "BRDiv"
        return f"{name}-{match.group(2)}", True
    match = re.fullmatch(r"lbrdiv_conf_(\d+)_(\d+)", core)
    if match:
        return f"lBRDiv-br {tuple_text(match.group(1), match.group(2))}", False
    match = re.fullmatch(r"comedi_(\d+)_(\d+)", core)
    if match:
        return f"CoMeDi-br {tuple_text(match.group(1), match.group(2))}", False
    for prefix, formatted in (
        ("independent_agent_", "Ind-"),
        ("onion_agent_", "Onion-"),
        ("plate_agent_", "Plate-"),
    ):
        if core.startswith(prefix):
            return formatted + core.removeprefix(prefix).replace("_", "."), True
    label_map = {
        "seq_agent_lexi": "Seq-Lexi",
        "seq_agent_rlexi": "Seq-RLexi",
        "seq_agent_col": "Seq-Col",
        "seq_agent_rcol": "Seq-RCol",
        "seq_agent_nearest": "Seq-Nearest",
        "seq_agent_farthest": "Seq-Farthest",
        "entitled_agent": "Entitled",
        "greedy_closest_teammate": "Greedy-closest",
        "greedy_lowest_level": "Greedy-lowest",
        "greedy_highest_level": "Greedy-highest",
        "human_proxy": "Human",
    }
    return label_map.get(core, core), True


def format_col_label(label: str) -> str:
    label = label.strip()
    if label.startswith("br_for_"):
        label = label.removeprefix("br_for_")
    match = re.fullmatch(r"(.+) \(([^)]+)\)", label)
    core = match.group(1) if match else label
    ckpt = match.group(2) if match else None
    formatted, keep_ckpt = format_col_core(core)
    if ckpt is not None and keep_ckpt:
        formatted = f"{formatted} ({ckpt})"
    return f"BR {formatted}"


def parse_matrix(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

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
    cmap: str | LinearSegmentedColormap = PAPER_CMAP,
    cbar_label: str = "Mean return",
    vmin: float | None = None,
    vmax: float | None = None,
):
    formatted_cols = [format_col_label(label) for label in col_labels]
    formatted_rows = [format_row_label(label) for label in row_labels]
    fig_width = max(5.0, len(col_labels) * 0.32 + 2.2)
    fig_height = max(4.0, len(row_labels) * 0.28 + 1.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=6)
    ax.set_xlabel("Best-response policy", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("Held-out teammates", fontsize=10, fontweight="bold", labelpad=2)
    ax.set_xticks(np.arange(len(formatted_cols)), labels=formatted_cols)
    ax.set_yticks(np.arange(len(formatted_rows)), labels=formatted_rows)
    ax.tick_params(axis="x", labelrotation=45, labelsize=6, pad=1)
    ax.tick_params(axis="y", labelsize=6, pad=2)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        text_size = 5 if max(values.shape) > 20 else 5.5
        threshold = (float(np.nanmax(values)) + float(np.nanmin(values))) / 2
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                color = "white" if value > threshold else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=text_size, color=color)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)
    cbar.ax.set_ylabel(cbar_label, rotation=90, va="center", labelpad=8, fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def generate_heatmap_from_csv(
    csv_path: Path,
    title: str | None = None,
    out_path: Path | None = None,
    annotate: bool = True,
    cmap: str | LinearSegmentedColormap = PAPER_CMAP,
    cbar_label: str = "Mean return",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    row_labels, col_labels, values = parse_matrix(csv_path)
    resolved_out_path = out_path or csv_path.with_suffix(".pdf")
    resolved_title = title or csv_path.stem
    plot_heatmap(
        row_labels,
        col_labels,
        values,
        resolved_title,
        resolved_out_path,
        annotate,
        cmap,
        cbar_label,
        vmin,
        vmax,
    )
    return resolved_out_path


def iter_csvs(input_path: Path) -> Iterable[Path]:
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
        help="Path to a CSV file or a directory containing XP CSV files.",
    )
    parser.add_argument("--title", default="XP Matrix Heatmap")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--no-annot",
        action="store_true",
        help="Disable per-cell numeric annotations.",
    )
    parser.add_argument(
        "--cmap",
        default=None,
        help="Optional matplotlib colormap name. Defaults to the paper-style red-yellow-green palette.",
    )
    parser.add_argument(
        "--cbar-label",
        default="Mean return",
        help="Colorbar label.",
    )
    parser.add_argument("--vmin", type=float, default=None, help="Optional lower color scale bound.")
    parser.add_argument("--vmax", type=float, default=None, help="Optional upper color scale bound.")
    args = parser.parse_args()

    input_path = args.input_path
    annotate = not args.no_annot
    cmap = args.cmap or PAPER_CMAP

    if input_path.is_file():
        out_path = generate_heatmap_from_csv(
            input_path,
            title=args.title,
            out_path=args.out,
            annotate=annotate,
            cmap=cmap,
            cbar_label=args.cbar_label,
            vmin=args.vmin,
            vmax=args.vmax,
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
            cmap=cmap,
            cbar_label=args.cbar_label,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        print(out_path)
        generated += 1

    if generated == 0:
        raise ValueError(f"No CSV files found in directory: {input_path}")


if __name__ == "__main__":
    main()
