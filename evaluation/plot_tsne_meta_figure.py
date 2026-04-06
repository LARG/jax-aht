"""Create a publication-ready meta-figure with t-SNE column embeddings per task.

Each input CSV contributes one subplot. The script uses XP matrix columns as points,
colors points by inferred agent type, and supports optional density overlays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from evaluation.plot_xp_csv_tsne import AGENT_TYPE_COLORS, infer_agent_type, parse_matrix, run_tsne


def _compute_grid(n_plots: int, ncols: int) -> tuple[int, int]:
    cols = max(1, min(ncols, n_plots))
    rows = int(np.ceil(n_plots / cols))
    return rows, cols


def _kde_backdrop(
    coords: np.ndarray,
    gridsize: int,
    bandwidth_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = coords[:, 0]
    y = coords[:, 1]

    x_pad = max(1e-9, 0.08 * (x.max() - x.min() + 1e-9))
    y_pad = max(1e-9, 0.08 * (y.max() - y.min() + 1e-9))

    x_grid = np.linspace(x.min() - x_pad, x.max() + x_pad, gridsize)
    y_grid = np.linspace(y.min() - y_pad, y.max() + y_pad, gridsize)
    xx, yy = np.meshgrid(x_grid, y_grid)

    std_x = float(np.std(x) + 1e-9)
    std_y = float(np.std(y) + 1e-9)
    n = max(1, len(x))
    silverman = n ** (-1.0 / 6.0)
    h_x = max(1e-9, bandwidth_scale * silverman * std_x)
    h_y = max(1e-9, bandwidth_scale * silverman * std_y)

    dx = (xx[..., None] - x[None, None, :]) / h_x
    dy = (yy[..., None] - y[None, None, :]) / h_y
    z = np.exp(-0.5 * (dx * dx + dy * dy)).sum(axis=2)
    z /= (2.0 * np.pi * h_x * h_y * n)
    return xx, yy, z


def _preset_defaults(preset: str) -> dict[str, float | int]:
    if preset == "journal-2col":
        return {
            "fig_width": 14.0,
            "fig_height": 8.0,
            "dpi": 420,
            "title_fontsize": 16,
            "axis_fontsize": 13,
            "tick_fontsize": 10,
            "legend_fontsize": 10,
            "legend_title_fontsize": 11,
            "point_size": 40.0,
        }
    if preset == "poster":
        return {
            "fig_width": 36.0,
            "fig_height": 21.0,
            "dpi": 420,
            "title_fontsize": 30,
            "axis_fontsize": 22,
            "tick_fontsize": 16,
            "legend_fontsize": 17,
            "legend_title_fontsize": 18,
            "point_size": 90.0,
        }

    # Default: journal-wide
    return {
        "fig_width": 18.0,
        "fig_height": 10.0,
        "dpi": 420,
        "title_fontsize": 18,
        "axis_fontsize": 14,
        "tick_fontsize": 11,
        "legend_fontsize": 11,
        "legend_title_fontsize": 12,
        "point_size": 52.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csvs",
        type=Path,
        nargs="+",
        help="CSV files to include as subplots (one task per CSV).",
    )
    parser.add_argument(
        "--titles",
        nargs="*",
        default=None,
        help="Optional subplot titles (must match number of CSVs if provided).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=None)
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument(
        "--preset",
        choices=("journal-wide", "journal-2col", "poster"),
        default="journal-wide",
        help="Figure style preset.",
    )
    parser.add_argument("--fig-width", type=float, default=None)
    parser.add_argument("--fig-height", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--point-size", type=float, default=None)
    parser.add_argument("--point-alpha", type=float, default=0.95)
    parser.add_argument(
        "--show-density",
        action="store_true",
        help="Overlay a light hexbin density map in each subplot.",
    )
    parser.add_argument(
        "--density-backdrop",
        choices=("none", "hexbin", "kde"),
        default="none",
        help="Background density style behind points.",
    )
    parser.add_argument(
        "--density-contours",
        action="store_true",
        help="Draw thin contour lines over the density backdrop (best with kde).",
    )
    parser.add_argument("--density-gridsize", type=int, default=70)
    parser.add_argument("--density-bandwidth-scale", type=float, default=1.0)
    parser.add_argument(
        "--inset-xp-heatmap",
        action="store_true",
        help="Draw a small XP-matrix heatmap inset in each subplot.",
    )
    parser.add_argument(
        "--inset-size",
        type=float,
        default=32.0,
        help="Inset XP heatmap size as a percentage of subplot width.",
    )
    parser.add_argument(
        "--suptitle",
        default="BR Policy Embeddings by Task (t-SNE, columns)",
    )
    parser.add_argument("--title-fontsize", type=int, default=None)
    parser.add_argument("--axis-fontsize", type=int, default=None)
    parser.add_argument("--tick-fontsize", type=int, default=None)
    parser.add_argument("--legend-fontsize", type=int, default=None)
    parser.add_argument("--legend-title-fontsize", type=int, default=None)
    parser.add_argument(
        "--no-subplot-axis-ticks",
        action="store_true",
        help="Hide all subplot tick marks and tick labels.",
    )
    parser.add_argument(
        "--hide-subplot-axis-labels",
        action="store_true",
        help="Hide per-subplot axis labels and use shared global labels instead.",
    )
    parser.add_argument("--global-xlabel", default="t-SNE 1")
    parser.add_argument("--global-ylabel", default="t-SNE 2")
    parser.add_argument("--wspace", type=float, default=0.08)
    parser.add_argument("--hspace", type=float, default=0.18)
    parser.add_argument("--margin-left", type=float, default=0.06)
    parser.add_argument("--margin-right", type=float, default=0.995)
    parser.add_argument("--margin-top", type=float, default=0.90)
    parser.add_argument("--margin-bottom", type=float, default=0.14)
    parser.add_argument("--title-pad", type=float, default=6.0)
    args = parser.parse_args()

    preset = _preset_defaults(args.preset)
    fig_width = args.fig_width if args.fig_width is not None else float(preset["fig_width"])
    fig_height = args.fig_height if args.fig_height is not None else float(preset["fig_height"])
    dpi = args.dpi if args.dpi is not None else int(preset["dpi"])
    point_size = args.point_size if args.point_size is not None else float(preset["point_size"])
    title_fontsize = args.title_fontsize if args.title_fontsize is not None else int(preset["title_fontsize"])
    axis_fontsize = args.axis_fontsize if args.axis_fontsize is not None else int(preset["axis_fontsize"])
    tick_fontsize = args.tick_fontsize if args.tick_fontsize is not None else int(preset["tick_fontsize"])
    legend_fontsize = args.legend_fontsize if args.legend_fontsize is not None else int(preset["legend_fontsize"])
    legend_title_fontsize = (
        args.legend_title_fontsize
        if args.legend_title_fontsize is not None
        else int(preset["legend_title_fontsize"])
    )

    csv_paths = [p.resolve() for p in args.input_csvs]
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise ValueError(f"Missing CSV: {csv_path}")

    if args.titles is not None and len(args.titles) > 0 and len(args.titles) != len(csv_paths):
        raise ValueError("--titles must be omitted or have the same count as input CSVs.")

    titles = (
        args.titles
        if args.titles is not None and len(args.titles) > 0
        else [csv_path.parent.parent.parent.name for csv_path in csv_paths]
    )

    n_plots = len(csv_paths)
    nrows, ncols = _compute_grid(n_plots, args.ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False)

    present_types: set[str] = set()

    for i, (csv_path, title) in enumerate(zip(csv_paths, titles)):
        ax = axes[i // ncols][i % ncols]

        _, col_labels, matrix = parse_matrix(csv_path)
        coords = run_tsne(matrix.T, perplexity=args.perplexity, seed=args.seed)
        agent_types = [infer_agent_type(label) for label in col_labels]
        point_colors = [AGENT_TYPE_COLORS.get(agent_type, AGENT_TYPE_COLORS["other"]) for agent_type in agent_types]

        if args.density_backdrop == "hexbin" or (args.show_density and args.density_backdrop == "none"):
            ax.hexbin(
                coords[:, 0],
                coords[:, 1],
                gridsize=26,
                cmap="Greys",
                bins="log",
                mincnt=1,
                alpha=0.18,
                linewidths=0,
                zorder=1,
            )
        elif args.density_backdrop == "kde":
            xx, yy, z = _kde_backdrop(
                coords,
                gridsize=args.density_gridsize,
                bandwidth_scale=args.density_bandwidth_scale,
            )
            ax.contourf(xx, yy, z, levels=10, cmap="Blues", alpha=0.25, zorder=1)
            if args.density_contours:
                ax.contour(xx, yy, z, levels=6, colors="#5f6f8f", linewidths=0.7, alpha=0.45, zorder=2)

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=point_size,
            c=point_colors,
            alpha=args.point_alpha,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

        if args.inset_xp_heatmap:
            inset = inset_axes(ax, width=f"{args.inset_size}%", height=f"{args.inset_size}%", loc="upper right")
            inset.imshow(matrix, cmap="viridis", aspect="auto", interpolation="nearest")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title("XP", fontsize=max(8, int(title_fontsize * 0.45)), pad=2)

        ax.set_title(title, fontsize=title_fontsize, pad=args.title_pad)
        if not args.hide_subplot_axis_labels:
            ax.set_xlabel(args.global_xlabel, fontsize=axis_fontsize)
            ax.set_ylabel(args.global_ylabel, fontsize=axis_fontsize)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")

        if args.no_subplot_axis_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.tick_params(axis="both", labelsize=tick_fontsize)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#666")

        for agent_type in agent_types:
            present_types.add(agent_type)

    for j in range(n_plots, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    legend_order = ["heuristic", "ippo", "comedi", "lbrdiv", "brdiv", "other"]
    present_ordered = [agent_type for agent_type in legend_order if agent_type in present_types]
    if present_ordered:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=agent_type,
                markerfacecolor=AGENT_TYPE_COLORS[agent_type],
                markersize=12,
            )
            for agent_type in present_ordered
        ]
        fig.legend(
            handles=handles,
            title="Agent type",
            loc="lower center",
            ncol=min(len(handles), 6),
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            frameon=False,
            bbox_to_anchor=(0.5, max(0.01, args.margin_bottom - 0.05)),
        )

    fig.suptitle(args.suptitle, fontsize=max(title_fontsize + 4, 18), y=min(0.998, args.margin_top + 0.08))
    if args.hide_subplot_axis_labels:
        fig.supxlabel(args.global_xlabel, fontsize=axis_fontsize)
        fig.supylabel(args.global_ylabel, fontsize=axis_fontsize)

    fig.subplots_adjust(
        left=args.margin_left,
        right=args.margin_right,
        top=args.margin_top,
        bottom=args.margin_bottom,
        wspace=args.wspace,
        hspace=args.hspace,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(args.out)


if __name__ == "__main__":
    main()
