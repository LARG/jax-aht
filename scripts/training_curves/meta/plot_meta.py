"""Render the meta-plot from `meta_data.pkl`.

Layout — 3 rows x 3 section bands, one band per section:

                 Teammate Generation | Unified Algorithms      | Ego Algorithms
  Row 1          FCP + BRDiv (XP)    | TrajeDi                 | PPO Ego
  Row 2          LBRDiv (XP | LMs)   | ROTATE regret | COLE XP | LIAM (ret | recon)
  Row 3          CoMeDi              | ROTATE ret | COLE curve | MeLIBA (ret | AE)

In the unified band ROTATE and COLE occupy side-by-side half-band columns
spanning rows 2-3, each stacking its two panels vertically.

Each band holds exactly three algorithm boxes, so every cell is used; see
the Layout section below for why the bands are columns and how the column
grid is built.

Section headers are anchored to each band's top panel; row labels
(algorithm names) are bold fig-text above each box. Per-panel titles are
Title-Cased.

Curves that concatenate per-teammate training passes (CoMeDi, COLE,
ROTATE) get a teammate-index x-axis instead of raw env steps — see
`_teammate_index_xaxis`.

Run with:
    python -m scripts.training_curves.meta.plot_meta
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# Artifacts live under results/ (gitignored); this module lives under scripts/
# so it is version-controlled. Path is repo-relative, matching
# `common.DEFAULT_CACHE_DIR`, so run from the repo root.
OUT_DIR = Path("results/figures/training_curves/aggregate_2")
DEFAULT_PICKLE = OUT_DIR / "meta_data.pkl"
DEFAULT_OUT = OUT_DIR / "meta_plot.png"

TEAMMATE_SET_COLORS = {
    "fcp":     "tab:blue",
    "comedi":  "tab:orange",
    "rotate":  "tab:green",
    "lbrdiv":  "tab:red",
    "trajedi": "tab:purple",
    "brdiv":   "tab:brown",
    "cole":    "tab:pink",
    "unknown": "gray",
}

LM_HORIZONTAL_COLOR = "tab:orange"
LM_VERTICAL_COLOR = "tab:green"

# Typography. The imported plot_globals sizes were set for a figure with far
# fewer, much larger panels. Tick and axis labels overhang each panel to the
# left, and that overhang is fixed in points, so it — not the data — sets how
# tightly the bands can be packed. These are scaled down so the layout can
# compress; everything else derives from them.
PANEL_TITLE_SIZE = 13
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 8.5
ROW_LABEL_SIZE = 16
SECTION_HEADER_SIZE = 19

plt.rcParams["xtick.labelsize"] = TICK_LABEL_SIZE
plt.rcParams["ytick.labelsize"] = TICK_LABEL_SIZE


def _stderr(values: np.ndarray) -> np.ndarray:
    return values.std(axis=0, ddof=1) / max(1.0, np.sqrt(values.shape[0]))


def _empty(ax, msg: str = "no data") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color="gray", fontsize=10)
    ax.axis("off")


def _set_panel_title(ax, title: str | None, pad: float = 8) -> None:
    if title:
        ax.set_title(title, fontsize=PANEL_TITLE_SIZE, pad=pad)


def fit_panel_titles(fig, axd, min_size: float = 9.0) -> None:
    """Shrink any panel title that renders wider than its panel.

    Titles are centred on the axes and otherwise unconstrained, so they are the
    one element that can silently overhang into a neighbouring panel — which is
    how two panels can collide even when their axes and labels fit fine. Titles
    are kept short by hand; this is the backstop for when a panel is narrowed.

    Call after layout, once titles have a renderer to measure against.
    """
    renderer = fig.canvas.get_renderer()
    fig_w = fig.get_size_inches()[0]
    for ax in axd.values():
        title = ax.title
        if not title.get_text():
            continue
        width_in = title.get_window_extent(renderer).width / fig.dpi
        avail_in = ax.get_position().width * fig_w
        if width_in > avail_in:
            scale = avail_in / width_in
            title.set_size(max(min_size, title.get_size() * scale))


def _standardize_return_axis(ax, axis: str = "y") -> None:
    """Standard return axis: limits 0.0..0.5 with one-decimal ticks every 0.1."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    if axis == "y":
        ax.set_ylim(0.0, 0.5)
    target.set_major_locator(mticker.FixedLocator(np.linspace(0.0, 0.5, 6)))
    target.set_major_formatter(mticker.FormatStrFormatter("%.1f"))


def _teammate_index_xaxis(ax, entry: dict, max_labels: int = 8) -> bool:
    """Two-level x-axis for curves built by concatenating per-teammate passes.

    CoMeDi/COLE train one new partner per iteration and ROTATE runs one
    open-ended iteration per confederate, so the flattened curve is really a
    sequence of independent training runs. Raw env steps hide that structure —
    the sawtooth in those panels is one tooth per teammate.

    Coarse level: labelled major ticks at segment centers giving the teammate
    index, subsampled to at most `max_labels` labels, with faint separators at
    the boundaries. Fine level: unlabelled minor ticks inside each segment for
    training progress, plus the per-teammate env-step budget in the axis label.

    A second full env-step axis was tried and rejected — it rendered outside the
    algorithm box and collided with the next section's header.

    Returns False and leaves the axis alone when the entry has no segment
    structure (e.g. TrajeDi), so callers can fall back to a plain steps axis.
    """
    n_segments = entry.get("n_segments")
    if not n_segments or n_segments < 2:
        return False
    env_steps = np.asarray(entry["env_steps"])
    total = float(env_steps[-1])
    seg = total / n_segments

    stride = max(1, int(np.ceil(n_segments / max_labels)))
    idx = np.arange(0, n_segments, stride)
    ax.set_xticks((idx + 0.5) * seg)
    ax.set_xticklabels([str(i) for i in idx])
    ax.set_xlim(0, total)
    # Minor ticks quarter each segment: enough to read progress within a
    # teammate's training pass without implying step-level precision.
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(seg / 4))
    ax.tick_params(axis="x", which="minor", length=2, color="0.6")
    for k in range(1, n_segments):
        ax.axvline(k * seg, color="0.85", linewidth=0.4, zorder=0)
    ax.set_xlabel(f"Teammate Index  ({seg / 1e6:.1f}e6 steps ea.)",
                  fontsize=AXIS_LABEL_SIZE - 1)
    return True


def _envstep_xaxis_in_millions(ax) -> None:
    """X-axis labelled `1.0`, `2.0`, ... with the `1e6` multiplier in the xlabel."""
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{x / 1e6:.1f}")
    )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.set_xlabel("Env Steps (1e6)", fontsize=AXIS_LABEL_SIZE)


# -----------------------------------------------------------------------------
# Per-panel renderers
# -----------------------------------------------------------------------------

def render_ego(ax, by_set: dict, title: str | None = None):
    if not by_set:
        _empty(ax)
        return
    for teammate_set, e in sorted(by_set.items()):
        env_steps = np.asarray(e["env_steps"])
        values = np.asarray(e["values"])
        mean = values.mean(axis=0)
        color = TEAMMATE_SET_COLORS.get(teammate_set, "gray")
        ax.plot(env_steps, mean, color=color, linewidth=1.8,
                label=teammate_set)
        sem = _stderr(values)
        ax.fill_between(env_steps, mean - sem, mean + sem, color=color, alpha=0.2)
    _set_panel_title(ax, title)
    ax.set_ylabel("Return", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("Env Steps", fontsize=AXIS_LABEL_SIZE)
    _standardize_return_axis(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE, loc="lower right")


def render_ego_losses(ax, by_set: dict, loss_keys: tuple[str, ...],
                      title: str | None = None, log_y: bool = False,
                      legend_loc: str = "best"):
    if not by_set:
        _empty(ax, "no losses")
        return
    linestyles = ["-", "--", ":", "-."]
    plotted = False
    for teammate_set, e in sorted(by_set.items()):
        losses = e.get("losses", {})
        env_steps = np.asarray(e["env_steps"])
        for li, k in enumerate(loss_keys):
            if k not in losses:
                continue
            arr = np.asarray(losses[k])
            mean = arr.mean(axis=0)
            color = TEAMMATE_SET_COLORS.get(teammate_set, "gray")
            ls = linestyles[li % len(linestyles)]
            label = (f"{teammate_set} · {k}" if len(loss_keys) > 1
                     else f"{teammate_set}")
            ax.plot(env_steps, mean, color=color, linewidth=1.5,
                    linestyle=ls, label=label)
            plotted = True
    if not plotted:
        _empty(ax, "no autoencoder losses")
        return
    if log_y:
        ax.set_yscale("log")
    # Log-decade labels ("10^-4") are the widest tick text of any panel, and
    # these panels always sit to the right of another one, so the labels eat
    # into the gap. Shrinking them keeps that gap clear.
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE - 1)
    _set_panel_title(ax, title)
    # No ylabel: both loss panels sit to the right of another panel, and their
    # log-decade tick labels are already the widest in the figure. The panel
    # title names the quantity, so the ylabel is redundant width.
    ax.set_xlabel("Env Steps", fontsize=AXIS_LABEL_SIZE)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=LEGEND_SIZE - 0.5, loc=legend_loc)


def render_fcp_partners(ax, fcp_partners: dict | None, title: str | None = None):
    if fcp_partners is None:
        _empty(ax, "no FCP partners")
        return
    env_steps = np.asarray(fcp_partners["env_steps"])
    values = np.asarray(fcp_partners["values"])
    n = values.shape[0]
    for r in range(n):
        ax.plot(env_steps, values[r], color="tab:blue", alpha=0.10, linewidth=0.5)
    ax.plot(env_steps, values.mean(axis=0), color="black", linewidth=1.8,
            label="mean")
    _set_panel_title(ax, title)
    ax.set_ylabel("Return", fontsize=AXIS_LABEL_SIZE)
    _standardize_return_axis(ax)
    _envstep_xaxis_in_millions(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE, loc="lower right")


def render_xp_matrix(ax, entry: dict | None, title: str | None = None,
                     vmin: float = 0.0, vmax: float = 0.5,
                     show_colorbar: bool = True,
                     colorbar_horizontal: bool = False,
                     square: bool = False):
    """Heatmap with a red→yellow→green colormap pinned to [vmin, vmax].

    Defaults match LBF return scale (0–0.5). Override `vmax` for Overcooked
    or other tasks with larger returns.

    Stores the colorbar axes on `ax._cbar_ax` so `draw_algo_boxes` can include
    it when measuring the panel's extent.
    """
    if entry is None:
        _empty(ax, "no XP matrix")
        return
    mat = np.asarray(entry["matrix"])
    pop_size = mat.shape[0]
    # Default `aspect="auto"`: the band cells are taller than they are wide, so
    # an equal-aspect matrix would be width-limited and leave most of the cell
    # empty. Stretching is harmless here — the axes are categorical. Pass
    # `square=True` for a cell wide enough to hold an undistorted matrix.
    im = ax.imshow(mat, cmap="RdYlGn", origin="upper",
                   aspect="equal" if square else "auto",
                   vmin=vmin, vmax=vmax)
    if show_colorbar:
        # A horizontal colorbar underneath costs height (of which these cells
        # have plenty) instead of width (of which they have none) — a vertical
        # one overruns the slot and collides with the panel to the right.
        if colorbar_horizontal:
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                                location="bottom", shrink=0.9, fraction=0.06,
                                pad=0.16)
        else:
            cbar = plt.colorbar(im, ax=ax, shrink=0.9, fraction=0.05, pad=0.04)
        cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE - 2)
        ax._cbar_ax = cbar.ax
    if pop_size <= 12:
        ax.set_xticks(range(pop_size))
        ax.set_yticks(range(pop_size))
    else:
        sparse = np.linspace(0, pop_size - 1, num=5, dtype=int)
        ax.set_xticks(sparse)
        ax.set_yticks(sparse)

    decimals = 0 if mat.max() >= 10 else 2
    fmt = f"{{:.{decimals}f}}"
    if pop_size <= 12:
        # Text-colour logic for RdYlGn: cells in the low-value (deep red) tail
        # benefit from white text; everything mid-range or higher reads cleaner
        # with black on yellow/green.
        threshold = vmin + 0.15 * (vmax - vmin)
        for i in range(pop_size):
            for j in range(pop_size):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] < threshold else "black",
                        fontsize=7)
    _set_panel_title(ax, title)


def render_lms_overlay(ax, lms: dict | None, title: str | None = None):
    if lms is None:
        _empty(ax, "no LBRDiv LMs")
        return
    env_steps = np.asarray(lms["env_steps"])
    h = np.asarray(lms["horizontal"])
    v = np.asarray(lms["vertical"])
    for k in range(h.shape[0]):
        ax.plot(env_steps, h[k], color=LM_HORIZONTAL_COLOR, alpha=0.45,
                linewidth=0.9, label="Horizontal" if k == 0 else None)
    for k in range(v.shape[0]):
        ax.plot(env_steps, v[k], color=LM_VERTICAL_COLOR, alpha=0.45,
                linewidth=0.9, label="Vertical" if k == 0 else None)
    _set_panel_title(ax, title)
    ax.set_ylabel("LM Value", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("Env Steps", fontsize=AXIS_LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right", title="LM Type",
              title_fontsize=LEGEND_SIZE, framealpha=0.85)


def render_regret(ax, entry: dict | None, title: str | None = None,
                  color: str = "tab:green"):
    """ROTATE train regret: `average_returns_br - average_returns_ego` per partner update.

    Unlike the return panels this is *not* pinned to the 0–0.5 return axis —
    regret is a difference of returns, so it sits near zero and can go
    negative. A zero reference line makes the sign readable.
    """
    if entry is None:
        _empty(ax, "no regret data")
        return
    env_steps = np.asarray(entry["env_steps"])
    values = np.asarray(entry["values"])
    mean = values.mean(axis=0)
    ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--", zorder=1)
    ax.plot(env_steps, mean, color=color, linewidth=1.8, zorder=3)
    sem = _stderr(values)
    ax.fill_between(env_steps, mean - sem, mean + sem, color=color, alpha=0.18,
                    zorder=2)
    _set_panel_title(ax, title)
    ax.set_ylabel("Regret", fontsize=AXIS_LABEL_SIZE)
    # Two decimals, not three: "-0.025" style labels are the widest tick text in
    # the figure and this panel has a neighbour immediately to its right.
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    if not _teammate_index_xaxis(ax, entry):
        ax.set_xlabel("Env Steps", fontsize=AXIS_LABEL_SIZE)
    ax.grid(True, alpha=0.3, axis="y")


def render_curve(ax, algo: str, entry: dict | None, title: str | None = None,
                 show_ylabel: bool = True):
    if entry is None:
        _empty(ax, f"no curve for {algo}")
        return
    env_steps = np.asarray(entry["env_steps"])
    values = np.asarray(entry["values"])
    mean = values.mean(axis=0)
    color = TEAMMATE_SET_COLORS.get(algo, "tab:blue")
    ax.plot(env_steps, mean, color=color, linewidth=1.8)
    sem = _stderr(values)
    ax.fill_between(env_steps, mean - sem, mean + sem, color=color, alpha=0.18)
    _set_panel_title(ax, title)
    if show_ylabel:
        ax.set_ylabel("Return", fontsize=AXIS_LABEL_SIZE)
    _standardize_return_axis(ax)
    if _teammate_index_xaxis(ax, entry):
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.set_xlabel("Env Steps", fontsize=AXIS_LABEL_SIZE)
        ax.grid(True, alpha=0.3)


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
# The unified section is a full-width block on top; the teammate-generation and
# ego sections sit side by side beneath it. This is a portrait arrangement: the
# previous all-side-by-side version was ~20in wide and squat, which wasted
# vertical space and cramped the panels.
#
# Columns are a single TOTAL_UNITS grid shared by both halves, with an explicit
# spacer between consecutive units and `wspace = 0`. A panel spanning k units
# absorbs the k-1 spacers inside it and occupies 2k-1 grid columns; the spacer
# left over at a panel boundary forms the visible gap. Explicit spacers (rather
# than one shared `wspace`) let the within-section and between-section gaps be
# set independently, which a single wspace cannot do: it would force the gap
# between sections to roughly twice the gap inside one.
TOTAL_UNITS = 12
# Unit boundaries that get the wider BAND_GAP instead of PANEL_GAP, given as
# the index of the unit *after* the boundary. 6 splits the teammate half from
# the ego half. 4 splits ROTATE from the COLE/TrajeDi column in the unified
# block: those two share a left edge, so the gap there has to clear TrajeDi's
# ylabel or ROTATE's box runs into theirs.
WIDE_GAP_UNITS = frozenset({4, 6})

FIG_WIDTH = 14.3
FIG_HEIGHT = 22.1

# Gap between two panels inside a section, relative to a unit column. Must clear
# the ylabel and tick labels of the panel to its right.
PANEL_GAP = 1.85
# Gap between the teammate and ego sections. Kept larger than PANEL_GAP so the
# two read as separate sections; if it drops below the widest within-section
# box gap the hierarchy inverts and they merge into one row.
BAND_GAP = 3.60

# Vertical: two unified rows, a thin spacer row, then three rows shared by the
# teammate and ego sections. The spacer isolates the one gap that has to be
# large — between the unified section box and the two below it, which must also
# clear their headers — from the gaps *inside* a section, which only separate
# algorithm boxes. The spacer has zero height: it is there purely because a
# spacer row incurs an `hspace` gap on each side, which is exactly the doubling
# the section boundary wants (~70px) while rows stay tight (~25px).
ROW_HEIGHTS = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
# Row spacing. Has to exceed BOX_PAD_TOP + BOX_MARGIN_Y so that vertically
# adjacent algorithm boxes do not overlap; the top pad is sized for a row label
# plus a panel title, so this is larger than it looks like it needs to be.
HSPACE = 0.65


def _row(*segments: tuple[str, int]) -> list[str]:
    """Expand [(panel, n_units), ...] into one mosaic row of grid columns."""
    cols: list[str] = []
    for i, (name, units) in enumerate(segments):
        if i:
            cols.append(".")           # boundary spacer -> visible gap
        cols += [name] * (2 * units - 1)
    assert len(cols) == 2 * TOTAL_UNITS - 1, (len(cols), cols)
    return cols


LAYOUT = [
    # --- Unified Algorithms: full-width block ---
    # ROTATE takes the left third across both rows, stacking its two panels;
    # COLE's two panels sit beside it on the first row and TrajeDi's single
    # curve spans the same width on the second.
    _row(("rotate_regret", 4), ("cole_xp", 4), ("cole_curve", 4)),
    _row(("rotate_ret", 4), ("trajedi", 8)),

    # --- spacer isolating the unified block from the two sections below ---
    ["."] * (2 * TOTAL_UNITS - 1),

    # --- Teammate Generation (units 1-6) | Ego Algorithms (units 7-12) ---
    # FCP is compacted so BRDIV's XP matrix can sit beside it with a box of its
    # own, which is what lets the teammate half fit in three rows, not four.
    _row(("fcp", 3), ("brdiv_xp", 3), ("ppo", 6)),
    _row(("lbrdiv_xp", 3), ("lbrdiv_lm", 3), ("liam_ret", 3), ("liam_loss", 3)),
    _row(("comedi", 6), ("meliba_ret", 3), ("meliba_loss", 3)),
]

assert len(ROW_HEIGHTS) == len(LAYOUT)

# Anchor panels for each section header, spanning the full width of the section
# so the header centres on it rather than on one panel.
SECTION_HEADERS = [
    ("Unified Algorithms",  ["rotate_regret", "cole_curve"]),
    ("Teammate Generation", ["fcp", "brdiv_xp"]),
    ("Ego Algorithms",      ["ppo"]),
]

# Which algorithm boxes each section box encloses, keyed by the labels used in
# ALGO_BOXES.
SECTIONS = [
    ("Unified Algorithms",  ["ROTATE", "COLE", "TrajeDi"]),
    ("Teammate Generation", ["FCP", "BRDiv", "LBRDiv", "CoMeDi"]),
    ("Ego Algorithms",      ["PPO Ego", "LIAM Ego", "MeLIBA Ego"]),
]

# Box pads and label offsets below are expressed as figure fractions but were
# tuned against the original 17x21 canvas. Scale them so they keep the same
# *physical* size as the canvas changes, rather than growing with it.
_V = 21 / FIG_HEIGHT
_H = 17 / FIG_WIDTH

ROW_LABELS = [
    ("FCP",        ["fcp"]),
    ("BRDiv",      ["brdiv_xp"]),
    ("LBRDiv",     ["lbrdiv_xp", "lbrdiv_lm"]),
    ("CoMeDi",     ["comedi"]),
    ("TrajeDi",    ["trajedi"]),
    ("COLE",       ["cole_xp", "cole_curve"]),
    ("ROTATE",     ["rotate_ret", "rotate_regret"]),
    ("PPO Ego",    ["ppo"]),
    ("LIAM Ego",   ["liam_ret", "liam_loss"]),
    ("MeLIBA Ego", ["meliba_ret", "meliba_loss"]),
]


def _panel_slot_bbox(fig, ax):
    """Return the panel's *original* gridspec slot in figure coords.

    `ax.get_position()` reflects shrinkage from colorbar attachment, which
    breaks alignment between matrix panels (with colorbars) and curve panels
    (without). The subplotspec position is the pre-shrink slot.
    """
    return ax.get_subplotspec().get_position(fig)


def add_section_header(fig, axd, panels: list[str], label: str,
                       y_offset: float = 0.048 * _V):
    """Place a section header above the first row of a section.

    Anchored to the section's first panel rather than a hard-coded figure
    y, so the header follows the row wherever the layout puts it. Sits above
    `add_row_label`'s offset so the two don't collide.
    """
    bbs = [_panel_slot_bbox(fig, axd[name]) for name in panels]
    x0 = min(b.x0 for b in bbs)
    x1 = max(b.x1 for b in bbs)
    y1 = max(b.y1 for b in bbs)
    return fig.text(x0 + (x1 - x0) / 2, y1 + y_offset, label,
                    ha="center", va="bottom",
                    fontsize=SECTION_HEADER_SIZE, fontweight="bold")


def add_row_label(fig, axd, panels: list[str], label: str,
                  gap_in: float = 0.06):
    """Place a bold row label centered just above the panel(s) and their titles.

    Sits `gap_in` inches above the tallest panel title rather than a fixed
    offset from the gridspec slot. A fixed offset has to assume the worst-case
    title and leaves dead space above every row; since the box top is measured
    from this label, that slack propagated into every vertical gap.
    """
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    slots = [_panel_slot_bbox(fig, axd[name]) for name in panels]
    x0 = min(b.x0 for b in slots)
    x1 = max(b.x1 for b in slots)
    content_top = max(
        inv.transform_bbox(axd[name].get_tightbbox(renderer)).y1
        for name in panels
    )
    y = content_top + gap_in / fig.get_size_inches()[1]
    return fig.text(x0 + (x1 - x0) / 2, y, label,
                    ha="center", va="bottom",
                    fontsize=ROW_LABEL_SIZE, fontweight="bold")


# Top of an algorithm box, measured from its gridspec slot so boxes sharing a
# row line up under their labels. Leaves room for the panel title + row label.
BOX_PAD_TOP = 0.038 * _V
# Margin between a box's other edges and the tight bbox of its contents. These
# are derived from content rather than the slot, so neighbouring boxes in a row
# cannot overlap regardless of how wide their labels are.
BOX_MARGIN_X = 0.09 / FIG_WIDTH
BOX_MARGIN_Y = 0.13 / FIG_HEIGHT
# Padding between a section box and the algorithm boxes it encloses.
SECTION_PAD_X = 0.16 / FIG_WIDTH
SECTION_PAD_Y = 0.16 / FIG_HEIGHT


def reclaim_matrix_margin(fig, axd, matrix: str, curve: str, siblings: list[str],
                          keep_in: float = 0.10):
    """Slide a square XP matrix left into the dead space its box inherits.

    `draw_algo_boxes` snaps boxes starting in the same grid column to a common
    left edge. In a band whose column-0 panels are mostly curves, that edge is
    set by a curve's "Return" ylabel plus its 0.0-0.5 tick labels. An XP matrix
    in the same column carries only narrow integer ticks, so it ends up with a
    wide band of empty box to its left.

    The matrix is already flush against its slot, so the space can only be
    reclaimed after layout: shift the matrix left to sit `keep_in` inches inside
    the shared box edge, then grow `curve` leftwards by the same amount so the
    pair stays balanced and the gap between them is unchanged. Net width of the
    algorithm's box is untouched; the empty margin becomes plot area.
    """
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    left = lambda n: inv.transform_bbox(axd[n].get_tightbbox(renderer)).x0

    shared_edge = min(left(n) for n in [matrix, *siblings])
    shift = left(matrix) - shared_edge - keep_in / fig.get_size_inches()[0]
    if shift <= 0:
        return

    m = axd[matrix].get_position()
    axd[matrix].set_position([m.x0 - shift, m.y0, m.width, m.height])
    # The colorbar is a separate axes positioned against the matrix at creation
    # time, so it has to be translated by the same amount or it is left behind.
    cbar_ax = getattr(axd[matrix], "_cbar_ax", None)
    if cbar_ax is not None:
        b = cbar_ax.get_position()
        cbar_ax.set_position([b.x0 - shift, b.y0, b.width, b.height])
    c = axd[curve].get_position()
    axd[curve].set_position([c.x0 - shift, c.y0, c.width + shift, c.height])


def draw_algo_boxes(fig, axd, algo_boxes: list[tuple[str, list[str]]],
                    row_label_texts: dict | None = None):
    """Draw the thin rectangle around each algorithm's panel(s).

    Bounds start from the panels' *tight* bboxes (axes plus tick labels, axis
    labels and any attached colorbar) with a small uniform margin, so that
    neighbouring boxes in a row never overlap. Top edges come from the gridspec
    slot instead, so boxes sharing a row line up under their row labels.

    Tight bboxes alone leave the boxes visibly ragged: a box whose leftmost
    panel is an XP matrix (narrow integer tick labels) starts further right than
    one fronted by a curve with a "Return" ylabel. So after measuring, boxes
    that begin in the same grid column are snapped to a common left edge, those
    ending in the same column to a common right edge, and those in the same row
    to a common bottom. Grouping is by *slot* coordinate, which is exact,
    rather than by the measured content.
    """
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    specs = []
    for label, panels in algo_boxes:
        tight = []
        for name in panels:
            ax = axd[name]
            tight.append(inv.transform_bbox(ax.get_tightbbox(renderer)))
            cbar_ax = getattr(ax, "_cbar_ax", None)
            if cbar_ax is not None:
                tight.append(inv.transform_bbox(cbar_ax.get_tightbbox(renderer)))
        slots = [_panel_slot_bbox(fig, axd[name]) for name in panels]
        # The row label sits above the panel titles and must be enclosed too.
        text = (row_label_texts or {}).get(label)
        label_top = ([inv.transform_bbox(text.get_window_extent(renderer)).y1]
                     if text is not None else [])
        specs.append({
            "label": label,
            "x0": min(b.x0 for b in tight) - BOX_MARGIN_X,
            "x1": max(b.x1 for b in tight) + BOX_MARGIN_X,
            "y0": min(b.y0 for b in tight) - BOX_MARGIN_Y,
            "y1": max([b.y1 for b in tight] + label_top) + BOX_MARGIN_Y,
            # Slot-space keys identifying which grid edge this box sits on.
            "slot_x0": round(min(s.x0 for s in slots), 4),
            "slot_x1": round(max(s.x1 for s in slots), 4),
            "slot_y0": round(min(s.y0 for s in slots), 4),
            "slot_y1": round(max(s.y1 for s in slots), 4),
        })

    for key, edge, combine in (("slot_x0", "x0", min),
                               ("slot_x1", "x1", max),
                               ("slot_y0", "y0", min),
                               ("slot_y1", "y1", max)):
        groups: dict[float, list[dict]] = {}
        for spec in specs:
            groups.setdefault(spec[key], []).append(spec)
        for members in groups.values():
            shared = combine(m[edge] for m in members)
            for m in members:
                m[edge] = shared

    for s in specs:
        fig.add_artist(mpatches.FancyBboxPatch(
            (s["x0"], s["y0"]), s["x1"] - s["x0"], s["y1"] - s["y0"],
            boxstyle="round,pad=0.002,rounding_size=0.005",
            transform=fig.transFigure,
            fill=False, edgecolor="0.55", linewidth=1.0,
            clip_on=False,
        ))
    return {s["label"]: (s["x0"], s["x1"], s["y0"], s["y1"]) for s in specs}


def draw_section_boxes(fig, algo_rects: dict, header_texts: dict):
    """Enclose each section's algorithm boxes, and its header, in an outer box.

    Nests one level above `draw_algo_boxes`: drawn darker and heavier so the
    grouping reads as section > algorithm > panel. The top is raised to clear
    the section header, which sits above the first row of algorithm boxes.
    """
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for label, members in SECTIONS:
        rects = [algo_rects[m] for m in members if m in algo_rects]
        if not rects:
            continue
        x0 = min(r[0] for r in rects) - SECTION_PAD_X
        x1 = max(r[1] for r in rects) + SECTION_PAD_X
        y0 = min(r[2] for r in rects) - SECTION_PAD_Y
        y1 = max(r[3] for r in rects) + SECTION_PAD_Y
        header = header_texts.get(label)
        if header is not None:
            top = inv.transform_bbox(header.get_window_extent(renderer)).y1
            y1 = max(y1, top + SECTION_PAD_Y)
        fig.add_artist(mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            transform=fig.transFigure,
            fill=False, edgecolor="0.30", linewidth=1.8,
            clip_on=False,
        ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle", default=str(DEFAULT_PICKLE))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    # Unit columns with a spacer between each; the teammate/ego split is wider.
    width_ratios: list[float] = []
    for unit in range(TOTAL_UNITS):
        if unit:
            width_ratios.append(
                BAND_GAP if unit in WIDE_GAP_UNITS else PANEL_GAP)
        width_ratios.append(1.0)

    # Margins are given in inches and converted, so they stay put if the canvas
    # is resized. The top margin has to clear the section header, the row label
    # and the panel title, which all sit above the first row's axes.
    # Top/bottom leave room for the section boxes, which extend past the
    # outermost algorithm boxes to enclose their headers; sized so nothing is
    # clipped at the canvas edge.
    margins_in = {"left": 1.10, "right": 0.45, "top": 1.60, "bottom": 0.90}
    fig, axd = plt.subplot_mosaic(
        LAYOUT,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        empty_sentinel=".",
        gridspec_kw={
            "hspace": HSPACE,
            # All horizontal spacing comes from spacer columns, not wspace.
            "wspace": 0.0,
            "width_ratios": width_ratios,
            "height_ratios": ROW_HEIGHTS,
            "left": margins_in["left"] / FIG_WIDTH,
            "right": 1 - margins_in["right"] / FIG_WIDTH,
            "top": 1 - margins_in["top"] / FIG_HEIGHT,
            "bottom": margins_in["bottom"] / FIG_HEIGHT,
        },
    )

    # ---- Teammate-gen panels (Title-Cased panel headings) ----
    # Titles here are kept short because they are centred and unconstrained;
    # `fit_panel_titles` shrinks any that still overflow.
    render_fcp_partners(axd["fcp"], data.get("fcp_partners"),
                        title="Per-Partner Return")
    # The 3x3 matrices annotate every cell, so their colorbars are redundant —
    # dropping them buys width back in the tightest row of the figure. COLE's
    # 18x18 is unannotated and keeps its colorbar.
    render_xp_matrix(axd["brdiv_xp"], data["xp_matrices"].get("brdiv"),
                     title="Final XP Matrix", show_colorbar=False, square=True)
    render_curve(axd["trajedi"], "trajedi",
                 data["teammate_curves"].get("trajedi"),
                 title="Training Return")
    render_xp_matrix(axd["lbrdiv_xp"], data["xp_matrices"].get("lbrdiv"),
                     title="Final XP Matrix", show_colorbar=False, square=True)
    render_lms_overlay(axd["lbrdiv_lm"], data.get("lbrdiv_lms"),
                       title="Lagrange Multipliers")
    render_curve(axd["comedi"], "comedi", data["teammate_curves"].get("comedi"),
                 title="Partner Training Return")
    render_curve(axd["cole_curve"], "cole", data["teammate_curves"].get("cole"),
                 title="Partner Training Return")
    render_xp_matrix(axd["cole_xp"], data["xp_matrices"].get("cole"),
                     title="Final XP Matrix", colorbar_horizontal=True,
                     square=True)
    rotate = data.get("rotate") or {}
    render_curve(axd["rotate_ret"], "rotate", rotate.get("return"),
                 title="Ego vs. Confederate")
    render_regret(axd["rotate_regret"], rotate.get("train_regret"),
                  title="Train Regret",
                  color=TEAMMATE_SET_COLORS["rotate"])

    # ---- Ego panels ----
    render_ego(axd["ppo"], data["ego"].get("ppo_ego", {}),
               title="Training Return")
    render_ego(axd["liam_ret"], data["ego"].get("liam_ego", {}),
               title="Training Return")
    render_ego_losses(axd["liam_loss"], data["ego"].get("liam_ego", {}),
                      loss_keys=("reconstruction_loss",),
                      title="Reconstruction Loss")
    render_ego(axd["meliba_ret"], data["ego"].get("meliba_ego", {}),
               title="Training Return")
    render_ego_losses(axd["meliba_loss"], data["ego"].get("meliba_ego", {}),
                      loss_keys=("reconstruction_loss", "kl_divergence_loss"),
                      title="Autoencoder Losses",
                      log_y=True, legend_loc="upper right")

    # Finalize layout positions before anything measures bboxes.
    fig.canvas.draw()

    fit_panel_titles(fig, axd)

    # LBRDIV's XP matrix starts its band's column 0 alongside curve panels, so
    # its box inherits a left edge sized for a curve's ylabel. Reclaim that
    # margin for the Lagrange-multiplier panel beside it.
    #
    # Neither other matrix needs this. BRDIV starts mid-row, so its box is
    # already tight. COLE's matrix sits directly above its curve rather than
    # beside it: the two share columns, so shifting the matrix left would pull
    # it out of alignment with the curve underneath.
    reclaim_matrix_margin(fig, axd, "lbrdiv_xp", "lbrdiv_lm",
                          ["fcp", "comedi", "rotate_regret", "rotate_ret"])
    reclaim_matrix_margin(fig, axd, "cole_xp", "cole_curve", ["trajedi"])
    fig.canvas.draw()          # refresh bboxes for the box/label measurements

    # Row labels and section headers (drawn after layout so bboxes are valid).
    row_label_texts = {label: add_row_label(fig, axd, panels, label)
                       for label, panels in ROW_LABELS}
    header_texts = {label: add_section_header(fig, axd, panels, label)
                    for label, panels in SECTION_HEADERS}

    # Group boxes: one rectangle per algorithm enclosing its panel(s) +
    # the row label sitting above. ALGO_BOXES uses the full panel set per algo
    # (LIAM/MeLIBA each have 2 stacked panels — both go in the same box).
    ALGO_BOXES = [
        ("FCP",        ["fcp"]),
        ("BRDiv",      ["brdiv_xp"]),
        ("TrajeDi",    ["trajedi"]),
        ("LBRDiv",     ["lbrdiv_xp", "lbrdiv_lm"]),
        ("CoMeDi",     ["comedi"]),
        ("COLE",       ["cole_xp", "cole_curve"]),
        ("ROTATE",     ["rotate_ret", "rotate_regret"]),
        ("PPO Ego",    ["ppo"]),
        ("LIAM Ego",   ["liam_ret", "liam_loss"]),
        ("MeLIBA Ego", ["meliba_ret", "meliba_loss"]),
    ]
    algo_rects = draw_algo_boxes(fig, axd, ALGO_BOXES, row_label_texts)
    draw_section_boxes(fig, algo_rects, header_texts)


    # Save both PNG (quick preview) and PDF (vector for paper/sharing).
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    print(f"[plot] wrote {pdf_path}")


if __name__ == "__main__":
    main()
