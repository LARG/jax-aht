"""Structural feature screening per Overcooked task; writes one drop list per task.

    python scripts/convention_analysis/screen_features.py --features features_derived.csv:coord_ring \
        --cell-support cell_support.csv:coord_ring --output screen.md --drop-list-dir drop_lists
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts.convention_analysis.feature_groups import (
    CONTENTION, MIN_EVENT_SUPPORT, PROXIMITY_CONDITIONED, RATIO_NUMERATOR,
)

# cell-coordinate feature -> (support basis in cell_support.csv, axis)
CELL_VALUED_COLUMNS: Dict[str, tuple] = {
    "onion_counter_x": ("modal_onion", "x"), "onion_counter_y": ("modal_onion", "y"),
    "dish_counter_x": ("modal_dish", "x"), "dish_counter_y": ("modal_dish", "y"),
}


def axis_support(support: pd.DataFrame, basis: str, axis: str) -> List[int]:
    """Distinct values of `axis` across teammates for one support basis."""
    col = f"{basis}_{axis}"
    if col not in support.columns:
        return []
    if basis.startswith("modal_"):
        vals = support.loc[support[col] >= 0, col].to_numpy(dtype=int)
        return sorted(set(int(v) for v in vals))
    out: set = set()
    for s in support[col].fillna(""):
        out.update(int(v) for v in str(s).split("|") if v != "")
    return sorted(out)

NON_FEATURE_COLS = {"agent", "env", "num_episodes", "mean_final_score", "mean_episode_length"}

NEAR_CONSTANT_STD = 1e-12
REL_SPREAD_THRESH = 0.01
MEDIAN_TOL = 1e-6


def feature_stats(x: np.ndarray) -> Dict[str, float]:
    med = float(np.median(x))
    return {
        "std": float(x.std()),
        "rel_spread": float(x.std() / (abs(float(x.mean())) + 1e-8)),
        "n_unique": int(len(np.unique(x))),
        "frac_at_median": float(np.mean(np.abs(x - med) <= MEDIAN_TOL)),
    }


def degenerate_cell_columns(support_csv: Path) -> Dict[str, str]:
    """Cell-valued columns whose axis carries a single value population-wide -> {column: reason}."""
    support = pd.read_csv(support_csv)
    out: Dict[str, str] = {}
    for col, (basis, axis) in CELL_VALUED_COLUMNS.items():
        sup = axis_support(support, basis, axis)
        if len(sup) <= 1:
            out[col] = f"single {axis}-value {sup} across all teammates"
    return out


def unsupported_ratio_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Derived fractions whose numerator event is essentially unobserved -> {column: reason}."""
    out: Dict[str, str] = {}
    for col, num in RATIO_NUMERATOR.items():
        if col in df.columns and num in df.columns:
            rate = float(df[num].mean())
            if rate < MIN_EVENT_SUPPORT:
                out[col] = f"numerator {num} mean {rate:.4f}/episode < {MIN_EVENT_SUPPORT}"
    return out


def disconnected_layout_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Contention and proximity-conditioned columns when the two agents start in different components."""
    envs = set(df["env"].astype(str)) if "env" in df.columns else set()
    if len(envs) != 1 or not next(iter(envs)).startswith("overcooked-"):
        return {}
    from envs.overcooked.augmented_layouts import augmented_layouts
    layout = augmented_layouts[next(iter(envs))[len("overcooked-"):]]
    comp = np.asarray(layout["free_space_map"]).ravel()[np.asarray(layout["agent_idx"])]
    if len(set(comp.tolist())) == 1:
        return {}
    reason = f"agents start in different layout components {comp.tolist()}"
    return {c: reason for c in CONTENTION + PROXIMITY_CONDITIONED}


def is_degenerate(s: Dict[str, float]) -> bool:
    """Exactly constant across teammates."""
    return s["std"] <= NEAR_CONSTANT_STD


def is_suspect(s: Dict[str, float]) -> bool:
    """Diagnostic only; never drops."""
    return s["n_unique"] <= 2 or s["rel_spread"] < REL_SPREAD_THRESH


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", nargs="+", required=True,
                   help="path:task_label entries (bare path: label = parent dir name).")
    p.add_argument("--output", type=Path, required=True, help="markdown report path.")
    p.add_argument("--cell-support", nargs="*", default=[],
                   help="path:task_label entries pointing at cell_support.csv files.")
    p.add_argument("--drop-list-dir", type=Path, default=None,
                   help="Also write each drop list to <dir>/<task>.txt.")
    args = p.parse_args()

    tasks: Dict[str, pd.DataFrame] = {}
    for entry in args.features:
        if ":" in entry and not entry.startswith("/"):
            path_s, label = entry.rsplit(":", 1)
        elif entry.count(":") >= 1:
            path_s, label = entry.rsplit(":", 1)
        else:
            path_s, label = entry, Path(entry).parent.name
        path = Path(path_s)
        if not path.exists():
            raise FileNotFoundError(path)
        tasks[label] = pd.read_csv(path)

    # common numeric feature columns, in first-task order
    first = next(iter(tasks.values()))
    feature_cols = [
        c for c in first.columns
        if c not in NON_FEATURE_COLS and c != "set"
        and pd.api.types.is_numeric_dtype(first[c])
        and all(c in df.columns for df in tasks.values())
    ]

    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_task_drop: Dict[str, List[str]] = {label: [] for label in tasks}
    for feat in feature_cols:
        per_task = {
            label: feature_stats(df[feat].to_numpy(dtype=np.float64))
            for label, df in tasks.items()
        }
        stats[feat] = per_task
        for label, s in per_task.items():
            if is_degenerate(s):
                per_task_drop[label].append(feat)

    # second rule: cell-support degeneracy
    cell_drop: Dict[str, Dict[str, str]] = {}
    for entry in args.cell_support:
        path_s, label = entry.rsplit(":", 1)
        if label not in tasks:
            raise SystemExit(f"--cell-support label {label!r} is not one of {sorted(tasks)}")
        found = degenerate_cell_columns(Path(path_s))
        # only drop columns that are actually in this feature set
        cell_drop[label] = {c: r for c, r in found.items() if c in feature_cols}
        for col in cell_drop[label]:
            if col not in per_task_drop[label]:
                per_task_drop[label].append(col)

    # third rule: ratio support
    ratio_drop: Dict[str, Dict[str, str]] = {}
    for label, df in tasks.items():
        found = unsupported_ratio_columns(df)
        ratio_drop[label] = {c: r for c, r in found.items() if c in feature_cols}
        for col in ratio_drop[label]:
            if col not in per_task_drop[label]:
                per_task_drop[label].append(col)

    # fourth rule: split layouts
    split_drop: Dict[str, Dict[str, str]] = {}
    for label, df in tasks.items():
        found = disconnected_layout_columns(df)
        split_drop[label] = {c: r for c, r in found.items() if c in feature_cols}
        for col in split_drop[label]:
            if col not in per_task_drop[label]:
                per_task_drop[label].append(col)

    L: List[str] = []
    L.append("# Feature structural screen\n")
    L.append(f"- tasks: {', '.join(tasks)}  |  features screened: {len(feature_cols)}")
    L.append(f"- degeneracy rule (structural): std <= {NEAR_CONSTANT_STD:g}; "
             "dropped per task if degenerate on THAT task")
    L.append(f"- diagnostic only (~, never drops): n_unique <= 2 OR relative spread < {REL_SPREAD_THRESH}\n")
    L.append("## Per-feature per-task stats\n")
    L.append("std / rel_spread / n_unique / frac_within_1e-6_of_median; * = dropped (constant) on that task, "
             "~ = low-spread diagnostic flag\n")
    header = "| feature | " + " | ".join(tasks) + " | dropped everywhere |"
    L.append(header)
    L.append("|---" * (len(tasks) + 2) + "|")
    for feat in feature_cols:
        cells = []
        for label in tasks:
            s = stats[feat][label]
            mark = "*" if is_degenerate(s) else ("~" if is_suspect(s) else "")
            cells.append(f"{s['std']:.3g}/{s['rel_spread']:.3g}/{s['n_unique']}/"
                         f"{s['frac_at_median']:.2f}{mark}")
        drop = "**DROP**" if all(feat in lst for lst in per_task_drop.values()) else ""
        L.append(f"| `{feat}` | " + " | ".join(cells) + f" | {drop} |")
    L.append("")

    if any(ratio_drop.values()):
        L.append("## Ratio-support rule (derived fractions)\n")
        for label, found in ratio_drop.items():
            for col, reason in found.items():
                L.append(f"- {label}: `{col}` ({reason})")
        L.append("")

    if any(split_drop.values()):
        L.append("## Split-layout rule\n")
        for label, found in split_drop.items():
            if found:
                L.append(f"- {label}: {' '.join('`'+c+'`' for c in found)} "
                         f"({next(iter(found.values()))})")
        L.append("")

    if cell_drop:
        L.append("## Cell-support rule (coordinate features)\n")
        L.append("A coordinate column is dropped when every teammate's *usual* cell shares "
                 "one value on that axis, so the column cannot express any between-teammate "
                 "difference on this layout. Computed before the 128-episode averaging, "
                 "which is what makes it immune to the off-wall drops that defeat the "
                 "variance and reliability rules.\n")
        for label, found in cell_drop.items():
            L.append(f"### {label}\n")
            if found:
                for col, reason in found.items():
                    L.append(f"- `{col}` -- {reason}")
            else:
                L.append("(nothing dropped by this rule)")
            L.append("")

    emit = per_task_drop
    L.append("## Per-task drop lists\n")
    L.append("Each list is the set of features degenerate on THAT task; feed the "
             "matching list to `cluster_conventions.py --drop-features` when "
             "clustering that task alone.\n")
    for label, lst in per_task_drop.items():
        L.append(f"### {label} ({len(lst)} dropped)\n")
        L.append(" ".join(lst) if lst else "(none)")
        L.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(L))
    if args.drop_list_dir is not None:
        args.drop_list_dir.mkdir(parents=True, exist_ok=True)
        for label, lst in emit.items():
            (args.drop_list_dir / f"{label}.txt").write_text(" ".join(lst))
        print(f"wrote {len(emit)} drop list(s) to {args.drop_list_dir}")
    print(f"screened {len(feature_cols)} features over {len(tasks)} tasks")
    for label, lst in emit.items():
        print(f"  {label}: {len(lst)} dropped: {' '.join(lst) if lst else '(none)'}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
