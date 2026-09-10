"""Screen and cluster every extracted population under one feature root.

    python scripts/convention_analysis/run_pipeline.py --pd-root results/conv_v12 --out results/conv_v12/eval
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from scripts.convention_analysis import feature_groups

ROOT = Path(__file__).resolve().parents[2]

HERE = "scripts/convention_analysis"


def discover(pd_root: Path) -> List[Tuple[str, str]]:
    """(name, variant) for every pd_<name>/overcooked-<variant>/features.csv."""
    # keep one variant per population (shortest directory name)
    best: Dict[str, str] = {}
    for f in sorted(pd_root.glob("pd_*/overcooked-*/features.csv")):
        name, variant = f.parents[1].name[len("pd_"):], f.parent.name[len("overcooked-"):]
        if name not in best or len(variant) < len(best[name]):
            best[name] = variant
    out = sorted(best.items())
    if not out:
        raise SystemExit(f"no pd_*/overcooked-*/features.csv under {pd_root}")
    return out


def sh(cmd: List[str]) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT), "JAX_PLATFORMS": "cpu"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r.stdout + r.stderr


def parse_summary(md: Path) -> Dict:
    out: Dict = {}
    for line in md.read_text().splitlines():
        if line.startswith("- selected **k = "):
            out["k"] = int(line.split("k = ")[1].split("**")[0])
        if "eta^2): **" in line:
            out["eta_sq"] = float(line.split("**")[1])
        if "**<- selected**" in line:
            out["silhouette"] = float(line.split("|")[2].split("**")[0])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pd-root", type=Path, required=True,
                   help="Directory holding pd_<name>/overcooked-<variant>/features.csv.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--modes", nargs="+", default=["split"], choices=["none", "split"])
    p.add_argument("--k-tolerance", type=float, default=0.05)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--exclude-groups", nargs="*", default=["throughput"])
    p.add_argument("--extra-drop", nargs="*", default=[], help="Dropped in every run.")
    p.add_argument("--only", nargs="*", default=None, help="Restrict to these population names.")
    args = p.parse_args()

    pops = discover(args.pd_root)
    if args.only:
        pops = [(n, v) for n, v in pops if n in args.only]
    out = args.out
    (out / "drop_lists").mkdir(parents=True, exist_ok=True)

    drops: Dict[str, List[str]] = {}
    derived: Dict[str, Path] = {}
    for name, variant in pops:
        raw = args.pd_root / f"pd_{name}" / f"overcooked-{variant}" / "features.csv"
        derived[name] = raw.with_name("features_derived.csv")
        feature_groups.derive(pd.read_csv(raw)).to_csv(derived[name], index=False)

        cs = derived[name].parent / "cell_support.csv"
        cmd = [sys.executable, f"{HERE}/screen_features.py",
               "--features", f"{derived[name]}:{name}",
               "--output", str(out / "drop_lists" / f"screen_{name}.md"),
               "--drop-list-dir", str(out / "drop_lists")]
        if cs.exists():
            cmd += ["--cell-support", f"{cs}:{name}"]
        else:
            print(f"warning: no cell support for {name}; cell-support rule skipped")
        sh(cmd)
        drops[name] = (out / "drop_lists" / f"{name}.txt").read_text().split()

    rows = []
    for name, _ in pops:
        for mode in args.modes:
            od = out / f"{name}__{mode}"
            cmd = [sys.executable, f"{HERE}/cluster_conventions.py",
                   "--features", str(derived[name]), "--set-labels", name, "--output-dir", str(od),
                   "--k-tolerance", str(args.k_tolerance), "--min-cluster-size", str(args.min_cluster_size),
                   "--groups", mode, "--drop-features", *drops[name], *args.extra_drop]
            if args.exclude_groups:
                cmd += ["--exclude-groups", *args.exclude_groups]
            log = sh(cmd)
            row = {"population": name, "mode": mode, **parse_summary(od / "cluster_summary.md")}
            row["n_features"] = int([l for l in log.splitlines() if l.startswith("k=")][0].split("features=")[1])
            sizes = pd.read_csv(od / "clusters.csv")["cluster"].value_counts().sort_values(ascending=False)
            row["sizes"] = "/".join(str(x) for x in sizes.tolist())
            rows.append(row)
            print(f"{name:26s} {mode:6s} k={row.get('k')} sil={row.get('silhouette')} "
                  f"eta2={row.get('eta_sq')} features={row['n_features']} sizes={row['sizes']}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "summary.csv", index=False)
    L = ["# Convention clustering\n", f"feature root: `{args.pd_root}`\n",
         "## Summary\n", res.to_markdown(index=False), "", "## Structural drop lists\n"]
    for name, lst in drops.items():
        L.append(f"- **{name}** ({len(lst)}): {' '.join(f'`{x}`' for x in lst) or '(none)'}")
    (out / "report.md").write_text("\n".join(L) + "\n")
    json.dump({"drop_lists": drops, "exclude_groups": args.exclude_groups, "extra_drop": args.extra_drop,
               "modes": args.modes, "k_tolerance": args.k_tolerance,
               "min_cluster_size": args.min_cluster_size}, open(out / "meta.json", "w"), indent=1)
    print(f"\nwrote {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
