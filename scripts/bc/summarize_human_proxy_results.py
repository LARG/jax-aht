import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


GROUP_KEYS = (
    "lbf_config",
    "method",
    "lambda",
    "human_reg_schedule",
    "total_timesteps",
)

METRICS = (
    "mean_return",
    "final_mean_return",
    "median_return",
    "human_cross_entropy",
    "human_action_accuracy",
)


def as_float(row: dict, key: str):
    value = row.get(key)
    if value in (None, "", "nan"):
        if key == "human_cross_entropy":
            value = row.get("bc_nll")
        elif key == "human_action_accuracy":
            value = row.get("bc_accuracy")
    if value in (None, "", "nan"):
        return None
    return float(value)


def metric_stats(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {"n": 0, "mean": "", "std": "", "min": "", "max": ""}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                row["_source_csv"] = str(path)
                rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in GROUP_KEYS)
        groups.setdefault(key, []).append(row)

    output = []
    for key, group in sorted(groups.items()):
        item = {k: v for k, v in zip(GROUP_KEYS, key)}
        item["num_rows"] = len(group)
        item["seeds"] = ";".join(row.get("seed", "") for row in group)
        item["source_csvs"] = ";".join(sorted({row["_source_csv"] for row in group}))
        for metric in METRICS:
            values = [v for row in group if (v := as_float(row, metric)) is not None]
            stats = metric_stats(values)
            item[f"{metric}_n"] = stats["n"]
            item[f"{metric}_mean"] = stats["mean"]
            item[f"{metric}_std"] = stats["std"]
            item[f"{metric}_min"] = stats["min"]
            item[f"{metric}_max"] = stats["max"]
        output.append(item)
    return output


def main(args):
    rows = summarize(read_rows([Path(p) for p in args.input_csv]))
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    main(parser.parse_args())
