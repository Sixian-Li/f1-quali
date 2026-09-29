"""Pair m1r1 evaluations with the rolling baselines; summarize annual training traces.

Inputs are the documented workflow's outputs: ``runs/models/<year>``,
``runs/evaluations/<year>`` and the ``baseline`` command's directory.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from f1_quali.integrity import verify

TASKS = ("q2", "q3", "top3", "pole")
PERIODS = {"validation_2017_2022": (2017, 2022), "seen_2023_2026": (2023, 2026)}


def comparison(evaluations, baselines):
    method = pd.concat(
        [pd.read_parquet(p) for p in sorted(evaluations.glob("*/event_metrics.parquet"))],
        ignore_index=True,
    )
    method = method[method.season_event_ordinal.ge(6)]
    rows = []
    for name, base in baselines.groupby("model", sort=True):
        joined = method.merge(base, on="event_id", suffixes=("", "_baseline"), validate="1:1")
        groups = [(str(s), g) for s, g in joined.groupby("season")]
        groups += [(p, joined[joined.season.between(*y)]) for p, y in PERIODS.items()]
        for period, group in groups:
            for task in TASKS:
                part = group[group[f"complete_{task}"] & group[f"complete_{task}_baseline"]]
                quota = part[f"quota_{task}"]
                rows.append(
                    {
                        "period": period,
                        "metric": f"{task}_overlap",
                        "baseline": name,
                        "events": len(part),
                        "slots": int(quota.sum()),
                        "baseline_hits": int(
                            (part[f"{task}_overlap_baseline"] * quota).round().sum()
                        ),
                        "m1r1_hits": int((part[f"{task}_overlap"] * quota).round().sum()),
                        "baseline_value": part[f"{task}_overlap_baseline"].mean(),
                        "m1r1_value": part[f"{task}_overlap"].mean(),
                    }
                )
            part = group[group.complete_rank & group.complete_rank_baseline]
            rows.append(
                {
                    "period": period,
                    "metric": "rank_mae",
                    "baseline": name,
                    "events": len(part),
                    "baseline_value": part.rank_mae_baseline.mean(),
                    "m1r1_value": part.rank_mae.mean(),
                }
            )
    return pd.DataFrame(rows)


def training(models):
    rows = []
    for path in sorted(models.glob("*/model.json")):
        verify(path.parent, "predictor")
        model = json.loads(path.read_text())
        trace = model["trace"]
        years = [t["trace"]["training_max_year"] for t in model["tasks"].values()]
        row = {
            "season": trace["year"],
            "training_years": (
                f"2010–{max(y for y in years if y is not None)}" if any(years) else ""
            ),
            "training_cutoff": trace["training_cutoff"],
            "four_task_training_events": trace["training_events"],
        }
        for task, head in model["tasks"].items():
            row[f"{task}_training_events"] = head["trace"]["training_events"]
            row[f"{task}_kind"] = head["kind"]
        rule = model["joint_rule"]
        names = ["driver_pace_fast", "driver_pace_slow", "driver_result", "beta_delta_fixed"]
        names += ["layout_pace", "layout_result"]
        row.update({f"rule_{n}": v for n, v in zip(names, rule["parameters"], strict=True)})
        row["rule_fallback"] = bool(rule.get("fallback", False))
        row["auxiliary_teammate_pairs"] = rule.get("auxiliary_pairs", 0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("season")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.baselines, "baselines")
    baselines = pd.read_parquet(args.baselines / "event_metrics.parquet")
    args.output.mkdir(parents=True, exist_ok=True)
    comparison(args.evaluations, baselines).to_csv(
        args.output / "m1r1_baseline_comparison.csv", index=False
    )
    training(args.models).to_csv(args.output / "m1r1_training_years.csv", index=False)
    print(args.output)


if __name__ == "__main__":
    main()
