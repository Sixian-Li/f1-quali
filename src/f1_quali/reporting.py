"""Annual summaries preserving task-specific complete-event denominators."""

import pandas as pd

from f1_quali.evaluation import TASKS

METRICS = [
    "score",
    "rank_mae",
    *[
        f"{task}_{metric}"
        for task in TASKS
        for metric in ("brier", "normalized_brier", "logloss", "overlap", "exact")
    ],
]


def annual(events):
    rows = []
    for year, group in events.groupby("season", sort=True):
        row = {
            "season": int(year),
            "target_events": len(group),
            "target_entrants": int(group.entrants.sum()),
            "complete_four_events": int(group.complete_four_tasks.sum()),
            "complete_four_entrants": int(group.loc[group.complete_four_tasks, "entrants"].sum()),
            "rank_events": int(group.complete_rank.sum()),
        }
        for task in TASKS:
            valid = group[group[f"complete_{task}"]]
            row[f"{task}_events"] = len(valid)
            row[f"{task}_entrants"] = int(valid.entrants.sum())
            row[f"{task}_hits"] = round((valid[f"{task}_overlap"] * valid[f"quota_{task}"]).sum())
            row[f"{task}_slots"] = int(valid[f"quota_{task}"].sum())
            row[f"{task}_exact_events"] = int(valid[f"{task}_exact"].sum())
        rows.append({**row, **group[METRICS].mean().to_dict()})
    return pd.DataFrame(rows)
