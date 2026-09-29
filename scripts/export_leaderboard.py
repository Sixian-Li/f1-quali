"""Turn `f1-quali ratings` output into a dated public leaderboard snapshot.

Example (after the documented m1r1 workflow):

    python scripts/export_leaderboard.py --ratings runs/ratings \
        --data data/full_era --cache data/raw --output ratings/2026-round-14-m1r1.json

Driver names come from the dataset; team and event display names from the pinned
F1DB release in the cache. The JSON keeps full precision; the printed Markdown
table uses two decimals.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from f1_quali.data.build import resource
from f1_quali.data.full_era import load_full_era
from f1_quali.integrity import sha256, verify
from f1_quali.sources.f1db import F1DB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.ratings, "ratings")
    snapshot = json.loads((args.ratings / "ratings.json").read_text())
    table = pd.read_parquet(args.ratings / "ratings.parquet")
    data, _ = load_full_era(args.data)
    source = resource("sources")["f1db"]
    db = F1DB(args.cache, source, offline=True)
    teams = db.table("constructors").set_index("id")["name"]
    event = data.events[data.events.event_id.eq(snapshot["last_result_event_id"])].iloc[0]
    event_name = db.table("grands-prix").set_index("id").fullName[event.grand_prix_id]
    names = data.drivers.set_index("driver_id").driver_name
    labels = data.qualifying[data.qualifying.event_id.eq(event.event_id)]
    drivers = [
        {
            "rank": int(row.rank),
            "driver_id": row.driver_id,
            "driver": names[row.driver_id],
            "constructor_id": row.constructor_id,
            "team": teams[row.constructor_id],
            "composite_rating": float(row.ability_score),
            "teammate_ability": float(row.joint_pure_score),
            "achievement_bonus_internal": float(row.achievement_bonus),
            "rule_correction_internal": float(row.joint_driver_delta),
            "teammate_comparisons": int(row.teammate_comparisons),
        }
        for row in table.itertuples()
    ]
    manifest = json.loads((args.ratings / "manifest.json").read_text())
    output = {
        "schema": "f1-quali.rating-leaderboard.v1",
        "method": "m1r1",
        "research_lock": "m1r1_full_era_de17567786cf",
        "season": int(event.season),
        "after_round": int(event["round"]),
        "after_event_id": int(event.event_id),
        "event_name": event_name,
        "circuit_layout_id": event.circuit_layout_id,
        "roster_basis": "entrants_in_the_latest_included_qualifying_event",
        "roster_event_id": int(event.event_id),
        "driver_count": len(drivers),
        "cutoff_exclusive": snapshot["cutoff"],
        "latest_qualifying_start": pd.Timestamp(event.prediction_cutoff).isoformat(),
        "max_label_available_at": labels.label_available_at.max().isoformat(),
        "published_at": datetime.now(UTC).isoformat(),
        "sort": {"key": "composite_rating", "direction": "descending", "tie_breaker": "driver_id"},
        "display_decimals": 2,
        "rating_definition": (
            "teammate ability plus fixed 0.6 x individually admitted final-qualifying "
            "achievement (one-year memory, no floor), with Top3-trained shared driver "
            "corrections; includes car/seat effects"
        ),
        "teammate_ability_definition": "teammate network ability with the shared correction",
        "circuit_effect_included_in_leaderboard": False,
        "rule_training_max_year": snapshot["rule_training_max_year"],
        "sources": {
            "f1db": {"version": source["version"], "sha256": source["sha256"]},
            "attribution": (
                "F1DB and contributors, CC BY 4.0; source facts normalized and reviewed "
                "by this project"
            ),
            "full_era_dataset_manifest_sha256": sha256((args.data / "manifest.json").read_bytes()),
            "ratings_output_manifest_sha256": sha256(json.dumps(manifest, sort_keys=True).encode()),
            "rating_bundle_sha256": snapshot["rating_bundle_sha256"],
            "history_start_year": 2010,
            "results_through_round": int(event["round"]),
            "original_raw_cache_distributed": False,
        },
        "drivers": drivers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print("| Rank | Driver | Team | **Rating / 100** | Teammate ability / 100 | Comparisons |")
    print("| ---: | --- | --- | ---: | ---: | ---: |")
    for d in drivers:
        print(
            f"| {d['rank']} | {d['driver']} | {d['team']} | **{d['composite_rating']:.2f}** | "
            f"{d['teammate_ability']:.2f} | {d['teammate_comparisons']} |"
        )


if __name__ == "__main__":
    main()
