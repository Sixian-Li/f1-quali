"""Public CLI: data, full-era extension, preparation, annual fit, evaluation, forecast.

The default method is RM (2010--2026). m1r1 and v6 stay
available for historical reproduction with ``--method`` on ``prepare`` and
``demo``; later commands follow the method recorded in their inputs.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from f1_quali import m1r1, rm
from f1_quali.config import load_config
from f1_quali.demo import run_demo
from f1_quali.integrity import verify
from f1_quali.pipeline import forecast, predict_prepared, prepare, train

METHODS = ("rm", "m1r1", "v6")


def _method(directory):
    return verify(directory)["metadata"].get("method", "")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="F1 qualifying ratings and prediction — RM (m1r1 and v6 optional)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a deterministic synthetic example offline")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--method", choices=METHODS, default="rm")
    data = sub.add_parser("data", help="Fetch pinned sources and construct canonical tables")
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--cache", type=Path, required=True)
    data.add_argument("--offline", action="store_true")
    era = sub.add_parser("full-era", help="Add 2010-2015 and career experience for RM/m1r1")
    era.add_argument("--data", type=Path, required=True, help="output of `f1-quali data`")
    era.add_argument("--cache", type=Path, required=True)
    era.add_argument("--output", type=Path, required=True)
    era.add_argument("--offline", action="store_true")
    prep = sub.add_parser("prepare", help="Build prior-event ratings, features and labels")
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--method", choices=METHODS, default="rm")
    prep.add_argument("--config", type=Path)
    fit = sub.add_parser("train", help="Fit one annual model using only earlier seasons")
    fit.add_argument("--prepared", type=Path, required=True)
    fit.add_argument("--year", type=int, required=True)
    fit.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate", help="Replay one model year on complete event groups")
    evaluate.add_argument("--prepared", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--event-id", type=int)
    evaluate.add_argument("--output", type=Path, required=True)
    predict = sub.add_parser("predict", help="Predict one event at an explicit cutoff")
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--prepared", type=Path, help="prepared history (required for RM/m1r1)")
    predict.add_argument("--event-id", type=int, required=True)
    predict.add_argument("--cutoff", required=True)
    predict.add_argument("--output", type=Path, required=True)
    ratings = sub.add_parser("ratings", help="Export RM/m1r1 driver and circuit ratings at a cutoff")
    ratings.add_argument("--prepared", type=Path, required=True)
    ratings.add_argument("--model", type=Path, required=True)
    ratings.add_argument("--cutoff", required=True)
    ratings.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify", help="Check an artifact's file hashes")
    check.add_argument("directory", type=Path)
    baseline = sub.add_parser("baseline", help="Run same-season rolling OLS and practice baselines")
    baseline.add_argument("--data", type=Path, required=True)
    baseline.add_argument("--years", type=int, nargs="+", required=True)
    baseline.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    def progress(message):
        print(message, flush=True)

    if args.command == "demo":
        result = run_demo(args.output, progress=progress, method=args.method)
    elif args.command == "data":
        from f1_quali.data.build import build_dataset

        result = build_dataset(args.output, args.cache, offline=args.offline)
    elif args.command == "full-era":
        from f1_quali.data.full_era import build_full_era

        result = build_full_era(args.data, args.cache, args.output, offline=args.offline)
    elif args.command == "prepare":
        if args.method in m1r1.JOINT_METHODS:
            engine = rm if args.method == "rm" else m1r1
            config = (rm.load_rm_config(args.config) if args.method == "rm"
                      else m1r1.load_m1r1_config(args.config))
            result = engine.prepare(args.data, args.output, config, progress=progress)
        else:
            result = prepare(args.data, args.output, load_config(args.config), progress=progress)
    elif args.command == "train":
        if _method(args.prepared) in m1r1.JOINT_METHODS:
            result = m1r1.train(args.prepared, args.year, args.output)
        else:
            result = train(args.prepared, args.year, args.output)
    elif args.command == "evaluate":
        if _method(args.model) in m1r1.JOINT_METHODS:
            result = m1r1.evaluate(args.prepared, args.model, args.output, event_id=args.event_id)
        else:
            result = predict_prepared(
                args.prepared, args.model, args.output, event_id=args.event_id
            )
        metrics = pd.read_parquet(args.output / "event_metrics.parquet")
        print(
            metrics[
                ["season", "round", "complete_four_tasks", "top3_overlap", "pole_overlap"]
            ].to_string(index=False)
        )
    elif args.command == "predict":
        if _method(args.model) in m1r1.JOINT_METHODS:
            if args.prepared is None:
                parser.error("RM/m1r1 prediction requires --prepared")
            result = m1r1.forecast(
                args.data, args.prepared, args.model, args.event_id, args.cutoff, args.output
            )
        else:
            result = forecast(args.data, args.model, args.event_id, args.cutoff, args.output)
    elif args.command == "ratings":
        result = m1r1.export_ratings(args.prepared, args.model, args.cutoff, args.output)
    elif args.command == "baseline":
        from f1_quali.baselines.rolling import run_baselines
        from f1_quali.data.portable import load_dataset

        dataset, _ = load_dataset(args.data)
        result = run_baselines(dataset, args.years, args.output)
    else:
        result = verify(args.directory)
    if "sha256" in result:
        result = {
            "kind": result["kind"],
            "verified_files": len(result["sha256"]),
            **result["metadata"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
