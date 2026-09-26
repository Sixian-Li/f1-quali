"""Small public CLI: demo, data preparation, annual fit, evaluation and forecast."""

import argparse
import json
from pathlib import Path

import pandas as pd

from f1_quali.config import load_config
from f1_quali.demo import run_demo
from f1_quali.integrity import verify
from f1_quali.pipeline import forecast, predict_prepared, prepare, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="F1 qualifying prediction — frozen v6 method")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a deterministic synthetic example offline")
    demo.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare", help="Build prior-event ratings, features and labels")
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--config", type=Path)
    fit = sub.add_parser("train", help="Fit fixed four-pool models using earlier seasons")
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
    predict.add_argument("--event-id", type=int, required=True)
    predict.add_argument("--cutoff", required=True)
    predict.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify", help="Check an artifact's file hashes")
    check.add_argument("directory", type=Path)
    data = sub.add_parser("data", help="Fetch pinned sources and construct canonical tables")
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--cache", type=Path, required=True)
    data.add_argument("--offline", action="store_true")
    baseline = sub.add_parser("baseline", help="Run same-season rolling OLS and practice baselines")
    baseline.add_argument("--data", type=Path, required=True)
    baseline.add_argument("--years", type=int, nargs="+", required=True)
    baseline.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "demo":
        result = run_demo(args.output, progress=lambda message: print(message, flush=True))
    elif args.command == "prepare":
        result = prepare(
            args.data,
            args.output,
            load_config(args.config),
            progress=lambda message: print(message, flush=True),
        )
    elif args.command == "train":
        result = train(args.prepared, args.year, args.output)
    elif args.command == "evaluate":
        result = predict_prepared(args.prepared, args.model, args.output, event_id=args.event_id)
        metrics = pd.read_parquet(args.output / "event_metrics.parquet")
        print(
            metrics[
                ["season", "round", "complete_four_tasks", "top3_overlap", "pole_overlap"]
            ].to_string(index=False)
        )
    elif args.command == "predict":
        result = forecast(args.data, args.model, args.event_id, args.cutoff, args.output)
    elif args.command == "data":
        from f1_quali.data.build import build_dataset

        result = build_dataset(args.output, args.cache, offline=args.offline)
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
