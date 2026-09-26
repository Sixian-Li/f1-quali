# F1 Quali

Predict **Q2, Q3, Top 3 and pole** from independent driver ratings and information
available before qualifying. The model returns four probabilities per driver and
one consistent set of nested selections with a full-field ordering.

The selected method is **v6** (`memory_h12_mean_median_50_50`), the sixth iteration of
the author's research model; later experimental variants did not replace it. Ratings update before
each event. Detailed prediction inputs use the current season; predictors train
annually on earlier seasons. Software version `0.1.0` identifies this standalone
implementation, not a new model selection.

## Quick start

Python 3.13 is required. GitHub Actions runs the offline tests and demo on Linux and
macOS; the historical numerical verification was performed locally on macOS arm64.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
f1-quali demo --output demo-run
```

After installation the demo is entirely offline. It generates fictional data,
builds eventwise ratings, fits an annual model, and produces **120 predictions
across six events**. Compare `demo-run/summary.json` with the
[expected output](https://github.com/Sixian-Li/f1-quali/blob/main/examples/expected_demo_summary.json). Synthetic scores are not
evidence of real-world accuracy.

## Real data and evaluation

**Known acquisition limit:** a fresh-cache attempt on 2026-09-26 stopped because
the first required Formula 1 article no longer matched its recorded byte checksum.
Exact historical reconstruction currently requires the verified original cache;
the download recipe alone cannot recreate every pinned snapshot. The offline demo
and cached-data replay were verified. See the [source check](https://github.com/Sixian-Li/f1-quali/blob/main/benchmarks/source_acquisition.json).
The verified raw cache is not distributed, so outside the author's environment only
the demo and tests are reproducible end to end; the commands below document the workflow.

```sh
f1-quali data --cache data/raw --output data/canonical
f1-quali prepare --data data/canonical --output runs/prepared
f1-quali train --prepared runs/prepared --year 2026 --output runs/models/2026
f1-quali evaluate --prepared runs/prepared --model runs/models/2026 --output runs/evaluations/2026
f1-quali baseline --data data/canonical --years 2024 2026 --output runs/baselines
```

The source recipe is fixed at **2026-09-15**, through **2026 R14**, using F1DB
`v2026.14.0`, versioned schedules and explicit source reviews. It includes the
reviewed 2010–2015 rating warmup. Downloaded files are checked against registered
hashes. `--offline` uses an existing verified raw cache. Old source pages may change
or become unavailable; the command reports the failure instead of replacing them
silently. No raw source archive or pretrained model is redistributed. A new source
snapshot would need explicit review and a new data identity before claiming exact
historical reproduction.

[Data and reproduction](https://github.com/Sixian-Li/f1-quali/blob/main/docs/data_and_reproduction.md) describes the complete
workflow, cutoff contract and how to supply a new event. This is a research package,
not an automatic live feed.

## What the results show

The table below compares identical complete events from the sixth round onward.
Pool numbers measure **driver membership overlap**, not exact finishing order.

| Year / complete events | Method | Q2 | Q3 | Top 3 | Pole | Rank MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2024 / 15 | Rolling course-style OLS | 81.33% | 74.00% | 51.11% | 26.67% | 3.547 |
| 2024 / 15 | v6 | 87.11% | 79.33% | 55.56% | 46.67% | 2.907 |
| 2026 / 8, through R14 | Rolling course-style OLS | 91.41% | 88.75% | 58.33% | 37.50% | 2.011 |
| 2026 / 8, through R14 | v6 | 94.53% | 87.50% | 50.00% | 25.00% | 1.830 |

v6 improves full-field rank error in these comparisons, but does not consistently
improve Top 3 or pole. Both methods identify 37 of 69 Top 3 slots across these two
years. These periods have already been inspected; they are not new untouched tests.

The [yearly scorecard](https://github.com/Sixian-Li/f1-quali/blob/main/benchmarks/v6_yearly.csv) includes every year, task-specific
coverage, Brier scores, log losses, first-five results and raw probabilities.
[Training years](https://github.com/Sixian-Li/f1-quali/blob/main/benchmarks/training_years.csv) and
[results and limitations](https://github.com/Sixian-Li/f1-quali/blob/main/docs/results_and_limitations.md) explain interpretation.

## Method and verification

- Independently fitted teammate ratings compare the last common qualifying segment,
  combine pace and final qualifying order, and retain a lifetime/annual structure.
- Old rating evidence receives a positive weight floor plus exponential decay.
  A bounded final-qualifying achievement component is displayed separately.
- The predictor uses 18 inputs. Three current-season summaries combine a weighted
  mean and median, with a 12-observation half-life and at most 24 observations.
- Q2/Q3/Top 3 use regularized logistic models; pole uses event-level softmax.
  The original v6 probability coordination and nested-list decoder are preserved.

The standalone refactor reproduced **228 eventwise rating snapshots, 4,629 feature
rows, all 11 annual fits and all historical predictions** against the author's
unpublished research implementation of v6 (research lock ID `763e7f1871a6a6d3`).
Maximum input and probability differences were zero in the tested environment.
See the [verification record](https://github.com/Sixian-Li/f1-quali/blob/main/benchmarks/refactor_verification.json).

```sh
python -m pytest -W error
ruff check src tests scripts
python -m build --no-isolation
```

Tests run without external network access. Full historical replay is a separate
release check, so ordinary CI does not download all Formula 1 data.

Read the [method](https://github.com/Sixian-Li/f1-quali/blob/main/docs/method.md), [data guide](https://github.com/Sixian-Li/f1-quali/blob/main/docs/data_and_reproduction.md),
[limitations](https://github.com/Sixian-Li/f1-quali/blob/main/docs/results_and_limitations.md), and
[research journey / 研究回顾 (in Chinese)](https://github.com/Sixian-Li/f1-quali/blob/main/docs/research_journey.md).

## License and sources

Code and documentation: [MIT](https://github.com/Sixian-Li/f1-quali/blob/main/LICENSE), copyright 2026 **Sixian Li**.
Underlying datasets and third-party materials retain their own terms; see
[third-party notices](https://github.com/Sixian-Li/f1-quali/blob/main/THIRD_PARTY_NOTICES.md). Data sources include
[F1DB](https://github.com/f1db/f1db),
[f1schedule](https://github.com/theOehrly/f1schedule), and official FIA / Formula 1
documents linked in the source manifests. No official logos are included.
