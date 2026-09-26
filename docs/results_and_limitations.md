# Results and limitations

The CSVs in `benchmarks/` are small summaries of a fixed historical study. They do
not represent a newly untouched test set or evidence that v6 always beats simpler
models. The refactor changes packaging and dependencies, not the chosen method.

## Roles of the years

| Period | Role |
| --- | --- |
| 2010–2015 | Independent rating warmup only |
| 2016–2017 | Predictor-history shortage; uniform quota fallback |
| 2018 | First year with fitted four-pool models |
| 2019–2022 | Development and method selection |
| 2023–2024 | Already inspected checks |
| 2025 | Initially held back, later inspected; now a fixed replay |
| 2026 | Already inspected fixed replay; stops at R14 |

Each year fits only earlier seasons, with a January 1 training boundary and the
first five events excluded from predictor optimization. In 2026, all-four-task
training contains 124 complete events / 2,498 entrant rows from 2016–2025; Q2/Q3 each
have 125 events / 2,518 rows. This is legitimate past-only fitting, while the repeated
inspection of evaluation periods still limits claims of independent generalization.

## Main results by year

| Year | Complete four-task events | Q2 | Q3 | Top 3 | Pole | Rank MAE | Probability score ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2016 | 9 | 65.28% | 21.11% | 0.00% | 0.00% | 9.5253 | 1.000000 |
| 2017 | 10 | 69.70% | 49.09% | 3.33% | 0.00% | 6.9700 | 1.000000 |
| 2018 | 13 | 90.26% | 80.77% | 76.92% | 38.46% | 2.4692 | 0.565668 |
| 2019 | 9 | 92.59% | 85.56% | 74.07% | 33.33% | 1.9444 | 0.523019 |
| 2020 | 12 | 92.78% | 81.67% | 91.67% | 58.33% | 2.0667 | 0.445314 |
| 2021 | 13 | 91.28% | 82.31% | 79.49% | 38.46% | 2.3308 | 0.530393 |
| 2022 | 17 | 86.27% | 81.76% | 78.43% | 52.94% | 2.7176 | 0.582816 |
| 2023 | 13 | 85.13% | 72.31% | 61.54% | 61.54% | 3.6385 | 0.707142 |
| 2024 | 15 | 87.11% | 79.33% | 55.56% | 46.67% | 2.9067 | 0.657107 |
| 2025 | 13 | 81.54% | 76.15% | 74.36% | 15.38% | 3.2923 | 0.698180 |
| 2026 | 8 | 94.53% | 87.50% | 50.00% | 25.00% | 1.8295 | 0.539548 |

These are sixth-event-onward metrics. Each pool uses its own complete-label
denominator: 2017 Q2/Q3 each have 11 events, versus 10 for Top 3/pole. The CSV
preserves these counts. Fallback years' hard ranks are deterministic tie-breaking
of equal probabilities, not a learned ranking.

## Reading the metrics

Top 3 overlap is the number of selected drivers who actually finished in the top
three, divided by three and averaged across complete events. It does not require
their order to be correct. Whole-list exact rate is separately reported. A tight
pack can reverse order despite small underlying pace differences; v6 expresses
four membership probabilities, not a full probabilistic lap-time distribution.

The course-style OLS baseline predicts absolute qualifying seconds, using the
current season's earlier races and a cutoff-corrected reconstruction of its four
features. It is not a byte-for-byte rerun of the original course code on new years.
It outputs rankings, not calibrated four-pool probabilities; Brier comparisons
against OLS are therefore not supplied.

In 2024, v6 improves all displayed membership metrics and rank MAE against this
baseline. In 2026, OLS is better on Q3, Top 3 and pole, while v6 is better on Q2 and
rank MAE. Each year's Top 3 difference is only two slots. The 70.59% reported by the
author's earlier STATS 507 course project (2022 Top 3 overlap, events 6–22) came from a
different setup with temporal-availability problems
and must not be pasted into the strict-cutoff comparison.

## Remaining weaknesses

- Development choices have been revisited repeatedly; new prospective evidence is needed.
- Some years contain few complete events, and incomplete labels reduce coverage.
- Summary practice times omit fuel, tyres, traffic and many other important conditions.
- The bounded achievement component includes car/seat effects.
- Euclidean probability coordination can shrink already small pole probabilities
  excessively; probability coherence does not ensure good log loss or calibration.
- More complex pace distributions, extra history and detailed-practice variants did
  not establish a stable upgrade of the primary hard selections. They remain research
  findings rather than additional production features.

The [refactor verification](../benchmarks/refactor_verification.json) establishes
equivalence in the checked environment. It does not increase the original model's
predictive validity or establish untested platform support.
