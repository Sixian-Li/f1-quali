# Results and limitations

The CSVs in `benchmarks/` are small summaries of a fixed historical study. They do
not represent a newly untouched test set or evidence that m1r1 always beats simpler
models. m1r1 was chosen by the author after a seven-method comparison because its
rating design was preferred while its predictions were close to the alternatives;
it did not win every metric.

## Roles of the years

| Period | Role |
| --- | --- |
| 2010–2016 | Initial training history; 2010–2015 is a retrospective reconstruction |
| 2017–2022 | Rolling validation used to compare methods |
| 2023–2026 R14 | Already inspected; replayed with past-only annual fits |

Each year fits only earlier seasons (back to 2010), with a January 1 training
boundary and the first five events excluded from predictor optimization. In 2026,
the Top 3 head and correction rules train on 179 complete events from 2010–2025
(Q2/Q3: 180). This is legitimate past-only fitting, while the repeated inspection of
evaluation periods still limits claims of independent generalization. 2025 was
initially held back in earlier research and later inspected.

## Main results by year

| Year | Complete four-task events | Q2 | Q3 | Top 3 | Pole | Rank MAE | Probability score ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2011 | 6 | 95.10% | 86.67% | 61.11% | 83.33% | 1.5278 | 0.417412 |
| 2012 | 9 | 95.42% | 77.78% | 62.96% | 55.56% | 2.3796 | 0.483981 |
| 2013 | 11 | 92.05% | 83.64% | 63.64% | 36.36% | 2.4050 | 0.588927 |
| 2014 | 9 | 93.35% | 88.89% | 74.07% | 33.33% | 2.0340 | 0.480291 |
| 2015 | 7 | 91.43% | 81.43% | 76.19% | 71.43% | 2.2000 | 0.437087 |
| 2016 | 9 | 93.06% | 84.44% | 62.96% | 66.67% | 2.2626 | 0.474540 |
| 2017 | 10 | 93.33% | 85.45% | 73.33% | 30.00% | 1.8600 | 0.490124 |
| 2018 | 13 | 90.77% | 83.08% | 79.49% | 53.85% | 2.3538 | 0.564271 |
| 2019 | 9 | 93.33% | 86.67% | 74.07% | 33.33% | 1.9000 | 0.539318 |
| 2020 | 12 | 92.78% | 81.67% | 88.89% | 66.67% | 2.1000 | 0.402852 |
| 2021 | 13 | 91.28% | 81.54% | 79.49% | 46.15% | 2.3769 | 0.540857 |
| 2022 | 17 | 86.27% | 81.76% | 78.43% | 41.18% | 2.7588 | 0.584760 |
| 2023 | 13 | 85.64% | 73.85% | 58.97% | 61.54% | 3.6385 | 0.711196 |
| 2024 | 15 | 86.67% | 78.67% | 62.22% | 40.00% | 2.9400 | 0.668041 |
| 2025 | 13 | 81.03% | 76.15% | 74.36% | 30.77% | 3.2615 | 0.701102 |
| 2026 | 8 | 93.75% | 88.75% | 50.00% | 25.00% | 1.7841 | 0.519297 |

These are sixth-event-onward metrics; each pool uses its own complete-label
denominator, preserved in `m1r1_yearly.csv`. 2010 has no earlier training data and
uses uniform quota priors; its row is in the CSV but is not a model result. Early
years have few complete events.

Summed over validation (2017–2022), m1r1 finds 176 of 222 Top 3 slots with a Top 3
normalized Brier of 0.3740; over 2023–2026 R14 it finds 92 of 147 (0.6022).

## Comparison with the course-style baseline

The course-style OLS baseline predicts absolute qualifying seconds, using the
current season's earlier races and a cutoff-corrected reconstruction of its four
features. It is not a byte-for-byte rerun of the original course code on new years.
It outputs rankings, not calibrated four-pool probabilities; Brier comparisons
against OLS are therefore not supplied. The public baseline skips three events with
no usable practice time (2017 R11, 2020 R11 and 2023 R12); comparisons use the
remaining identical events.

| Period · events | Method | Q2 | Q3 | Top 3 slots | Pole | Rank MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2017–2022 · 72 | Rolling OLS | 88.13% | 76.30% | 150 / 216 | 36.11% | 2.888 |
| 2017–2022 · 72 | m1r1 | 90.87% | 83.15% | 171 / 216 | 47.22% | 2.292 |
| 2023–2026 · 48 | Rolling OLS | 83.57% | 77.08% | 83 / 144 | 27.08% | 3.283 |
| 2023–2026 · 48 | m1r1 | 86.18% | 78.13% | 90 / 144 | 39.58% | 3.020 |

m1r1 is better on every displayed metric in validation. The later advantage is
smaller and not uniform: in 2025 OLS has slightly better Q2, Q3 and rank MAE, and in
2026 OLS finds 14 of 24 Top 3 slots against m1r1's 12, and 3 of 8 poles against 2.
Block-bootstrap intervals from the research comparison place the Top 3 gain over OLS
clearly above zero in validation but touching zero afterwards; they are descriptive
and not corrected for method selection. The 70.59% reported by the author's earlier
STATS 507 course project (2022 Top 3 overlap, events 6–22) came from a different setup
with temporal-availability problems and must not be pasted into this comparison.

## Relation to v6 and to the training range

`v6_yearly.csv` keeps the first release's results. v6 had no fitted models in
2016–2017 because its training started in 2016. From 2018, the per-year Top 3 and
pole results of the two methods move in both directions, and probability scores are
mixed (for example 2024: Top 3 62.22% vs 55.56%, probability score 0.668 vs 0.657).

The seven-method research comparison also found that the six rating variants it
tested had the same number of Top 3 hits in every event and very close probabilities;
their ratings differed much more than their predictions. On matched events, each
variant trained on 2010+ had slightly worse Top 3 probability error than the same
design trained on 2016+, while its Top 3 hits changed by at most a few slots. The expansion changed
several things at once (early practice admission, the 136 early pace pairs and early
achievement admission), so it is not a clean test of the training window; a
single-factor comparison has not been run.

## Reading the metrics

Top 3 overlap is the number of selected drivers who actually finished in the top
three, divided by three and averaged across complete events. It does not require
their order to be correct. Whole-list exact rate is separately reported. A tight
pack can reverse order despite small underlying pace differences; the model expresses
four membership probabilities, not a full probabilistic lap-time distribution.

## Remaining weaknesses

- Development choices have been revisited repeatedly; new prospective evidence is needed.
- 2010–2015 is reconstructed after the fact: session clocks and original
  publication times are unknown and rosters are post-event participants.
- Some years contain few complete events, and incomplete labels reduce coverage.
- Summary practice times omit fuel, tyres, traffic and many other important conditions.
- The achievement component includes car/seat effects; the leaderboard is not a
  pure same-car skill ranking. Preferences about any particular driver order are
  not used as training truth.
- The learned corrections have no uncertainty estimate, and the network solver and
  final projection are not differentiated during training.
- Euclidean probability coordination can shrink already small pole probabilities
  excessively; probability coherence does not ensure good log loss or calibration.

The [m1r1 verification](../benchmarks/m1r1_verification.json) establishes numerical
equivalence with the research implementation in the checked environment. It does
not increase predictive validity or establish untested platform support.
