# Method

## Prediction boundary

A prediction is tied to an event, the complete entrant roster, a circuit layout,
and a timezone-aware cutoff no later than main qualifying. Each historical row
uses the information available before that row's event. A target result never
enters its own rating or feature construction. Practice drivers keep their own
identities, even when they do not enter qualifying.

The current method has two independently fitted parts. The predictor cannot change
the rating parameters, states or calibration. Updating a rating after an event
affects subsequent predictions, never a previously stored prediction.

## Independent driver ratings

The teammate component combines:

1. A robust Student-t loss on the gap in the **last common qualifying segment**.
2. A logistic loss on final qualifying order, with additional weight for a larger
   normalized final-position gap.
3. A driver lifetime effect, a stationary annual AR deviation, and a layout effect.

For example, two Q3 participants are compared in Q3. A Q1-eliminated driver and a
Q3 participant are compared in Q1. Missing or invalid laps do not imply elimination
and do not authorize silently substituting an earlier segment.

Evidence of age `a` years receives `floor + (1 - floor) * 2**(-a / half_life)`.
The floor preserves older teammate links. Parameters and calibration were selected
independently using past rating validation and are recorded per year in `configs/v6.json`.
For a target year, validation reads only 2019 through `min(target year − 1, 2022)`;
2016–2019 use preregistered defaults. `selection_max_year: 2022` is that cap, not a use
of later seasons.
This release rebuilds the selected states directly; users do not need all discarded
candidate runs. The exported year-specific choices must not be replaced by a single
latest-year configuration during historical reconstruction.

A separate achievement component shrinks full-field final qualifying performance
toward a neutral value and adds a bounded nonnegative bonus. It can include the
car/seat environment; it is not a pure same-car skill estimate. Sunday race results
and grid penalties are not qualifying-performance targets. Race/grid entries are
used only to cross-check identities in the early warmup data.

Displayed ability is `1 + 99 * normal_cdf(ability / display_scale)`; layouts use
`1 + 9 * normal_cdf(layout_effect / display_layout_scale)`. These displays are not
win percentages. The rating API also retains latent estimates, evidence counts
and conditional uncertainty. Composite uncertainty is not fabricated by adding an
independent bonus variance.

## Current-season inputs

The 18 model columns are:

```text
practice_pct, practice_missing, practice_laps_log,
fp1_pct, fp2_pct, fp3_pct, fp1_missing, fp2_missing, fp3_missing,
team_recent, driver_recent, driver_median, team_track,
driver_history_log, team_history_log, changed_constructor,
ability, track_effect
```

Only the two rating columns summarize previous seasons. All detailed context is
restricted to the target season before the cutoff, including available practice
for the current weekend. The latest eligible practice session and each FP session
retain separate inputs. Missing practice uses a neutral value plus missingness flags.

The v6 change concerns `driver_recent`, `team_recent` and `team_track`: use the last
24 valid current-season observations, weights with half-life 12 observations, and
the equal average of weighted mean and weighted median. The median is the first
value in stable value order whose cumulative weight reaches half the total.
The independent `driver_median` feature retains its original definition. Team/layout
effects retain the original shrinkage toward the team's current-season center.

## Four-pool prediction and loss

Features are centered within the complete pre-qualifying field and scaled on past
training events. Q2, Q3 and Top 3 use regularized binary logistic loss, with a quota
offset; pole uses a grouped softmax loss. Regularization is fixed at `0.3, 3, 0.3, 3`
for Q2, Q3, Top 3 and pole respectively. Events have equal weight, with equal driver
weight within a binary-task event. Pole contributes one categorical loss per event.

Each annual predictor uses earlier seasons with labels available before January 1
of the target year. The first five events of each training season do not enter the
fit. At least 12 complete training events are required per task. Otherwise the
task returns a quota-based uniform prior. Predictions still work during the first
five events, and their observed data may inform subsequent current-season inputs.

Raw probabilities are retained. The selected v6 uses Euclidean projection to
enforce nested probabilities and event-level capacities, then jointly assigns ranks
to maximize equal-task expected membership overlap. Fine ordering within a pool is
secondary. This coherence operation is not a calibration guarantee; tiny projected
tail probabilities are a known limitation.

## Evaluation

Each task is scored only on events with a complete matching roster and complete
labels for that task. Missing labels remain unknown. Q2/Q3 quotas use reviewed
season/format rules, not the number of drivers who happened to set a time.

For task quota `k` in a field of `n`, normalized Brier divides mean squared probability
error by `(k/n) * (1-k/n)`. The primary probability score averages the four normalized
Brier scores per complete event, then averages events. Membership overlap,
whole-list exact matches, log loss and full-field rank MAE are reported separately.
First-five and all-event summaries are separate from the main sixth-event-onward score.
