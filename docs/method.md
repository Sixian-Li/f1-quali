# Method

The default in 0.3.0 is **[RM](rm.md)**. This page documents the shared pipeline
and the preserved **m1r1** baseline; RM changes the annual variance prior and
achievement mean memory as specified in its method note. Section [Previous method: v6](#previous-method-v6)
lists what changed relative to the first release; `--method v6` still runs it.

## Prediction boundary

A prediction is tied to an event, the complete entrant roster, a circuit layout,
and a timezone-aware cutoff no later than main qualifying. Each historical row
uses the information available before that row's event. A target result never
enters its own rating or feature construction. Practice drivers keep their own
identities, even when they do not enter qualifying.

Updating a rating after an event affects subsequent predictions, never a
previously stored prediction. An annual model for year `Y` is fitted only on
seasons before `Y`, so its learned rules never read that year's results.

## Driver ratings

A driver's rating at an event has three parts, each computed from evidence
available strictly before the cutoff.

### 1. Teammate network

The teammate component combines:

1. A robust Student-t loss on the gap in the **last common qualifying segment**.
2. A logistic loss on final qualifying order, with additional weight for a larger
   normalized final-position gap.
3. A driver lifetime effect, a stationary annual AR deviation, and a layout effect.

For example, two Q3 participants are compared in Q3. A Q1-eliminated driver and a
Q3 participant are compared in Q1. Missing or invalid laps do not imply elimination
and do not authorize silently substituting an earlier segment.

Teammate evidence of age `a` years receives `floor + (1 - floor) * 2**(-a / half_life)`.
The floor preserves older teammate links across seasons. Parameters and calibration
were selected independently using past rating validation and are recorded per year
in `configs/m1r1.json`. For a target year, validation reads only 2019 through
`min(target year − 1, 2022)`; 2010–2019 use preregistered initial constants
(`selection_max_year: 2022` is that cap, not a use of later seasons). The network is
refitted before every one of the 343 events from 2010 to 2026 R14.

### 2. Final qualifying achievement

A separate component rewards final main-qualifying positions. Each driver's own
valid classification counts individually: another entrant's disqualification or
missing result does not remove it. Positions shared by two drivers are quarantined,
and excluded, non-started or non-participating classifications do not count. The
original field size is kept, so position `p` in a field of `n` has quality
`q = (n - p) / (n - 1)`.

Evidence of age `a` years has weight `2**(-a / 1 year)`, with **no permanent floor**.
With prior mass `m0 = 20`, a driver's shrunk quality is
`(0.5 * m0 + Σ w q) / (m0 + Σ w)` and the bonus unit is
`max(0, 2 * (shrunk quality − 0.5))`, a value in `[0, 1]`. Both the weighted average
and its reliability therefore fade within about a year, while a driver with few
recent results stays close to neutral. The bonus adds **0.6 × unit** to the internal
ability, a fixed absolute weight in every year that is never trained. Drivers at or
below the neutral middle receive no bonus. Sunday race results and grid penalties
are not used. This component can include car/seat environment; it is not a pure
same-car skill estimate.

### 3. Shared corrections

After each event, every teammate pair yields an "innovation": how much the result
and last-common-segment gap exceeded what the pre-event network expected. For each
driver and target cutoff, earlier available innovations are summarized with fast
(0.5-year), slow (4-year) and 2-year half-lives, plus the extra performance at the
exact target layout. Five shared parameters map these summaries through bounded
`tanh` functions to a driver correction (at most ±0.2 internal units) and a
driver-by-layout correction (at most ±0.075). There are no free per-driver
parameters.

The rules are fitted inside each annual model, on earlier seasons only (see below).
Their loss includes the Top 3 prediction loss, an anchor penalty keeping corrections
small, the original same-car teammate evidence and a ridge penalty. The resulting
corrections are small (about ±0.01 internal units for current drivers). Their
uncertainty is not estimated, and the network solver itself is not re-optimized.

### Display

Displayed ability is `1 + 99 * normal_cdf(ability / 0.4)`; layouts use
`1 + 9 * normal_cdf(layout_effect / 0.1)`. These displays are not win percentages.
The composite rating uses teammate ability + correction + 0.6 × achievement unit;
"teammate ability" in the leaderboard is the same without the achievement term.
Rating bundles also retain latent estimates, evidence counts and conditional
uncertainty. Composite uncertainty is not fabricated by adding an independent
bonus variance.

## Current-season inputs

The 18 model columns are:

```text
practice_pct, practice_missing, practice_laps_log,
fp1_pct, fp2_pct, fp3_pct, fp1_missing, fp2_missing, fp3_missing,
team_recent, driver_recent, driver_median, team_track,
driver_history_log, team_history_log, changed_constructor,
ability, track_effect
```

`ability` is the composite rating and `track_effect` the corrected layout effect.
All detailed context is restricted to the target season before the cutoff, including
available practice for the current weekend. The latest eligible practice session and
each FP session retain separate inputs. Missing practice uses a neutral value plus
missingness flags.

`driver_recent`, `team_recent` and `team_track` use the last 24 valid
current-season observations, weights with half-life 12 observations, and the equal
average of weighted mean and weighted median. The median is the first value in
stable value order whose cumulative weight reaches half the total. The independent
`driver_median` feature retains its original definition. Team/layout effects keep
the original shrinkage toward the team's current-season center.

## Four-pool prediction and loss

Features are centered within the complete pre-qualifying field. Each annual model
for year `Y` is fitted in three steps on complete events from **2010 to `Y − 1`**
whose labels were available before January 1 of `Y`, excluding the first five events
of every season:

1. **Anchor fit.** Independent Q2/Q3/Top 3 logistic models and an event-grouped pole
   softmax on the network-anchor inputs (the teammate network plus the complete-session
   achievement memory at the same fixed weight 0.6). This fixes the feature scaling,
   and its Top 3 head is refitted with the quota-constrained Brier loss below to seed step 2.
2. **Joint Top 3 and rules.** The Top 3 head uses a sigmoid with an implicit event
   intercept that makes the probabilities sum to exactly three. Its event-weighted
   Brier loss, divided by `2q(1 − q)` for the event prevalence `q`, is minimized
   jointly with the shared correction rules from three starting points, using exact
   gradients through the corrections.
3. **Other heads.** Q2/Q3 logistic and pole softmax heads are refitted on the corrected
   m1r1 inputs.

Regularization is fixed at `0.3, 3, 0.3, 3` for Q2, Q3, Top 3 and pole. At least 12
complete training events are required per task; otherwise the task returns a
quota-based uniform prior (this happens only for 2010). Predictions still work
during the first five events, and their observed data inform later inputs.

Raw probabilities are retained. Euclidean projection enforces nested probabilities
and event-level capacities, then ranks are assigned jointly to maximize equal-task
expected membership overlap. Fine ordering within a pool is secondary. This
coherence operation is not a calibration guarantee; tiny projected tail
probabilities are a known limitation.

## 2010–2015 reconstruction

The 115 early events are rebuilt retrospectively from F1DB and reviewed official
classifications. Actual session clocks and original publication times are unknown
and stay null. A monotone order proxy keeps time arithmetic working but never admits
practice: FP1/FP2/FP3 before the first qualifying segment are admitted by reviewed
session order (the cancelled 2015 US FP2 and Q3 are handled explicitly). Results
count as available from race date + 36 hours. Early rosters are post-event actual
participants, and early qualifying quotas follow the reviewed rules of each season.

Early teammate pace evidence additionally admits a fixed, reviewed set of **136**
pairs from 2013–2014 whose last common segment was confirmed from primary rule and
report sources (`resources/early_pace_supplement.json`). Final classifications and
outcomes are not changed. From 2011 onward, early events enter annual training like
any other season, subject to the same cutoffs.

## Evaluation

Each task is scored only on events with a complete matching roster and complete
labels for that task. Missing labels remain unknown. Q2/Q3 quotas use reviewed
season/format rules, not the number of drivers who happened to set a time.

For task quota `k` in a field of `n`, normalized Brier divides mean squared probability
error by `(k/n) * (1-k/n)`. The primary probability score averages the four normalized
Brier scores per complete event, then averages events. Membership overlap,
whole-list exact matches, log loss and full-field rank MAE are reported separately.
First-five and all-event summaries are separate from the main sixth-event-onward score.

## Previous method: v6

v6 (`memory_h12_mean_median_50_50`, research lock `763e7f1871a6a6d3`) shares the
teammate network, current-season inputs and projection. It differs in that:

- the achievement component admitted only sessions where every entrant had a
  complete numeric classification, and shared the network's floor and half-life
  (floor 0.25, two years from 2020), so old final positions never fully faded;
- its achievement weight was 0.2 (0.1 before 2020), and there were no learned
  corrections: ratings were completely independent of prediction training;
- Top 3 used a logistic model like Q2/Q3; and
- predictors trained on 2016 onward, with 2010–2015 used only for rating warmup.

Run it with `f1-quali prepare --method v6 --data data/canonical ...`; its configuration
is `configs/v6.json`.
