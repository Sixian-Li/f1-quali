# Fixed benchmark summaries

These small files document the selected v6 and its limitations. They contain no raw
source archive, pretrained weights, driver-level prediction dump or experiment grid.

| File | Contents |
| --- | --- |
| `v6_yearly.csv` | 2016–2026 metrics and task-specific counts; main, first-five, all-event and raw-probability scopes |
| `training_years.csv` | Annual model training years, sample counts and fallback status |
| `baseline_comparison.csv` | Rolling OLS and practice baselines compared with v6 on identical complete events in 2024/2026 |
| `refactor_verification.json` | Numerical equivalence checks against the locked research implementation |
| `provenance.json` | Selected method, original lock, fixed source identity and hashes of summary files |
| `source_acquisition.json` | Fresh-cache check and the known remote snapshot limitation |

Pool overlap measures selected-driver membership, regardless of within-pool order.
Each task retains its own complete-event denominator. First-five performance is
reported separately from the sixth-event-onward optimization scope. Baselines do
not invent four-pool probabilities; their comparison covers membership and rank
error only. The practice baseline uses the course feature's FP2, then FP3, then FP1
preference among sessions available at the cutoff.

The source cutoff is 2026-09-15, through 2026 R14. 2010–2015 supplies independent
rating warmup, not prediction training examples. Development and evaluation periods
have already been inspected. Consult [results and limitations](../docs/results_and_limitations.md)
before interpreting these numbers as evidence of future accuracy.

The statistics are computed by Sixian Li's F1 Quali project using normalized and
reviewed [F1DB](https://github.com/f1db/f1db) data (CC BY 4.0), f1schedule metadata
and referenced official sources. They are derived evaluation results, not original
source tables. Attribution and source terms are detailed in
[third-party notices](../THIRD_PARTY_NOTICES.md).

To reproduce all four annual scopes of `v6_yearly.csv`, run the documented full
workflow and `scripts/report_benchmarks.py`. `baseline_comparison.csv` pairs the
`baseline` command's event metrics with the v6 evaluations on identical complete events,
and `training_years.csv` summarizes the training traces stored in each `model.json`;
both were assembled by the research code, and this release has no dedicated script for them. The raw-probability scope keeps the same final hard
selections, changing only the probabilities used by Brier and log-loss metrics.
