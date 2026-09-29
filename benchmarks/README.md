# Fixed benchmark summaries

These small files document the selected m1r1 method, the previous release method v6,
and their limitations. They contain no raw source archive, pretrained weights,
driver-level prediction dump or experiment grid.

| File | Contents |
| --- | --- |
| `m1r1_yearly.csv` | 2010–2026 m1r1 metrics and task-specific counts; main, first-five, all-event and raw-probability scopes |
| `m1r1_training_years.csv` | Annual m1r1 training years, event counts, head types and learned correction rules |
| `m1r1_baseline_comparison.csv` | Rolling OLS and practice baselines against m1r1 on identical complete events, per year and for 2017–2022 / 2023–2026 |
| `m1r1_verification.json` | Numerical equivalence of the public m1r1 port with the locked research run |
| `v6_yearly.csv`, `training_years.csv`, `baseline_comparison.csv` | The same summaries for the previous method v6 (2016–2026; baselines 2024/2026) |
| `refactor_verification.json` | v6 numerical equivalence with its research implementation |
| `provenance.json` | Selected method, locks, fixed source identity and hashes of summary files |
| `source_acquisition.json` | Fresh-cache check and the known remote snapshot limitation |

Pool overlap measures selected-driver membership, regardless of within-pool order.
Each task retains its own complete-event denominator. First-five performance is
reported separately from the sixth-event-onward optimization scope. Baselines do
not invent four-pool probabilities; their comparison covers membership and rank
error only. The practice baseline uses the course feature's FP2, then FP3, then FP1
preference among sessions available at the cutoff. Baselines skip events without a
usable practice time; comparisons use only events both methods scored.

The source cutoff is 2026-09-15, through 2026 R14. For m1r1, 2010–2015 is a
retrospective reconstruction used for ratings and predictor training; 2010 has no
earlier training data and uses uniform priors. For v6, 2010–2015 supplied rating
warmup only. Validation and later periods have already been inspected. Consult
[results and limitations](../docs/results_and_limitations.md) before interpreting
these numbers as evidence of future accuracy.

The statistics are computed by Sixian Li's F1 Quali project using normalized and
reviewed [F1DB](https://github.com/f1db/f1db) data (CC BY 4.0), f1schedule metadata
and referenced official sources. They are derived evaluation results, not original
source tables. Attribution and source terms are detailed in
[third-party notices](../THIRD_PARTY_NOTICES.md).

To reproduce the m1r1 files, run the documented full workflow; `scripts/report_benchmarks.py`
writes `m1r1_yearly.csv` from m1r1 evaluations (or `v6_yearly.csv` from v6 ones), and
`scripts/report_m1r1_benchmarks.py` writes the baseline comparison and training table.
The v6 `baseline_comparison.csv` and `training_years.csv` were assembled by the
research code. The raw-probability scope keeps the same final hard selections,
changing only the probabilities used by Brier and log-loss metrics.
