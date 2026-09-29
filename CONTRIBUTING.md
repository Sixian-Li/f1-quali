# Maintaining the selected m1r1

Install the locked Python 3.13 environment as described in the README, then run:

```sh
python -m pytest -W error
ruff check src tests scripts
python -m build --no-isolation
```

Tests block external network connections. Use the synthetic demo for functional
checks; real-data reconstruction is a separate, explicit workflow.

Preserve these boundaries when changing the implementation:

- Every input has a prediction cutoff. Target results, unavailable sessions and
  future-season statistics must not affect earlier features, ratings or fits.
- Driver, constructor, layout and session identities remain explicit; a practice
  substitute is not silently assigned to the regular driver.
- Rating networks update per event and never read prediction results. m1r1's learned
  correction rules are fitted inside each annual model on earlier seasons only.
- The 2010–2015 reconstruction never invents session or publication times.
- Missing source rows are not elimination labels. Report complete-event and
  task-specific coverage alongside accuracy.
- Years already inspected do not become untouched test sets again.
- Keep source versions, reviews, checksums and reasons for missing values.

The previous method v6 stays runnable for reproduction; do not change its numbers.
Changing the selected statistical method requires a separately named configuration,
predeclared comparison and new evidence; it is not a packaging fix. Keep numerical
refactor checks separate from claims about predictive improvement. New source
versions or rules require review instead of silently changing pinned inputs.

Do not commit raw caches, local environments, model outputs, credentials, private
work logs or experiment archives. Run packaging and the demo from a clean installed
wheel before a release. GitHub Actions runs the offline checks on macOS and Linux;
the historical numerical verification covers macOS arm64 only.
