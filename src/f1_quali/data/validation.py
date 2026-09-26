"""Conservative qualifying label checks."""

import numpy as np
import pandas as pd


def _positive_time(row, stage):
    value = row.get(f"q{stage}_seconds") if row is not None else None
    return bool(pd.notna(value) and np.isfinite(value) and value > 0)


def _valid_label(row, size):
    if row is None:
        return False
    position = row.position
    return bool(
        pd.notna(row.label_usable)
        and row.label_usable
        and row.source_row_present
        and row.classification_status not in {"NC", "DSQ", "DQ", "EX"}
        and pd.notna(position)
        and np.isfinite(position)
        and position == int(position)
        and 1 <= position <= size
        and pd.notna(row.label_available_at)
    )


def participation(row, size, q2_limit, q3_limit, exclusion_event):
    """Stage reached/eligible, including advanced drivers who set no valid lap."""
    if not _valid_label(row, size) or exclusion_event:
        if _positive_time(row, 3):
            return 3, "positive_q3_time"
        return 0, "classification_stage_ambiguous"
    stage = 3 if row.position <= q3_limit else 2 if row.position <= q2_limit else 1
    if any(_positive_time(row, s) for s in range(stage + 1, 4)):
        raise ValueError("Positive later-stage time contradicts inferred classification stage")
    if _positive_time(row, 3):
        return 3, "positive_q3_time"
    return stage, "rules_and_classification_inference"
