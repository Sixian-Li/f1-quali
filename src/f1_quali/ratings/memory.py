"""m1r1 achievement memory: individually admitted results, no permanent floor.

The teammate network in a rating bundle is kept exactly. Only its external
final-qualifying achievement component is rebuilt: each driver's own valid
main-Q classification counts even when another entrant in that session was
excluded, and the weighted mean and its reliability both decay with a one-year
half-life towards a neutral prior. Sunday race results are never used.
"""

import copy

import numpy as np
import pandas as pd

from f1_quali.ratings.core import canonical_hash, verify_bundle
from f1_quali.ratings.rolling import with_achievement_bonus

UNUSABLE = ["EX", "DSQ", "DQ", "NC", "RT", "DNS", "DNQ", "NOT_PARTICIPATED"]


def individual_achievements(data):
    """Admit valid persons, keep the original field size, quarantine duplicate places."""
    counts = data.entries.groupby("event_id").size()
    rows = data.qualifying.copy()
    rows["field_size"] = rows.event_id.map(counts)
    rows = rows[
        rows.label_usable
        & rows.source_row_present
        & rows.position.gt(0)
        & rows.position.le(rows.field_size)
        & rows.position.mod(1).eq(0)
        & rows.label_available_at.notna()
        & ~rows.classification_status.isin(UNUSABLE)
    ].copy()
    duplicates = rows.duplicated(["event_id", "position"], keep=False)
    quarantined = rows[duplicates].copy()
    rows = rows[~duplicates].copy()
    meta = data.events[["event_id", "prediction_cutoff", "circuit_layout_id"]]
    rows = rows.merge(meta, on="event_id", validate="many_to_one")
    rows["age_at"] = rows.prediction_cutoff
    # Conservative timing: the whole session's latest label availability, even
    # when only one person's result is used.
    upper = data.qualifying.groupby("event_id").label_available_at.max()
    rows["available_at"] = pd.to_datetime(rows.event_id.map(upper), utc=True)
    return rows, quarantined


def eligible_individual(rows, cutoff, target_event_id):
    """Earlier, available personal main-Q results; never the target event."""
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("Timezone-aware cutoff required")
    part = rows[rows.event_id.ne(target_event_id)].copy()
    if "minimum_target_season" in part:
        part = part[part.minimum_target_season.isna() | part.minimum_target_season.le(cutoff.year)]
    if "rating_warmup_eligible" in part:
        part = part[~part.rating_warmup_eligible.eq(False)]
    for column in ["age_at", "available_at"]:
        part[column] = pd.to_datetime(part[column], utc=True)
    part = part[part.age_at.lt(cutoff) & part.available_at.le(cutoff)].copy()
    if not part.session_type.eq("Q").all():
        raise ValueError("Only main qualifying is allowed")
    if part.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Duplicate personal qualifying evidence")
    numbers = part[["position", "field_size"]].to_numpy(float)
    if not np.isfinite(numbers).all() or (numbers != np.floor(numbers)).any():
        raise ValueError("Finite integer positions and entrant counts required")
    if ((part.field_size < 2) | (part.position < 1) | (part.position > part.field_size)).any():
        raise ValueError("Invalid personal qualifying position")
    if part.duplicated(["event_id", "position"]).any():
        raise ValueError("Conflicting numeric positions must be quarantined")
    if "classification_status" in part and part.classification_status.isin(UNUSABLE).any():
        raise ValueError("Unusable personal classification")
    return part.sort_values(["age_at", "event_id", "driver_id"]).reset_index(drop=True)


def achievement_statistics(rows, cutoff, *, floor, half_life_years, prior_mass):
    """Shrunk weighted final-position quality per driver, on the original bonus scale."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or not 0 <= floor <= 1 or half_life_years <= 0 or prior_mass <= 0:
        raise ValueError("Timezone-aware cutoff and valid positive memory parameters required")
    if not np.isfinite([floor, half_life_years, prior_mass]).all():
        raise ValueError("Finite memory parameters required")
    part = rows.copy()
    ages = (cutoff - part.age_at).dt.total_seconds().to_numpy() / (365.25 * 86400)
    if not np.isfinite(ages).all() or (ages < 0).any():
        raise ValueError("Evidence must precede the cutoff")
    if (
        len(part)
        and ((part.field_size < 2) | (part.position < 1) | (part.position > part.field_size)).any()
    ):
        raise ValueError("Invalid qualifying position")
    part["weight"] = floor + (1 - floor) * np.exp2(-ages / half_life_years)
    part["q"] = (part.field_size - part.position) / (part.field_size - 1)
    result, components = {}, []
    for driver, g in part.groupby("driver_id", sort=True):
        w = g.weight.to_numpy(float)
        mass = float(w.sum())
        mean = float(np.dot(w, 2 * g.q - 1) / mass) if mass else 0.0
        q = float((0.5 * prior_mass + np.dot(w, g.q)) / (prior_mass + mass))
        result[driver] = {
            "q_shrunk": q,
            "bonus_unit": max(0.0, 2 * (q - 0.5)),
            "events": len(g),
            "weight_mass": mass,
            "effective_sample_size": float(mass**2 / np.square(w).sum()) if mass else 0.0,
        }
        components.append(
            {
                "driver_id": driver,
                "weight_mass": mass,
                "weighted_centered_mean": mean,
                "reliability": mass / (mass + prior_mass),
                "bonus_unit": result[driver]["bonus_unit"],
            }
        )
    return result, pd.DataFrame(components)


def memory_statistics(rows, cutoff, memory):
    """Decay the mean and its reliability separately; old mass is never renormalized."""
    mean_half, mass_half, prior = (
        memory["mean_half_life"],
        memory["mass_half_life"],
        memory["prior_mass"],
    )
    if (
        not np.isfinite([mean_half, mass_half, prior]).all()
        or min(mean_half, mass_half, prior) <= 0
    ):
        raise ValueError("Finite positive memory parameters required")
    if memory.get("floor", 0.0) != 0.0:
        raise ValueError("m1r1 memory has no permanent floor")
    mean_summary, parts = achievement_statistics(
        rows, cutoff, floor=0.0, half_life_years=mean_half, prior_mass=prior
    )
    if mean_half == mass_half:
        return mean_summary
    mass_summary, _ = achievement_statistics(
        rows, cutoff, floor=0.0, half_life_years=mass_half, prior_mass=prior
    )
    answer = {}
    for row in parts.itertuples():
        mass = mass_summary[row.driver_id]["weight_mass"]
        centered = row.weighted_centered_mean * mass / (prior + mass)
        answer[row.driver_id] = {
            **mean_summary[row.driver_id],
            "weight_mass": mass,
            "mean_weight_mass": row.weight_mass,
            "mass_effective_sample_size": mass_summary[row.driver_id]["effective_sample_size"],
            "q_shrunk": 0.5 + 0.5 * centered,
            "bonus_unit": max(0.0, centered),
        }
    return answer


def rebuild_achievement(bundle, rows, memory, evidence_sha256):
    """Keep the network exactly; replace only the achievement component."""
    verify_bundle(bundle)
    admitted = eligible_individual(rows, bundle["cutoff"], bundle["target_event_id"])
    summary = memory_statistics(admitted, bundle["cutoff"], memory)
    out = copy.deepcopy(bundle)
    out["achievement"] = summary
    out["achievement_memory"] = {
        "floor": 0.0,
        "prior_mass": memory["prior_mass"],
        "admission": memory["admission"],
        "mean_half_life": memory["mean_half_life"],
        "mass_half_life": memory["mass_half_life"],
        "dual": False,
    }
    out["achievement_provenance"] = {
        "parent_bundle_sha256": bundle["bundle_sha256"],
        "raw_sha256": evidence_sha256,
        "rows": len(admitted),
        "events": int(admitted.event_id.nunique()),
        "max_available_at": admitted.available_at.max().isoformat() if len(admitted) else None,
        "max_age_at": admitted.age_at.max().isoformat() if len(admitted) else None,
        "network_diagnostics_scope": ("original network fit; achievement admission replaced here"),
    }
    out.pop("bundle_sha256")
    out["bundle_sha256"] = canonical_hash(out)
    out = with_achievement_bonus(
        out, bundle["achievement_beta"], calibration_slope=bundle["composite_calibration_slope"]
    )
    for name in ["config", "layout", "base_joint_covariance"]:
        if out[name] != bundle[name]:
            raise RuntimeError(f"Achievement rebuild changed the network: {name}")
    for driver, before in bundle["base"].items():
        for key, value in before.items():
            if key not in {
                "achievement_bonus",
                "composite_ability",
                "composite_score",
                "achievement",
            }:
                if out["base"][driver][key] != value:
                    raise RuntimeError(f"Achievement rebuild changed the network: {key}")
    return out
