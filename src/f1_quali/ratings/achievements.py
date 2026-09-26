"""Full-field final qualifying evidence, excluding Sunday results."""

from f1_quali.features.evidence import complete_labels


def qualifying_achievements(data):
    """Complete final main-Q classifications; no grid or Sunday race results."""
    labels = complete_labels(data)
    q = data.qualifying.merge(
        labels[["event_id", "driver_id", "training_label_available_at"]],
        on=["event_id", "driver_id"],
        validate="one_to_one",
    )
    if "season" in q:
        q = q.drop(columns="season")
    q = q.merge(
        data.events[["event_id", "season", "prediction_cutoff"]],
        on="event_id",
        validate="many_to_one",
    )
    q["field_size"] = q.groupby("event_id").event_id.transform("size")
    q["age_at"] = q.prediction_cutoff
    q["available_at"] = q.training_label_available_at
    q["age_time_basis"] = "canonical_main_Q_cutoff_reconstructed_session_start"
    return q.drop(columns="training_label_available_at")
