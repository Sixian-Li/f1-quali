"""Schedule dates are planned/revised starts, not measured timing or publication times."""

import json
from pathlib import Path

import pandas as pd

from f1_quali.sources.cache import fetch_snapshot

SESSION_NAMES = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SQ",
    "Sprint": "S",
    "Race": "R",
}
F1DB_PREFIXES = {
    "FP1": "freePractice1",
    "FP2": "freePractice2",
    "FP3": "freePractice3",
    "Q": "qualifying",
    "SQ": "sprintQualifying",
    "S": "sprintRace",
}


def load_schedules(root: Path, config: dict, *, offline: bool = False) -> dict:
    commit = config["commit"]
    schedules = {}
    for year, digest in config["sha256"].items():
        name = f"schedule_{year}.json"
        data = fetch_snapshot(
            f"https://raw.githubusercontent.com/theOehrly/f1schedule/{commit}/{name}",
            root / "schedules" / commit / name,
            digest,
            offline=offline,
        )
        schedules[int(year)] = pd.DataFrame(json.loads(data))
    return schedules


def schedule_candidate(schedule: pd.DataFrame | None, race_date: str) -> dict:
    """Match dates, never assume different providers' round numbers agree."""
    if schedule is None:
        return {}
    date = pd.Timestamp(race_date).date()
    matches = schedule[
        (schedule.round_number > 0) & (pd.to_datetime(schedule.event_date).dt.date == date)
    ]
    # Night races can have a UTC calendar date different from the local date.
    if matches.empty:
        matches = schedule[
            schedule.apply(
                lambda row: (
                    row.round_number > 0
                    and any(
                        row[f"session{i}"] == "Race"
                        and pd.notna(row[f"session{i}_date"])
                        and pd.Timestamp(str(row[f"session{i}_date"]) + row.gmt_offset)
                        .tz_convert("UTC")
                        .date()
                        == date
                        for i in range(1, 6)
                    )
                ),
                axis=1,
            )
        ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous schedule date: {date}")
    if matches.empty:
        return {}
    row = matches.iloc[0]
    output = {}
    for i in range(1, 6):
        name, local_date = row[f"session{i}"], row[f"session{i}_date"]
        if name in SESSION_NAMES and local_date and pd.notna(local_date):
            timestamp = pd.Timestamp(str(local_date) + row.gmt_offset).tz_convert("UTC")
            output[SESSION_NAMES[name]] = timestamp
    return output


def build_sessions(
    races: pd.DataFrame, schedules: dict, overrides: list[dict], *, buffer_minutes: int
) -> pd.DataFrame:
    curated = {(r["year"], r["round"]): r for r in overrides}
    records = []
    for race in races.to_dict("records"):
        year, rnd = int(race["year"]), int(race["round"])
        candidate = schedule_candidate(schedules.get(year), race["date"])
        manual = curated.get((year, rnd), {})
        types = {"FP1", "FP2", "FP3", "Q"} | set(candidate)
        for kind in sorted(types):
            prefix = F1DB_PREFIXES.get(kind)
            day = race.get(f"{prefix}Date") if prefix else None
            time = race.get(f"{prefix}Time") if prefix else None
            start, source = pd.NaT, "unknown"
            conflict = False
            if pd.notna(day) and pd.notna(time) and day and time:
                start = pd.Timestamp(f"{day}T{time}:00Z")
                source = "f1db"
                conflict = kind in candidate and candidate[kind] != start
            elif kind in candidate:
                start, source = candidate[kind], "f1schedule"
            elif kind in manual.get("sessions", {}):
                start = pd.Timestamp(manual["sessions"][kind])
                source = "curated_schedule"
            # Unknown dates/conflicts fail closed in features; never infer Friday/Saturday.
            available = (
                start + pd.Timedelta(minutes=buffer_minutes)
                if pd.notna(start) and not conflict
                else pd.NaT
            )
            records.append(
                {
                    "event_id": int(race["id"]),
                    "season": year,
                    "round": rnd,
                    "session_id": f"{int(race['id'])}:{kind}",
                    "session_type": kind,
                    "scheduled_start_utc": start,
                    "available_at": available,
                    "actual_start_utc": pd.NaT,
                    "actual_end_utc": pd.NaT,
                    "time_source": source,
                    "schedule_conflict": conflict,
                    "alternate_start_utc": candidate.get(kind, pd.NaT),
                    "availability_basis": "scheduled_start_plus_buffer_reconstruction",
                    "schedule_reference": manual.get("source", "")
                    if source.startswith("curated")
                    else source,
                }
            )
    out = pd.DataFrame(records)
    for col in [
        "scheduled_start_utc",
        "available_at",
        "actual_start_utc",
        "actual_end_utc",
        "alternate_start_utc",
    ]:
        out[col] = pd.to_datetime(out[col], utc=True)
    return out.sort_values(["season", "round", "session_type"]).reset_index(drop=True)
