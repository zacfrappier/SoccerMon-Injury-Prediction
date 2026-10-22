
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 01. Paths and reconstruction settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "paper_inspired"
)

PLAYER_DAY_FILE = (
    BASE_DIR
    / "player_day"
    / "player_day_unimputed.csv"
)

WINDOW_AUDIT_FILE = (
    BASE_DIR
    / "player_day"
    / "audit"
    / "player_day_window_eligibility.csv"
)

OUTPUT_DIR = BASE_DIR / "survival"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FILE = OUTPUT_DIR / "survival_targets.csv"
EVENT_FILE = OUTPUT_DIR / "survival_event_summary.csv"
PLAYER_FILE = OUTPUT_DIR / "survival_player_summary.csv"
SUMMARY_FILE = OUTPUT_DIR / "survival_target_summary.txt"

HISTORY_DAYS = 21
HORIZON_DAYS = 7

# Use public candidate injury-episode start dates.
# This is not equivalent to the private objective injury
# reports used in the published study.

# We retain incomplete follow-up as administrative
# censoring candidates, rather than calling them
# confirmed injury-free observations.

# Injury surveillance completeness is NOT verified.


# ============================================================
# 02. Helpers
# ============================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input missing:\n{path}"
        )


def parse_bool(series):
    values = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    unknown = ~values.isin(mapping)

    if unknown.any():
        raise ValueError(
            "Unexpected boolean values: "
            + str(values[unknown].unique().tolist())
        )

    return values.map(mapping).astype(bool)


# ============================================================
# 03. Load player-day data
# ============================================================

def load_player_day():
    require_file(PLAYER_DAY_FILE)

    df = pd.read_csv(PLAYER_DAY_FILE)

    required = {
        "player_name",
        "team",
        "date",
        "injury_event",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing player-day columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.normalize()

    df["injury_event"] = pd.to_numeric(
        df["injury_event"], errors="raise"
    ).astype(int)

    if not df["injury_event"].isin([0, 1]).all():
        raise ValueError(
            "injury_event must contain only 0 or 1."
        )

    if df.duplicated(["player_name", "date"]).any():
        raise ValueError(
            "Duplicate player-date rows in canonical table."
        )

    if not df["team"].eq("TeamB").all():
        raise ValueError(
            "Script 10 expects TeamB-only data."
        )

    return df.sort_values(
        ["player_name", "date"]
    ).reset_index(drop=True)


# ============================================================
# 04. Load Audit 07 window diagnostics
# ============================================================

def load_window_audit():
    require_file(WINDOW_AUDIT_FILE)

    audit = pd.read_csv(WINDOW_AUDIT_FILE)

    required = {
        "player_name",
        "prediction_date",
        "has_full_history_calendar",
        "has_full_followup_calendar",
        "calendar_eligible",
    }

    missing = required - set(audit.columns)

    if missing:
        raise ValueError(
            f"Missing audit columns: {sorted(missing)}"
        )

    audit["prediction_date"] = pd.to_datetime(
        audit["prediction_date"],
        errors="raise",
    ).dt.normalize()

    for column in audit.columns:
        if (
            column.endswith("_eligible")
            or column.startswith("has_full_")
            or column == "injury_on_prediction_day"
        ):
            audit[column] = parse_bool(audit[column])

    if audit.duplicated(
        ["player_name", "prediction_date"]
    ).any():
        raise ValueError(
            "Duplicate player prediction dates in audit."
        )

    return audit


# ============================================================
# 05. Build survival targets
#
# Prediction date: t
# History:         t-21 through t-1
# Future:          t+1 through t+7
#
# No imputed feature values are used to create labels.
# ============================================================

def build_survival_targets(player_day, audit):
    rows = []

    for player, group in player_day.groupby(
        "player_name", sort=True
    ):
        group = group.sort_values("date").copy()

        team = group["team"].iloc[0]

        first_date = group["date"].min()
        last_date = group["date"].max()

        expected_dates = pd.date_range(
            first_date, last_date, freq="D"
        )

        if not pd.DatetimeIndex(
            group["date"]
        ).equals(expected_dates):
            raise ValueError(
                f"Calendar contains gaps for {player}"
            )

        injury_dates = pd.DatetimeIndex(
            group.loc[
                group["injury_event"] == 1,
                "date",
            ].sort_values()
        )

        # Need 21 complete calendar days before prediction.
        prediction_dates = group.loc[
            group["date"] >= first_date
            + pd.Timedelta(days=HISTORY_DAYS),
            "date",
        ]

        for prediction_date in prediction_dates:

            # Injury-start days are not prediction origins.
            if prediction_date in injury_dates:
                continue

            history_start = (
                prediction_date
                - pd.Timedelta(days=HISTORY_DAYS)
            )

            history_end = (
                prediction_date - pd.Timedelta(days=1)
            )

            followup_start = (
                prediction_date + pd.Timedelta(days=1)
            )

            planned_followup_end = (
                prediction_date
                + pd.Timedelta(days=HORIZON_DAYS)
            )

            # Available future calendar days.
            observed_followup_end = min(
                planned_followup_end, last_date
            )

            available_followup_days = (
                observed_followup_end
                - prediction_date
            ).days

            # No future follow-up means no usable target.
            if available_followup_days < 1:
                continue

            future_injuries = injury_dates[
                (injury_dates >= followup_start)
                & (injury_dates <= observed_followup_end)
            ]

            if len(future_injuries) > 0:
                first_future_injury = future_injuries[0]

                duration = int(
                    (first_future_injury
                     - prediction_date).days
                )

                event = 1
                censor_reason = "injury_observed"

            else:
                first_future_injury = pd.NaT
                duration = int(available_followup_days)
                event = 0

                if available_followup_days == HORIZON_DAYS:
                    censor_reason = "horizon_reached"
                else:
                    censor_reason = "calendar_end"

            # 0 indicates censoring, not a verified
            # seven-day injury-free clinical assessment.
            full_followup = (
                available_followup_days == HORIZON_DAYS
            )

            rows.append({
                "player_name": player,
                "team": team,
                "prediction_date": prediction_date,
                "history_start": history_start,
                "history_end": history_end,
                "followup_start": followup_start,
                "planned_followup_end":
                    planned_followup_end,
                "observed_followup_end":
                    observed_followup_end,
                "available_followup_days":
                    available_followup_days,
                "has_full_history_calendar": True,
                "has_full_followup_calendar":
                    full_followup,
                "event": event,
                "duration_days": duration,
                "injury_within_7d": bool(event),
                "first_future_injury_date":
                    first_future_injury,
                "censor_reason": censor_reason,
                "injury_surveillance_verified": False,
                "target_assumption":
                    "public_injury_records_complete",
            })

    targets = pd.DataFrame(rows)

    if targets.empty:
        raise ValueError(
            "No survival targets could be generated."
        )

    # --------------------------------------------------------
    # Merge coverage diagnostics from Audit 06.
    # --------------------------------------------------------

    audit_columns = [
        column for column in audit.columns
        if column not in {
            "team",
            "history_start",
            "history_end",
            "followup_start",
            "followup_end",
            "first_future_injury_date",
            "days_until_injury",
            "injury_within_7d",
            "future_injury_event_count",
            "has_full_history_calendar",
            "has_full_followup_calendar",
        }
    ]

    targets = targets.merge(
        audit[audit_columns],
        on=["player_name", "prediction_date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not targets["_merge"].eq("both").all():
        raise ValueError(
            "Some targets are missing Audit 07 rows."
        )

    targets = targets.drop(columns="_merge")

    # Audit 07 calendar_eligible additionally requires
    # full seven-day follow-up. Thus some administrative
    # censoring candidates will not be calendar_eligible.

    return targets.sort_values(
        ["player_name", "prediction_date"]
    ).reset_index(drop=True)


# ============================================================
# 07. Validate target consistency
# ============================================================

def validate_targets(targets, player_day):
    if targets.duplicated(
        ["player_name", "prediction_date"]
    ).any():
        raise ValueError("Duplicate survival targets.")

    if not targets["event"].isin([0, 1]).all():
        raise ValueError("Invalid event indicator.")

    if not targets["duration_days"].between(
        1, HORIZON_DAYS
    ).all():
        raise ValueError("Invalid survival duration.")

    if not (
        (targets["history_end"]
         - targets["history_start"]).dt.days
        == HISTORY_DAYS - 1
    ).all():
        raise ValueError("Invalid history length.")

    if not (
        (targets["prediction_date"]
         - targets["history_end"]).dt.days == 1
    ).all():
        raise ValueError("History includes prediction date.")

    positive = targets[targets["event"] == 1]

    if positive["first_future_injury_date"].isna().any():
        raise ValueError(
            "Positive target without injury date."
        )

    expected_duration = (
        positive["first_future_injury_date"]
        - positive["prediction_date"]
    ).dt.days

    if not expected_duration.eq(
        positive["duration_days"]
    ).all():
        raise ValueError(
            "Time-to-event calculation mismatch."
        )

    if targets.loc[
        targets["event"] == 0,
        "first_future_injury_date",
    ].notna().any():
        raise ValueError(
            "Censored target has future injury date."
        )

    # Verify each positive label maps to an actual
    # injury date in the original canonical table.
    real_events = set(
        zip(
            player_day.loc[
                player_day["injury_event"] == 1,
                "player_name",
            ],
            player_day.loc[
                player_day["injury_event"] == 1,
                "date",
            ],
        )
    )

    for row in positive.itertuples():
        if (
            row.player_name,
            row.first_future_injury_date,
        ) not in real_events:
            raise ValueError(
                "Positive target does not match "
                "an observed injury-event date."
            )


# ============================================================
# 08. Event-level summary
#
# One injury can generate several positive windows.
# ============================================================

def build_event_summary(player_day, targets):
    events = player_day.loc[
        player_day["injury_event"] == 1
    ].copy()

    rows = []

    for event in events.itertuples():
        supported = targets.loc[
            (targets["player_name"] == event.player_name)
            & (targets["event"] == 1)
            & (
                targets["first_future_injury_date"]
                == event.date
            )
        ]

        rows.append({
            "player_name": event.player_name,
            "team": event.team,
            "injury_date": event.date,
            "positive_prediction_windows":
                len(supported),
            "has_positive_window":
                len(supported) > 0,
            "injury_episode_count":
                getattr(event, "injury_episode_count", np.nan),
            "body_regions":
                getattr(event, "body_regions", ""),
        })

    return pd.DataFrame(rows)


# ============================================================
# 09. Player-level summary
# ============================================================

def build_player_summary(targets):
    rows = []

    for player, group in targets.groupby(
        "player_name", sort=True
    ):
        rows.append({
            "player_name": player,
            "team": group["team"].iloc[0],
            "survival_windows": len(group),
            "positive_windows":
                int(group["event"].sum()),
            "censored_windows":
                int((group["event"] == 0).sum()),
            "full_followup_windows":
                int(group[
                    "has_full_followup_calendar"
                ].sum()),
            "calendar_end_censored_windows":
                int((
                    group["censor_reason"]
                    == "calendar_end"
                ).sum()),
        })

    return pd.DataFrame(rows)


# ============================================================
# 10. Main
# ============================================================

def main():
    print("=" * 80)
    print("SoccerMon Paper-Inspired Survival Target Builder")
    print("=" * 80)

    player_day = load_player_day()
    audit = load_window_audit()

    print(f"Players: {player_day['player_name'].nunique()}")
    print(f"Player-days: {len(player_day):,}")
    print(f"History: {HISTORY_DAYS} days")
    print(f"Forecast horizon: {HORIZON_DAYS} days")
    print()

    targets = build_survival_targets(
        player_day, audit
    )

    validate_targets(targets, player_day)

    events = build_event_summary(
        player_day, targets
    )

    players = build_player_summary(targets)

    targets.to_csv(TARGET_FILE, index=False)
    events.to_csv(EVENT_FILE, index=False)
    players.to_csv(PLAYER_FILE, index=False)

    positive = int(targets["event"].sum())
    censored = int((targets["event"] == 0).sum())

    supported_events = int(
        events["has_positive_window"].sum()
    )

    summary = [
        "=" * 80,
        "SoccerMon Paper-Inspired Survival Summary",
        "=" * 80,
        "",
        f"Players: {targets['player_name'].nunique()}",
        f"Survival observations: {len(targets):,}",
        f"Positive prediction windows: {positive:,}",
        f"Censored prediction windows: {censored:,}",
        f"Observed public injury-event days: {len(events):,}",
        f"Injuries supporting positive windows: "
        f"{supported_events:,}",
        "",
        "Censoring reasons:",
        targets["censor_reason"].value_counts().to_string(),
        "",
        "Interpretation:",
        "History: t-21 through t-1.",
        "Outcome horizon: t+1 through t+7.",
        "The earliest injury in follow-up determines duration.",
        "Missing predictors are not filtered or imputed.",
        "Injury surveillance completeness is unverified.",
        "Full follow-up is based on calendar coverage only.",
        "No return-to-play dates are available.",
        "Multiple windows may refer to the same injury.",
        "Fold-specific imputation is still required.",
    ]

    SUMMARY_FILE.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print("\n".join(summary))
    print()
    print(f"Targets: {TARGET_FILE}")
    print(f"Event summary: {EVENT_FILE}")
    print(f"Player summary: {PLAYER_FILE}")
    print(f"Text summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
