from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audit"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "paper_inspired"
    / "cohort"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Input files
#
# These are outputs from the audits we already completed.
# ============================================================

PLAYER_OVERLAP_FILE = (
    AUDIT_DIR
    / "player_overlap_audit.csv"
)

OBJECTIVE_MANIFEST_FILE = (
    AUDIT_DIR
    / "objective_file_manifest.csv"
)

INJURY_PLAYER_FILE = (
    AUDIT_DIR
    / "injury_player_summary.csv"
)


# ============================================================
# Output files
# ============================================================

COHORT_FILE = (
    OUTPUT_DIR
    / "player_cohort.csv"
)

OBSERVATION_WINDOW_FILE = (
    OUTPUT_DIR
    / "player_observation_windows.csv"
)

EXCLUSION_FILE = (
    OUTPUT_DIR
    / "cohort_exclusions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "cohort_summary.txt"
)


# ============================================================
# Paper-inspired cohort settings
# ============================================================

TARGET_TEAM = "TeamB"

# For the initial paper-inspired DeepHit cohort,
# require both objective and subjective data.
REQUIRE_OBJECTIVE = True
REQUIRE_SUBJECTIVE = True

# Session JSON is informative, but for now we do not
# make it an additional exclusion rule because the paper-inspired
# feature table will be constructed later.
REQUIRE_SESSION = False


# ============================================================
# Helpers
# ============================================================

def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )


def to_bool(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def normalize_date(
    series: pd.Series,
) -> pd.Series:

    return (
        pd.to_datetime(
            series,
            errors="coerce",
        )
        .dt.normalize()
    )


# ============================================================
# Load player overlap audit
# ============================================================

def load_player_overlap() -> pd.DataFrame:

    require_file(
        PLAYER_OVERLAP_FILE
    )

    frame = pd.read_csv(
        PLAYER_OVERLAP_FILE
    )

    print(
        "Player-overlap columns:"
    )

    print(
        list(frame.columns)
    )

    required_columns = {
        "player_name",
        "team",
        "player_id",
        "in_objective",
        "in_subjective_wide",
        "in_session_json",
        "in_injury",
    }

    missing = (
        required_columns
        - set(frame.columns)
    )

    if missing:

        raise ValueError(
            "player_overlap_audit.csv is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    for column in [
        "in_objective",
        "in_subjective_wide",
        "in_session_json",
        "in_injury",
    ]:

        frame[column] = to_bool(
            frame[column]
        )

    return frame


# ============================================================
# Load objective observation windows
# ============================================================

def load_objective_windows() -> pd.DataFrame:

    require_file(
        OBJECTIVE_MANIFEST_FILE
    )

    frame = pd.read_csv(
        OBJECTIVE_MANIFEST_FILE
    )

    print()
    print(
        "Objective-manifest columns:"
    )

    print(
        list(frame.columns)
    )

    required_columns = {
        "team",
        "player_id",
        "date",
    }

    missing = (
        required_columns
        - set(frame.columns)
    )

    if missing:

        raise ValueError(
            "objective_file_manifest.csv is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    frame["date"] = normalize_date(
        frame["date"]
    )

    frame = frame.dropna(
        subset=[
            "player_id",
            "date",
        ]
    )

    # There can be multiple objective files on the same
    # player-date, as discovered by Audit 02.
    #
    # Therefore count:
    #   objective_files
    # and
    #   objective_active_days
    # separately.
    windows = (
        frame
        .groupby(
            [
                "team",
                "player_id",
            ],
            as_index=False,
        )
        .agg(
            objective_first_date=(
                "date",
                "min",
            ),
            objective_last_date=(
                "date",
                "max",
            ),
            objective_files=(
                "date",
                "size",
            ),
            objective_active_days=(
                "date",
                "nunique",
            ),
        )
    )

    windows[
        "objective_calendar_span_days"
    ] = (
        windows[
            "objective_last_date"
        ]
        - windows[
            "objective_first_date"
        ]
    ).dt.days + 1

    return windows


# ============================================================
# Load injury history
# ============================================================

def load_injury_summary() -> pd.DataFrame:

    require_file(
        INJURY_PLAYER_FILE
    )

    frame = pd.read_csv(
        INJURY_PLAYER_FILE
    )

    print()
    print(
        "Injury-summary columns:"
    )

    print(
        list(frame.columns)
    )

    required_columns = {
        "player_name",
        "team",
        "injury_records",
        "unique_injury_dates",
        "first_injury_date",
        "last_injury_date",
    }

    missing = (
        required_columns
        - set(frame.columns)
    )

    if missing:

        raise ValueError(
            "injury_player_summary.csv is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    frame[
        "first_injury_date"
    ] = normalize_date(
        frame[
            "first_injury_date"
        ]
    )

    frame[
        "last_injury_date"
    ] = normalize_date(
        frame[
            "last_injury_date"
        ]
    )

    return frame[
        [
            "player_name",
            "injury_records",
            "unique_injury_dates",
            "first_injury_date",
            "last_injury_date",
        ]
    ].copy()


# ============================================================
# Determine inclusion / exclusion
# ============================================================

def assign_cohort_status(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    result = frame.copy()

    result[
        "is_target_team"
    ] = (
        result[
            "team"
        ]
        == TARGET_TEAM
    )

    exclusion_reason = []

    include = []

    for _, row in result.iterrows():

        reasons = []

        if not row[
            "is_target_team"
        ]:

            reasons.append(
                "not_target_team"
            )

        if (
            REQUIRE_OBJECTIVE
            and not row[
                "in_objective"
            ]
        ):

            reasons.append(
                "missing_objective"
            )

        if (
            REQUIRE_SUBJECTIVE
            and not row[
                "in_subjective_wide"
            ]
        ):

            reasons.append(
                "missing_subjective"
            )

        if (
            REQUIRE_SESSION
            and not row[
                "in_session_json"
            ]
        ):

            reasons.append(
                "missing_session"
            )

        eligible = (
            len(reasons)
            == 0
        )

        include.append(
            eligible
        )

        if eligible:

            exclusion_reason.append(
                ""
            )

        else:

            exclusion_reason.append(
                ";".join(
                    reasons
                )
            )

    result[
        "include_in_deephit"
    ] = include

    result[
        "exclusion_reason"
    ] = exclusion_reason

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 80
    )

    print(
        "SoccerMon Paper-Inspired DeepHit Cohort Builder"
    )

    print(
        "=" * 80
    )

    print()

    # --------------------------------------------------------
    # Load audit outputs
    # --------------------------------------------------------

    overlap = load_player_overlap()

    objective = load_objective_windows()

    injuries = load_injury_summary()


    # --------------------------------------------------------
    # Merge objective window information
    # --------------------------------------------------------

    cohort = overlap.merge(
        objective,
        on=[
            "team",
            "player_id",
        ],
        how="left",
        validate="one_to_one",
    )


    # --------------------------------------------------------
    # Merge injury information
    # --------------------------------------------------------

    cohort = cohort.merge(
        injuries,
        on="player_name",
        how="left",
        validate="one_to_one",
    )


    # --------------------------------------------------------
    # Fill players with no public injury history
    # --------------------------------------------------------

    cohort[
        "injury_records"
    ] = (
        cohort[
            "injury_records"
        ]
        .fillna(0)
        .astype(int)
    )

    cohort[
        "unique_injury_dates"
    ] = (
        cohort[
            "unique_injury_dates"
        ]
        .fillna(0)
        .astype(int)
    )

    cohort[
        "has_public_injury_history"
    ] = (
        cohort[
            "injury_records"
        ]
        > 0
    )


    # --------------------------------------------------------
    # Cohort inclusion
    # --------------------------------------------------------

    cohort = assign_cohort_status(
        cohort
    )


    # --------------------------------------------------------
    # Basic objective history eligibility
    #
    # This does NOT yet mean that a player has a valid
    # 21-day DeepHit window.
    #
    # It simply means the objective observation span is long
    # enough that such windows could potentially exist.
    # --------------------------------------------------------

    cohort[
        "objective_span_at_least_21d"
    ] = (
        cohort[
            "objective_calendar_span_days"
        ]
        .fillna(0)
        >= 21
    )


    # --------------------------------------------------------
    # Identify injury history relative to objective monitoring
    # --------------------------------------------------------

    cohort[
        "injury_history_before_objective"
    ] = (
        cohort[
            "has_public_injury_history"
        ]
        & cohort[
            "first_injury_date"
        ].notna()
        & cohort[
            "objective_first_date"
        ].notna()
        & (
            cohort[
                "first_injury_date"
            ]
            < cohort[
                "objective_first_date"
            ]
        )
    )


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    cohort = cohort.sort_values(
        [
            "include_in_deephit",
            "team",
            "player_name",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Save player cohort
    # --------------------------------------------------------

    cohort.to_csv(
        COHORT_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # Observation-window table
    #
    # For now this contains objective coverage only.
    #
    # Subjective first/last valid dates will be added in
    # the player-day construction phase, where we inspect
    # actual subjective observations rather than guessing
    # from calendar missingness.
    # --------------------------------------------------------

    observation_windows = cohort[
        [
            "player_name",
            "team",
            "player_id",
            "objective_first_date",
            "objective_last_date",
            "objective_calendar_span_days",
            "objective_files",
            "objective_active_days",
            "objective_span_at_least_21d",
            "first_injury_date",
            "last_injury_date",
            "injury_records",
            "unique_injury_dates",
            "injury_history_before_objective",
            "include_in_deephit",
        ]
    ].copy()

    observation_windows.to_csv(
        OBSERVATION_WINDOW_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # Exclusions table
    # --------------------------------------------------------

    exclusions = cohort[
        ~cohort[
            "include_in_deephit"
        ]
    ][
        [
            "player_name",
            "team",
            "in_objective",
            "in_subjective_wide",
            "in_session_json",
            "in_injury",
            "exclusion_reason",
        ]
    ].copy()

    exclusions.to_csv(
        EXCLUSION_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # Summary calculations
    # --------------------------------------------------------

    total_players = len(
        cohort
    )

    target_team_players = int(
        (
            cohort[
                "team"
            ]
            == TARGET_TEAM
        ).sum()
    )

    eligible = cohort[
        cohort[
            "include_in_deephit"
        ]
    ].copy()

    eligible_players = len(
        eligible
    )

    eligible_with_injury = int(
        eligible[
            "has_public_injury_history"
        ].sum()
    )

    eligible_without_injury = (
        eligible_players
        - eligible_with_injury
    )

    eligible_injury_records = int(
        eligible[
            "injury_records"
        ].sum()
    )

    eligible_21d_span = int(
        eligible[
            "objective_span_at_least_21d"
        ].sum()
    )

    eligible_prior_injury = int(
        eligible[
            "injury_history_before_objective"
        ].sum()
    )


    # --------------------------------------------------------
    # Exclusion reason counts
    # --------------------------------------------------------

    reason_counts = (
        exclusions[
            "exclusion_reason"
        ]
        .value_counts()
        .sort_index()
    )


    # --------------------------------------------------------
    # Text summary
    # --------------------------------------------------------

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Paper-Inspired DeepHit Cohort Summary"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "Cohort definition"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Target team: {TARGET_TEAM}"
    )

    lines.append(
        f"Require objective data: "
        f"{REQUIRE_OBJECTIVE}"
    )

    lines.append(
        f"Require subjective data: "
        f"{REQUIRE_SUBJECTIVE}"
    )

    lines.append(
        f"Require session JSON: "
        f"{REQUIRE_SESSION}"
    )

    lines.append("")

    lines.append(
        "Player counts"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Players across Audit 14: "
        f"{total_players:,}"
    )

    lines.append(
        f"{TARGET_TEAM} players: "
        f"{target_team_players:,}"
    )

    lines.append(
        f"Initial DeepHit cohort: "
        f"{eligible_players:,}"
    )

    lines.append(
        f"Cohort players with public injury history: "
        f"{eligible_with_injury:,}"
    )

    lines.append(
        f"Cohort players without public injury history: "
        f"{eligible_without_injury:,}"
    )

    lines.append(
        f"Public injury records among cohort players: "
        f"{eligible_injury_records:,}"
    )

    lines.append("")

    lines.append(
        "Objective coverage"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Cohort players with >=21-day objective span: "
        f"{eligible_21d_span:,}"
    )

    lines.append(
        f"Cohort players with injury history "
        f"before objective monitoring: "
        f"{eligible_prior_injury:,}"
    )

    lines.append("")

    lines.append(
        "Important interpretation"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "Inclusion in this cohort does not yet imply that a player "
        "has a valid 21-day DeepHit history window or adequate "
        "7-day future follow-up."
    )

    lines.append(
        "Date-level eligibility will be determined after the canonical "
        "player-day dataset is constructed."
    )

    lines.append(
        "Players without public injury records are intentionally retained "
        "because censored/non-event observations are required for survival "
        "analysis."
    )

    lines.append("")

    lines.append(
        "Exclusions"
    )

    lines.append(
        "-" * 80
    )

    if reason_counts.empty:

        lines.append(
            "No players excluded."
        )

    else:

        for reason, count in (
            reason_counts.items()
        ):

            lines.append(
                f"{reason}: {count:,}"
            )


    SUMMARY_FILE.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print()

    print(
        "\n".join(lines)
    )

    print()

    print(
        f"Player cohort written to:\n"
        f"{COHORT_FILE}"
    )

    print()

    print(
        f"Observation windows written to:\n"
        f"{OBSERVATION_WINDOW_FILE}"
    )

    print()

    print(
        f"Exclusions written to:\n"
        f"{EXCLUSION_FILE}"
    )

    print()

    print(
        f"Summary written to:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":

    main()