from pathlib import Path
import json

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SESSION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "subjective"
    / "subjective"
    / "training-load"
    / "session.json"
)

COHORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "paper_inspired"
    / "cohort"
    / "player_cohort.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "paper_inspired"
    / "training_load"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DAILY_FILE = (
    OUTPUT_DIR
    / "training_load_daily.csv"
)

SESSION_FILE_OUT = (
    OUTPUT_DIR
    / "training_sessions_cleaned.csv"
)

COVERAGE_FILE = (
    OUTPUT_DIR
    / "training_load_coverage.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "training_load_daily_summary.txt"
)


# ============================================================
# Helpers
# ============================================================

def require_file(path: Path) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def load_selected_cohort() -> pd.DataFrame:

    require_file(
        COHORT_FILE
    )

    cohort = pd.read_csv(
        COHORT_FILE
    )

    include = (
        cohort["include_in_deephit"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )

    selected = (
        cohort.loc[
            include,
            [
                "player_name",
                "team",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            "player_name"
        )
        .reset_index(
            drop=True
        )
    )

    if selected.empty:

        raise ValueError(
            "No players marked include_in_deephit=True."
        )

    return selected


# ============================================================
# Load and normalize session.json
# ============================================================

def load_sessions() -> pd.DataFrame:

    require_file(
        SESSION_FILE
    )

    with SESSION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            "Expected session.json top level to be a dictionary."
        )

    rows = []

    for player_name, sessions in data.items():

        if not isinstance(
            sessions,
            list,
        ):

            continue

        for session in sessions:

            if not isinstance(
                session,
                dict,
            ):

                continue

            row = session.copy()

            row[
                "player_name"
            ] = player_name

            rows.append(
                row
            )

    frame = pd.DataFrame(
        rows
    )

    required_columns = {
        "player_name",
        "date",
        "rpe",
        "duration",
        "srpe",
    }

    missing = (
        required_columns
        - set(frame.columns)
    )

    if missing:

        raise ValueError(
            "Normalized session data missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    return frame


# ============================================================
# Clean session records
# ============================================================

def clean_sessions(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    result = frame.copy()

    # --------------------------------------------------------
    # Parse dates.
    #
    # session.json was already audited separately, but we use
    # day-first parsing explicitly because SoccerMon subjective
    # dates use European day/month ordering.
    # --------------------------------------------------------

    result[
        "date"
    ] = pd.to_datetime(
        result[
            "date"
        ],
        dayfirst=True,
        errors="coerce",
    ).dt.normalize()

    for column in [
        "rpe",
        "duration",
        "srpe",
    ]:

        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )

    invalid_dates = int(
        result[
            "date"
        ]
        .isna()
        .sum()
    )

    if invalid_dates:

        print(
            f"WARNING: removing "
            f"{invalid_dates:,} session records "
            f"with invalid dates."
        )

        result = result.dropna(
            subset=[
                "date"
            ]
        )

    return (
        result
        .sort_values(
            [
                "player_name",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Keep selected cohort
# ============================================================

def restrict_to_cohort(
    sessions: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:

    selected_players = set(
        cohort[
            "player_name"
        ]
    )

    result = sessions[
        sessions[
            "player_name"
        ].isin(
            selected_players
        )
    ].copy()

    result = result.merge(
        cohort,
        on="player_name",
        how="left",
        validate="many_to_one",
    )

    return result


# ============================================================
# Validate sRPE
#
# We already audited this relationship previously.
# Here we retain a lightweight preprocessing check so that
# model-building does not silently consume inconsistent data.
# ============================================================

def validate_srpe(
    sessions: pd.DataFrame,
) -> pd.DataFrame:

    result = sessions.copy()

    result[
        "srpe_reconstructed"
    ] = (
        result[
            "rpe"
        ]
        * result[
            "duration"
        ]
    )

    result[
        "srpe_difference"
    ] = (
        result[
            "srpe"
        ]
        - result[
            "srpe_reconstructed"
        ]
    )

    result[
        "srpe_matches"
    ] = (
        result[
            "srpe_difference"
        ]
        .abs()
        < 1e-9
    )

    return result


# ============================================================
# Aggregate sessions to player-day
# ============================================================

def build_daily_table(
    sessions: pd.DataFrame,
) -> pd.DataFrame:

    daily = (
        sessions
        .groupby(
            [
                "player_name",
                "team",
                "date",
            ],
            as_index=False,
        )
        .agg(
            session_count=(
                "srpe",
                "size",
            ),

            rpe_mean=(
                "rpe",
                "mean",
            ),

            rpe_max=(
                "rpe",
                "max",
            ),

            duration_total=(
                "duration",
                "sum",
            ),

            duration_mean=(
                "duration",
                "mean",
            ),

            srpe_total=(
                "srpe",
                "sum",
            ),

            srpe_mean=(
                "srpe",
                "mean",
            ),

            srpe_max=(
                "srpe",
                "max",
            ),
        )
        .sort_values(
            [
                "player_name",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return daily


# ============================================================
# Build coverage summary
# ============================================================

def build_coverage(
    daily: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, player in (
        cohort.iterrows()
    ):

        player_name = (
            player[
                "player_name"
            ]
        )

        team = (
            player[
                "team"
            ]
        )

        player_daily = daily[
            daily[
                "player_name"
            ]
            == player_name
        ]

        if player_daily.empty:

            rows.append(
                {
                    "player_name":
                    player_name,

                    "team":
                    team,

                    "first_session_date":
                    pd.NaT,

                    "last_session_date":
                    pd.NaT,

                    "session_days":
                    0,

                    "session_records":
                    0,

                    "multi_session_days":
                    0,
                }
            )

            continue

        rows.append(
            {
                "player_name":
                player_name,

                "team":
                team,

                "first_session_date":
                player_daily[
                    "date"
                ].min(),

                "last_session_date":
                player_daily[
                    "date"
                ].max(),

                "session_days":
                len(
                    player_daily
                ),

                "session_records":
                int(
                    player_daily[
                        "session_count"
                    ].sum()
                ),

                "multi_session_days":
                int(
                    (
                        player_daily[
                            "session_count"
                        ]
                        > 1
                    ).sum()
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "player_name"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 80
    )

    print(
        "SoccerMon Paper-Inspired Training-Load Daily Builder"
    )

    print(
        "=" * 80
    )

    print()


    # --------------------------------------------------------
    # Cohort
    # --------------------------------------------------------

    cohort = (
        load_selected_cohort()
    )

    print(
        f"Selected cohort players: "
        f"{len(cohort):,}"
    )


    # --------------------------------------------------------
    # Sessions
    # --------------------------------------------------------

    sessions = (
        load_sessions()
    )

    print(
        f"All session records loaded: "
        f"{len(sessions):,}"
    )

    sessions = (
        clean_sessions(
            sessions
        )
    )

    sessions = (
        restrict_to_cohort(
            sessions,
            cohort,
        )
    )

    print(
        f"Cohort session records: "
        f"{len(sessions):,}"
    )


    # --------------------------------------------------------
    # Validate sRPE
    # --------------------------------------------------------

    sessions = (
        validate_srpe(
            sessions
        )
    )

    srpe_matches = int(
        sessions[
            "srpe_matches"
        ].sum()
    )

    srpe_mismatches = (
        len(sessions)
        - srpe_matches
    )


    # --------------------------------------------------------
    # Daily aggregation
    # --------------------------------------------------------

    daily = (
        build_daily_table(
            sessions
        )
    )

    coverage = (
        build_coverage(
            daily,
            cohort,
        )
    )


    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    duplicate_days = int(
        daily.duplicated(
            subset=[
                "player_name",
                "date",
            ]
        ).sum()
    )

    if duplicate_days:

        raise ValueError(
            "Daily table contains duplicate "
            "player-date rows."
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    sessions.to_csv(
        SESSION_FILE_OUT,
        index=False,
    )

    daily.to_csv(
        DAILY_FILE,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    players_with_sessions = (
        daily[
            "player_name"
        ]
        .nunique()
    )

    total_session_days = len(
        daily
    )

    total_sessions = int(
        daily[
            "session_count"
        ].sum()
    )

    multi_session_days = int(
        (
            daily[
                "session_count"
            ]
            > 1
        ).sum()
    )

    maximum_sessions_day = int(
        daily[
            "session_count"
        ].max()
    )

    first_date = (
        daily[
            "date"
        ].min()
    )

    last_date = (
        daily[
            "date"
        ].max()
    )


    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Paper-Inspired Training-Load Daily Summary"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "Cohort"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Selected cohort players: "
        f"{len(cohort):,}"
    )

    lines.append(
        f"Players with session records: "
        f"{players_with_sessions:,}"
    )

    lines.append("")

    lines.append(
        "Sessions"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Session records: "
        f"{total_sessions:,}"
    )

    lines.append(
        f"Player-days with sessions: "
        f"{total_session_days:,}"
    )

    lines.append(
        f"Player-days with multiple sessions: "
        f"{multi_session_days:,}"
    )

    lines.append(
        f"Maximum sessions on one player-day: "
        f"{maximum_sessions_day:,}"
    )

    lines.append(
        f"Date range: "
        f"{first_date.date()} "
        f"to "
        f"{last_date.date()}"
    )

    lines.append("")

    lines.append(
        "sRPE validation"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"sRPE records matching RPE * duration: "
        f"{srpe_matches:,}"
    )

    lines.append(
        f"sRPE records not matching RPE * duration: "
        f"{srpe_mismatches:,}"
    )

    lines.append("")

    lines.append(
        "Important interpretation"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "Multiple sessions occurring on the same player-day "
        "are preserved in the cleaned session table and aggregated "
        "into one daily record."
    )

    lines.append(
        "Daily sRPE is represented here as the sum of session-level "
        "sRPE values for that player-day."
    )

    lines.append(
        "No ATL, CTL, ACWR, monotony, strain, rolling workload, "
        "imputation, scaling, or survival targets are created here."
    )

    lines.append(
        "Those transformations belong to later feature-engineering "
        "and modeling phases."
    )


    SUMMARY_FILE.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


    print()

    print(
        "\n".join(
            lines
        )
    )

    print()

    print(
        f"Cleaned sessions written to:\n"
        f"{SESSION_FILE_OUT}"
    )

    print()

    print(
        f"Daily training-load table written to:\n"
        f"{DAILY_FILE}"
    )

    print()

    print(
        f"Coverage table written to:\n"
        f"{COVERAGE_FILE}"
    )

    print()

    print(
        f"Summary written to:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":

    main()