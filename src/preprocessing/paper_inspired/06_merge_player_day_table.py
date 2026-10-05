from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "paper_inspired"
)

COHORT_FILE = (
    BASE_DIR
    / "cohort"
    / "player_cohort.csv"
)

SUBJECTIVE_FILE = (
    BASE_DIR
    / "subjective"
    / "subjective_daily_unimputed.csv"
)

TRAINING_FILE = (
    BASE_DIR
    / "training_load"
    / "training_load_daily.csv"
)

OBJECTIVE_FILE = (
    BASE_DIR
    / "objective"
    / "objective_daily.csv"
)

INJURY_FILE = (
    BASE_DIR
    / "injury"
    / "injury_daily.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "player_day"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLAYER_DAY_FILE = (
    OUTPUT_DIR
    / "player_day_unimputed.csv"
)

COVERAGE_FILE = (
    OUTPUT_DIR
    / "player_day_coverage.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "player_day_summary.txt"
)


# ============================================================
# Helpers
# ============================================================

def require_file(path: Path) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def parse_bool(series: pd.Series) -> pd.Series:

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
            }
        )
        .fillna(False)
        .astype(bool)
    )


def normalize_date_column(
    frame: pd.DataFrame,
    column: str = "date",
) -> pd.DataFrame:

    result = frame.copy()

    result[column] = pd.to_datetime(
        result[column],
        errors="coerce",
    ).dt.normalize()

    return result


# ============================================================
# Cohort
# ============================================================

def load_cohort() -> pd.DataFrame:

    require_file(
        COHORT_FILE
    )

    cohort = pd.read_csv(
        COHORT_FILE
    )

    required = {
        "player_name",
        "team",
        "include_in_deephit",
    }

    missing = (
        required
        - set(cohort.columns)
    )

    if missing:

        raise ValueError(
            "Cohort file missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    include = parse_bool(
        cohort["include_in_deephit"]
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

    return selected


# ============================================================
# Load modality files
# ============================================================

def load_subjective() -> pd.DataFrame:

    require_file(
        SUBJECTIVE_FILE
    )

    frame = pd.read_csv(
        SUBJECTIVE_FILE
    )

    frame = normalize_date_column(
        frame
    )

    return frame


def load_training() -> pd.DataFrame:

    require_file(
        TRAINING_FILE
    )

    frame = pd.read_csv(
        TRAINING_FILE
    )

    frame = normalize_date_column(
        frame
    )

    return frame


def load_objective() -> pd.DataFrame:

    require_file(
        OBJECTIVE_FILE
    )

    frame = pd.read_csv(
        OBJECTIVE_FILE
    )

    frame = normalize_date_column(
        frame
    )

    return frame


def load_injury() -> pd.DataFrame:

    require_file(
        INJURY_FILE
    )

    frame = pd.read_csv(
        INJURY_FILE
    )

    frame = normalize_date_column(
        frame
    )

    return frame


# ============================================================
# Determine per-player calendar bounds
#
# We intentionally use the minimum and maximum dates across
# all available modalities for each selected player.
#
# This creates a full calendar while preserving structural
# absence rather than silently dropping missing days.
# ============================================================

def build_player_calendar(
    cohort: pd.DataFrame,
    subjective: pd.DataFrame,
    training: pd.DataFrame,
    objective: pd.DataFrame,
    injury: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    modality_frames = [
        subjective,
        training,
        objective,
        injury,
    ]

    for _, player in cohort.iterrows():

        player_name = player["player_name"]
        team = player["team"]

        dates = []

        for frame in modality_frames:

            subset = frame[
                frame["player_name"]
                == player_name
            ]

            if not subset.empty:

                valid_dates = (
                    subset["date"]
                    .dropna()
                )

                if not valid_dates.empty:

                    dates.extend(
                        valid_dates.tolist()
                    )

        if not dates:

            continue

        first_date = min(dates)
        last_date = max(dates)

        player_dates = pd.date_range(
            first_date,
            last_date,
            freq="D",
        )

        rows.append(
            pd.DataFrame(
                {
                    "player_name":
                        player_name,

                    "team":
                        team,

                    "date":
                        player_dates,
                }
            )
        )

    if not rows:

        raise ValueError(
            "No calendar rows could be created."
        )

    calendar = (
        pd.concat(
            rows,
            ignore_index=True,
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

    return calendar


# ============================================================
# Prepare subjective modality
# ============================================================

def prepare_subjective(
    subjective: pd.DataFrame,
) -> pd.DataFrame:

    result = subjective.copy()

    result[
        "has_subjective"
    ] = (
        result[
            "any_wellness_present"
        ]
        .fillna(False)
        .astype(bool)
    )

    return result


# ============================================================
# Prepare training modality
# ============================================================

def prepare_training(
    training: pd.DataFrame,
) -> pd.DataFrame:

    result = training.copy()

    result[
        "has_training"
    ] = True

    return result


# ============================================================
# Prepare objective modality
# ============================================================

def prepare_objective(
    objective: pd.DataFrame,
) -> pd.DataFrame:

    result = objective.copy()

    result[
        "has_objective"
    ] = True

    return result


# ============================================================
# Prepare injury modality
# ============================================================

def prepare_injury(
    injury: pd.DataFrame,
) -> pd.DataFrame:

    result = injury.copy()

    result[
        "has_injury"
    ] = True

    return result


# ============================================================
# Merge all sources
# ============================================================

def merge_player_day(
    calendar: pd.DataFrame,
    subjective: pd.DataFrame,
    training: pd.DataFrame,
    objective: pd.DataFrame,
    injury: pd.DataFrame,
) -> pd.DataFrame:

    result = calendar.copy()

    # --------------------------------------------------------
    # Subjective
    # --------------------------------------------------------

    subjective_drop = [
        column
        for column in [
            "team",
        ]
        if column in subjective.columns
    ]

    result = result.merge(
        subjective.drop(
            columns=subjective_drop
        ),
        on=[
            "player_name",
            "date",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    training_drop = [
        column
        for column in [
            "team",
        ]
        if column in training.columns
    ]

    result = result.merge(
        training.drop(
            columns=training_drop
        ),
        on=[
            "player_name",
            "date",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    objective_drop = [
        column
        for column in [
            "team",
        ]
        if column in objective.columns
    ]

    result = result.merge(
        objective.drop(
            columns=objective_drop
        ),
        on=[
            "player_name",
            "date",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Injury
    # --------------------------------------------------------

    injury_drop = [
        column
        for column in [
            "team",
        ]
        if column in injury.columns
    ]

    result = result.merge(
        injury.drop(
            columns=injury_drop
        ),
        on=[
            "player_name",
            "date",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Presence indicators
    # --------------------------------------------------------

    for column in [
        "has_subjective",
        "has_training",
        "has_objective",
        "has_injury",
    ]:

        if column not in result.columns:
            result[column] = False

        result[column] = (
            result[column]
            .fillna(False)
            .astype(bool)
        )

    # --------------------------------------------------------
    # Injury event defaults
    #
    # No injury record on a player-day means event = 0.
    # --------------------------------------------------------

    if "injury_event" not in result.columns:
        result["injury_event"] = 0

    result[
        "injury_event"
    ] = (
        result[
            "injury_event"
        ]
        .fillna(0)
        .astype(int)
    )

    for column in [
        "injury_episode_count",
        "minor_episode_count",
        "major_episode_count",
        "unique_body_regions",
    ]:

        if column in result.columns:

            result[column] = (
                result[column]
                .fillna(0)
                .astype(int)
            )

    if "body_regions" in result.columns:

        result[
            "body_regions"
        ] = (
            result[
                "body_regions"
            ]
            .fillna("")
        )

    # --------------------------------------------------------
    # Combined presence indicators
    # --------------------------------------------------------

    result[
        "has_any_model_data"
    ] = (
        result[
            [
                "has_subjective",
                "has_training",
                "has_objective",
            ]
        ]
        .any(
            axis=1
        )
    )

    result[
        "has_all_three_modalities"
    ] = (
        result[
            [
                "has_subjective",
                "has_training",
                "has_objective",
            ]
        ]
        .all(
            axis=1
        )
    )

    result[
        "available_modality_count"
    ] = (
        result[
            [
                "has_subjective",
                "has_training",
                "has_objective",
            ]
        ]
        .sum(
            axis=1
        )
        .astype(int)
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
# Build coverage summary
# ============================================================

def build_coverage(
    player_day: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        player_name,
        frame,
    ) in player_day.groupby(
        "player_name",
        sort=True,
    ):

        frame = (
            frame
            .sort_values(
                "date"
            )
            .copy()
        )

        total_days = len(
            frame
        )

        subjective_days = int(
            frame[
                "has_subjective"
            ].sum()
        )

        training_days = int(
            frame[
                "has_training"
            ].sum()
        )

        objective_days = int(
            frame[
                "has_objective"
            ].sum()
        )

        all_three_days = int(
            frame[
                "has_all_three_modalities"
            ].sum()
        )

        any_data_days = int(
            frame[
                "has_any_model_data"
            ].sum()
        )

        injury_days = int(
            frame[
                "injury_event"
            ].sum()
        )

        rows.append(
            {
                "player_name":
                    player_name,

                "team":
                    frame[
                        "team"
                    ].iloc[0],

                "calendar_start":
                    frame[
                        "date"
                    ].min(),

                "calendar_end":
                    frame[
                        "date"
                    ].max(),

                "calendar_days":
                    total_days,

                "subjective_days":
                    subjective_days,

                "training_days":
                    training_days,

                "objective_days":
                    objective_days,

                "all_three_days":
                    all_three_days,

                "any_data_days":
                    any_data_days,

                "injury_event_days":
                    injury_days,

                "subjective_coverage":
                    (
                        subjective_days
                        / total_days
                        if total_days
                        else np.nan
                    ),

                "training_coverage":
                    (
                        training_days
                        / total_days
                        if total_days
                        else np.nan
                    ),

                "objective_coverage":
                    (
                        objective_days
                        / total_days
                        if total_days
                        else np.nan
                    ),

                "all_three_coverage":
                    (
                        all_three_days
                        / total_days
                        if total_days
                        else np.nan
                    ),

                "any_data_coverage":
                    (
                        any_data_days
                        / total_days
                        if total_days
                        else np.nan
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
        "SoccerMon Paper-Inspired Player-Day Merger"
    )

    print(
        "=" * 80
    )

    print()

    cohort = load_cohort()

    subjective = prepare_subjective(
        load_subjective()
    )

    training = prepare_training(
        load_training()
    )

    objective = prepare_objective(
        load_objective()
    )

    injury = prepare_injury(
        load_injury()
    )

    print(
        f"Selected cohort players: "
        f"{len(cohort):,}"
    )

    print(
        f"Subjective rows: "
        f"{len(subjective):,}"
    )

    print(
        f"Training rows: "
        f"{len(training):,}"
    )

    print(
        f"Objective rows: "
        f"{len(objective):,}"
    )

    print(
        f"Injury-event rows: "
        f"{len(injury):,}"
    )

    print()

    calendar = build_player_calendar(
        cohort,
        subjective,
        training,
        objective,
        injury,
    )

    print(
        f"Calendar rows created: "
        f"{len(calendar):,}"
    )

    player_day = merge_player_day(
        calendar,
        subjective,
        training,
        objective,
        injury,
    )

    coverage = build_coverage(
        player_day
    )

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    duplicate_rows = int(
        player_day.duplicated(
            subset=[
                "player_name",
                "date",
            ]
        ).sum()
    )

    if duplicate_rows:

        raise ValueError(
            f"Duplicate player-date rows found: "
            f"{duplicate_rows:,}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    player_day.to_csv(
        PLAYER_DAY_FILE,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_rows = len(
        player_day
    )

    subjective_days = int(
        player_day[
            "has_subjective"
        ].sum()
    )

    training_days = int(
        player_day[
            "has_training"
        ].sum()
    )

    objective_days = int(
        player_day[
            "has_objective"
        ].sum()
    )

    all_three_days = int(
        player_day[
            "has_all_three_modalities"
        ].sum()
    )

    any_data_days = int(
        player_day[
            "has_any_model_data"
        ].sum()
    )

    injury_days = int(
        player_day[
            "injury_event"
        ].sum()
    )

    zero_modality_days = int(
        (
            player_day[
                "available_modality_count"
            ]
            == 0
        ).sum()
    )

    one_modality_days = int(
        (
            player_day[
                "available_modality_count"
            ]
            == 1
        ).sum()
    )

    two_modality_days = int(
        (
            player_day[
                "available_modality_count"
            ]
            == 2
        ).sum()
    )

    three_modality_days = int(
        (
            player_day[
                "available_modality_count"
            ]
            == 3
        ).sum()
    )

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Paper-Inspired Player-Day Summary"
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
        f"Selected players: "
        f"{len(cohort):,}"
    )

    lines.append(
        f"Player-day rows: "
        f"{total_rows:,}"
    )

    lines.append("")

    lines.append(
        "Modality availability"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Days with subjective data: "
        f"{subjective_days:,}"
    )

    lines.append(
        f"Days with training data: "
        f"{training_days:,}"
    )

    lines.append(
        f"Days with objective data: "
        f"{objective_days:,}"
    )

    lines.append(
        f"Days with all three modalities: "
        f"{all_three_days:,}"
    )

    lines.append(
        f"Days with any model data: "
        f"{any_data_days:,}"
    )

    lines.append("")

    lines.append(
        "Available modality count"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"0 modalities: "
        f"{zero_modality_days:,}"
    )

    lines.append(
        f"1 modality: "
        f"{one_modality_days:,}"
    )

    lines.append(
        f"2 modalities: "
        f"{two_modality_days:,}"
    )

    lines.append(
        f"3 modalities: "
        f"{three_modality_days:,}"
    )

    lines.append("")

    lines.append(
        "Injury events"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Injury-event days: "
        f"{injury_days:,}"
    )

    lines.append("")

    lines.append(
        "Interpretation"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "The merged table preserves missing modality values "
        "without imputation."
    )

    lines.append(
        "Absence of a training record is not automatically "
        "interpreted as zero training load."
    )

    lines.append(
        "Absence of an objective record is not automatically "
        "interpreted as zero objective exposure."
    )

    lines.append(
        "Injury-event days are retained even when one or more "
        "predictor modalities are absent."
    )

    lines.append(
        "The next audit will determine which player-days can "
        "support valid temporal histories for the DeepHit model."
    )

    SUMMARY_FILE.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(lines)
    )

    print()

    print(
        f"Player-day table written to:\n"
        f"{PLAYER_DAY_FILE}"
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