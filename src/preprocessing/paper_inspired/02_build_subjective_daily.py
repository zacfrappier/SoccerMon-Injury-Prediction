from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

WELLNESS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "subjective"
    / "subjective"
    / "wellness"
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
    / "subjective"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Output files
# ============================================================

DAILY_FILE = (
    OUTPUT_DIR
    / "subjective_daily_unimputed.csv"
)

COVERAGE_FILE = (
    OUTPUT_DIR
    / "subjective_daily_coverage.csv"
)

FEATURE_MISSINGNESS_FILE = (
    OUTPUT_DIR
    / "subjective_feature_missingness.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "subjective_daily_summary.txt"
)


# ============================================================
# Wellness features
#
# All seven public SoccerMon wellness variables are preserved
# here.
#
# The paper-inspired model may later omit sleep_quality if we
# continue following the published paper's listed feature set.
#
# We keep it now because this stage represents source data,
# not final model feature selection.
# ============================================================

WELLNESS_FEATURES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_duration",
    "sleep_quality",
    "soreness",
    "stress",
]


# Features listed for the paper-inspired DeepHit model.
#
# This is NOT used to remove data in this script.
# It is included for reporting purposes only.
MODEL_WELLNESS_FEATURES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_duration",
    "soreness",
    "stress",
]


# ============================================================
# Helpers
# ============================================================

def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def parse_soccer_mon_dates(
    series: pd.Series,
) -> pd.Series:
    """
    SoccerMon subjective wide-table dates use DD.MM.YYYY.

    Explicit parsing is important because automatic parsing
    previously produced incorrect date interpretations.
    """

    return pd.to_datetime(
        series,
        format="%d.%m.%Y",
        errors="coerce",
    ).dt.normalize()


# ============================================================
# Load the 22-player paper-inspired cohort
# ============================================================

def load_selected_cohort() -> pd.DataFrame:

    require_file(
        COHORT_FILE
    )

    cohort = pd.read_csv(
        COHORT_FILE
    )

    required_columns = {
        "player_name",
        "team",
        "include_in_deephit",
    }

    missing = (
        required_columns
        - set(cohort.columns)
    )

    if missing:

        raise ValueError(
            "Cohort file is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # CSV booleans may load as actual bools or strings,
    # depending on how the file was written/read.
    include = (
        cohort[
            "include_in_deephit"
        ]
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
        cohort[
            include
        ][
            [
                "player_name",
                "team",
            ]
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
            "No players are marked include_in_deephit=True."
        )

    return selected


# ============================================================
# Load one wide wellness table
# ============================================================

def load_wellness_feature(
    feature: str,
    selected_players: set[str],
) -> pd.DataFrame:
    """
    Convert one SoccerMon wellness table from:

        Date | Player A | Player B | Player C | ...

    to:

        player_name | date | feature_value
    """

    path = (
        WELLNESS_DIR
        / f"{feature}.csv"
    )

    require_file(
        path
    )

    frame = pd.read_csv(
        path
    )

    if frame.empty:

        raise ValueError(
            f"{feature}.csv is empty."
        )

    # --------------------------------------------------------
    # SoccerMon wide tables use the first column as calendar
    # date. Usually it is named "Date".
    #
    # We intentionally use the first column rather than relying
    # solely on its name because previous audits found that some
    # wide-table date columns were not detected automatically.
    # --------------------------------------------------------

    date_column = (
        frame.columns[0]
    )

    frame[
        date_column
    ] = parse_soccer_mon_dates(
        frame[
            date_column
        ]
    )

    invalid_dates = int(
        frame[
            date_column
        ]
        .isna()
        .sum()
    )

    if invalid_dates:

        raise ValueError(
            f"{feature}.csv contains "
            f"{invalid_dates:,} unparseable dates."
        )

    # --------------------------------------------------------
    # Identify the selected cohort columns that actually occur
    # in this wellness table.
    # --------------------------------------------------------

    available_players = [
        column
        for column
        in frame.columns[1:]
        if column in selected_players
    ]

    missing_cohort_players = (
        selected_players
        - set(
            available_players
        )
    )

    if missing_cohort_players:

        print()
        print(
            f"WARNING: {feature}.csv is missing "
            f"{len(missing_cohort_players)} selected cohort players."
        )

        for player in sorted(
            missing_cohort_players
        ):

            print(
                f"  {player}"
            )

    # --------------------------------------------------------
    # Wide -> long
    # --------------------------------------------------------

    long_frame = (
        frame[
            [
                date_column,
                *available_players,
            ]
        ]
        .melt(
            id_vars=[
                date_column
            ],
            var_name="player_name",
            value_name=feature,
        )
        .rename(
            columns={
                date_column:
                "date"
            }
        )
    )

    # --------------------------------------------------------
    # Convert feature values to numeric.
    #
    # Anything non-numeric becomes NaN. We do NOT fill NaNs.
    # --------------------------------------------------------

    long_frame[
        feature
    ] = pd.to_numeric(
        long_frame[
            feature
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Sanity check:
    # one row per player-date for this feature.
    # --------------------------------------------------------

    duplicate_count = int(
        long_frame.duplicated(
            subset=[
                "player_name",
                "date",
            ]
        ).sum()
    )

    if duplicate_count:

        raise ValueError(
            f"{feature}.csv produced "
            f"{duplicate_count:,} duplicate player-date rows."
        )

    return long_frame


# ============================================================
# Merge all wellness features
# ============================================================

def build_subjective_daily(
    selected: pd.DataFrame,
) -> pd.DataFrame:

    selected_players = set(
        selected[
            "player_name"
        ]
    )

    daily = None

    for feature in (
        WELLNESS_FEATURES
    ):

        print(
            f"Loading wellness feature: "
            f"{feature}"
        )

        feature_frame = (
            load_wellness_feature(
                feature,
                selected_players,
            )
        )

        if daily is None:

            daily = (
                feature_frame
            )

        else:

            daily = daily.merge(
                feature_frame,
                on=[
                    "player_name",
                    "date",
                ],
                how="outer",
                validate="one_to_one",
            )

    if daily is None:

        raise RuntimeError(
            "No wellness data were loaded."
        )

    # Add team from cohort definition.
    daily = daily.merge(
        selected,
        on="player_name",
        how="left",
        validate="many_to_one",
    )

    daily = daily[
        [
            "player_name",
            "team",
            "date",
            *WELLNESS_FEATURES,
        ]
    ]

    daily = daily.sort_values(
        [
            "player_name",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    return daily


# ============================================================
# Determine subjective active periods
#
# IMPORTANT:
#
# The full wellness tables span a global calendar.
#
# A player may have NaN values before they joined data
# collection or after they left.
#
# We therefore define the initial subjective active period as:
#
#   first date with >=1 wellness response
#          through
#   last date with >=1 wellness response
#
# Dates outside that range are NOT treated as ordinary
# within-player missingness.
#
# This is an explicit reconstruction choice, not a claim that
# the original paper used this exact definition.
# ============================================================

def restrict_to_active_period(
    daily: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    frame = daily.copy()

    frame[
        "wellness_values_present"
    ] = (
        frame[
            WELLNESS_FEATURES
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    frame[
        "any_wellness_present"
    ] = (
        frame[
            "wellness_values_present"
        ]
        > 0
    )

    frame[
        "complete_wellness"
    ] = (
        frame[
            "wellness_values_present"
        ]
        == len(
            WELLNESS_FEATURES
        )
    )

    frame[
        "partial_wellness"
    ] = (
        (
            frame[
                "wellness_values_present"
            ]
            > 0
        )
        &
        (
            frame[
                "wellness_values_present"
            ]
            < len(
                WELLNESS_FEATURES
            )
        )
    )

    active_rows = []

    coverage_rows = []

    for (
        player_name,
        player_frame,
    ) in frame.groupby(
        "player_name",
        sort=True,
    ):

        player_frame = (
            player_frame
            .sort_values(
                "date"
            )
            .copy()
        )

        observed = player_frame[
            player_frame[
                "any_wellness_present"
            ]
        ]

        if observed.empty:

            coverage_rows.append(
                {
                    "player_name":
                    player_name,

                    "team":
                    player_frame[
                        "team"
                    ].iloc[0],

                    "first_subjective_date":
                    pd.NaT,

                    "last_subjective_date":
                    pd.NaT,

                    "active_calendar_days":
                    0,

                    "days_with_any_wellness":
                    0,

                    "days_with_complete_wellness":
                    0,

                    "days_with_partial_wellness":
                    0,

                    "days_with_no_wellness":
                    0,

                    "any_wellness_coverage_fraction":
                    float("nan"),

                    "complete_wellness_fraction":
                    float("nan"),
                }
            )

            continue

        first_date = (
            observed[
                "date"
            ]
            .min()
        )

        last_date = (
            observed[
                "date"
            ]
            .max()
        )

        active = player_frame[
            (
                player_frame[
                    "date"
                ]
                >= first_date
            )
            &
            (
                player_frame[
                    "date"
                ]
                <= last_date
            )
        ].copy()

        active[
            "within_subjective_active_period"
        ] = True

        active_rows.append(
            active
        )

        active_calendar_days = len(
            active
        )

        any_days = int(
            active[
                "any_wellness_present"
            ].sum()
        )

        complete_days = int(
            active[
                "complete_wellness"
            ].sum()
        )

        partial_days = int(
            active[
                "partial_wellness"
            ].sum()
        )

        no_wellness_days = (
            active_calendar_days
            - any_days
        )

        coverage_rows.append(
            {
                "player_name":
                player_name,

                "team":
                active[
                    "team"
                ].iloc[0],

                "first_subjective_date":
                first_date,

                "last_subjective_date":
                last_date,

                "active_calendar_days":
                active_calendar_days,

                "days_with_any_wellness":
                any_days,

                "days_with_complete_wellness":
                complete_days,

                "days_with_partial_wellness":
                partial_days,

                "days_with_no_wellness":
                no_wellness_days,

                "any_wellness_coverage_fraction":
                (
                    any_days
                    / active_calendar_days
                    if active_calendar_days
                    else float("nan")
                ),

                "complete_wellness_fraction":
                (
                    complete_days
                    / active_calendar_days
                    if active_calendar_days
                    else float("nan")
                ),
            }
        )

    if active_rows:

        active_daily = (
            pd.concat(
                active_rows,
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

    else:

        active_daily = pd.DataFrame()

    coverage = (
        pd.DataFrame(
            coverage_rows
        )
        .sort_values(
            "player_name"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        active_daily,
        coverage,
    )


# ============================================================
# Feature-level missingness within active periods
# ============================================================

def build_feature_missingness(
    daily: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for feature in (
        WELLNESS_FEATURES
    ):

        total_cells = len(
            daily
        )

        present = int(
            daily[
                feature
            ]
            .notna()
            .sum()
        )

        missing = (
            total_cells
            - present
        )

        rows.append(
            {
                "feature":
                feature,

                "player_day_cells":
                total_cells,

                "present_values":
                present,

                "missing_values":
                missing,

                "missing_fraction":
                (
                    missing
                    / total_cells
                    if total_cells
                    else float("nan")
                ),

                "used_in_initial_paper_inspired_model":
                (
                    feature
                    in MODEL_WELLNESS_FEATURES
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 80
    )

    print(
        "SoccerMon Paper-Inspired Subjective Daily Builder"
    )

    print(
        "=" * 80
    )

    print()

    # --------------------------------------------------------
    # Load cohort
    # --------------------------------------------------------

    selected = (
        load_selected_cohort()
    )

    print(
        f"Selected cohort players: "
        f"{len(selected):,}"
    )

    print()


    # --------------------------------------------------------
    # Load and reshape all wellness files
    # --------------------------------------------------------

    full_daily = (
        build_subjective_daily(
            selected
        )
    )

    print()

    print(
        "Full calendar rows before "
        "active-period restriction:",
        f"{len(full_daily):,}",
    )


    # --------------------------------------------------------
    # Restrict each player to observed subjective active period
    # --------------------------------------------------------

    (
        subjective_daily,
        coverage,
    ) = restrict_to_active_period(
        full_daily
    )


    # --------------------------------------------------------
    # Feature-level missingness
    # --------------------------------------------------------

    feature_missingness = (
        build_feature_missingness(
            subjective_daily
        )
    )


    # --------------------------------------------------------
    # Final integrity checks
    # --------------------------------------------------------

    duplicate_player_dates = int(
        subjective_daily
        .duplicated(
            subset=[
                "player_name",
                "date",
            ]
        )
        .sum()
    )

    if duplicate_player_dates:

        raise ValueError(
            "Final subjective daily table contains "
            f"{duplicate_player_dates:,} duplicate player-date rows."
        )


    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    subjective_daily.to_csv(
        DAILY_FILE,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_FILE,
        index=False,
    )

    feature_missingness.to_csv(
        FEATURE_MISSINGNESS_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    players_in_daily = (
        subjective_daily[
            "player_name"
        ]
        .nunique()
    )

    total_player_days = len(
        subjective_daily
    )

    dates = (
        subjective_daily[
            "date"
        ]
    )

    first_date = (
        dates.min()
        if not dates.empty
        else pd.NaT
    )

    last_date = (
        dates.max()
        if not dates.empty
        else pd.NaT
    )

    total_days_with_any = int(
        subjective_daily[
            "any_wellness_present"
        ]
        .sum()
    )

    total_days_complete = int(
        subjective_daily[
            "complete_wellness"
        ]
        .sum()
    )

    total_days_partial = int(
        subjective_daily[
            "partial_wellness"
        ]
        .sum()
    )

    total_days_none = (
        total_player_days
        - total_days_with_any
    )


    # --------------------------------------------------------
    # Coverage distribution
    # --------------------------------------------------------

    coverage_series = (
        coverage[
            "any_wellness_coverage_fraction"
        ]
        .dropna()
    )

    complete_series = (
        coverage[
            "complete_wellness_fraction"
        ]
        .dropna()
    )


    # --------------------------------------------------------
    # Build text summary
    # --------------------------------------------------------

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Paper-Inspired Subjective Daily Summary"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "Source"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Wellness directory: {WELLNESS_DIR}"
    )

    lines.append(
        f"Cohort file: {COHORT_FILE}"
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
        f"{len(selected):,}"
    )

    lines.append(
        f"Players represented after active-period restriction: "
        f"{players_in_daily:,}"
    )

    lines.append("")

    lines.append(
        "Daily table"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Player-day rows: "
        f"{total_player_days:,}"
    )

    lines.append(
        f"Overall date range: "
        f"{first_date.date() if pd.notna(first_date) else 'UNKNOWN'} "
        f"to "
        f"{last_date.date() if pd.notna(last_date) else 'UNKNOWN'}"
    )

    lines.append(
        f"Days with at least one wellness value: "
        f"{total_days_with_any:,}"
    )

    lines.append(
        f"Days with complete wellness data: "
        f"{total_days_complete:,}"
    )

    lines.append(
        f"Days with partial wellness data: "
        f"{total_days_partial:,}"
    )

    lines.append(
        f"Days with no wellness values "
        f"inside active period: "
        f"{total_days_none:,}"
    )

    lines.append("")

    lines.append(
        "Player-level wellness coverage"
    )

    lines.append(
        "-" * 80
    )

    if not coverage_series.empty:

        lines.append(
            f"Minimum any-wellness coverage: "
            f"{coverage_series.min():.4f}"
        )

        lines.append(
            f"Median any-wellness coverage: "
            f"{coverage_series.median():.4f}"
        )

        lines.append(
            f"Mean any-wellness coverage: "
            f"{coverage_series.mean():.4f}"
        )

        lines.append(
            f"Maximum any-wellness coverage: "
            f"{coverage_series.max():.4f}"
        )

    if not complete_series.empty:

        lines.append(
            f"Median complete-wellness coverage: "
            f"{complete_series.median():.4f}"
        )

    lines.append("")

    lines.append(
        "Feature-level missingness within active periods"
    )

    lines.append(
        "-" * 80
    )

    for _, row in (
        feature_missingness.iterrows()
    ):

        lines.append(
            f"{row['feature']}: "
            f"{int(row['missing_values']):,} missing / "
            f"{int(row['player_day_cells']):,} "
            f"({row['missing_fraction']:.4f})"
        )

    lines.append("")

    lines.append(
        "Important interpretation"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "No wellness values are imputed in this script."
    )

    lines.append(
        "The subjective active period is defined as the interval "
        "from a player's first date with at least one observed "
        "wellness value through their last such date."
    )

    lines.append(
        "Dates before the first observed wellness response and "
        "after the last observed wellness response are excluded "
        "from within-player missingness calculations."
    )

    lines.append(
        "This active-period definition is a documented reconstruction "
        "choice for the paper-inspired model."
    )

    lines.append(
        "sleep_quality is preserved in the source-level table but "
        "is not currently part of the initial paper-inspired wellness "
        "feature set."
    )


    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    SUMMARY_FILE.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print()

    print(
        "\n".join(
            lines
        )
    )

    print()

    print(
        f"Subjective daily table written to:\n"
        f"{DAILY_FILE}"
    )

    print()

    print(
        f"Coverage table written to:\n"
        f"{COVERAGE_FILE}"
    )

    print()

    print(
        f"Feature missingness table written to:\n"
        f"{FEATURE_MISSINGNESS_FILE}"
    )

    print()

    print(
        f"Summary written to:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":

    main()