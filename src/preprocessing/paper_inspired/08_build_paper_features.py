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

INPUT_FILE = (
    BASE_DIR
    / "player_day"
    / "player_day_unimputed.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_FILE = (
    OUTPUT_DIR
    / "paper_features_unimputed.csv"
)

COVERAGE_FILE = (
    OUTPUT_DIR
    / "paper_feature_coverage.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "paper_feature_summary.txt"
)


# ============================================================
# Reconstruction settings
# ============================================================

ATL_WINDOW = 7
WEEKLY_LOAD_WINDOW = 7
MONOTONY_WINDOW = 7
CTL28_WINDOW = 28
CTL42_WINDOW = 42
SUBJECTIVE_MISSINGNESS_WINDOW = 7


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


# ============================================================
# Load canonical player-day data
# ============================================================

def load_player_day() -> pd.DataFrame:

    require_file(
        INPUT_FILE
    )

    frame = pd.read_csv(
        INPUT_FILE
    )

    required = {
        "player_name",
        "team",
        "date",
        "has_subjective",
        "has_training",
        "has_objective",
        "injury_event",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:

        raise ValueError(
            "Missing required player-day columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    frame[
        "date"
    ] = pd.to_datetime(
        frame[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()

    for column in [
        "has_subjective",
        "has_training",
        "has_objective",
    ]:

        frame[
            column
        ] = parse_bool(
            frame[
                column
            ]
        )

    frame[
        "injury_event"
    ] = (
        pd.to_numeric(
            frame[
                "injury_event"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    duplicate_count = int(
        frame.duplicated(
            subset=[
                "player_name",
                "date",
            ]
        ).sum()
    )

    if duplicate_count:

        raise ValueError(
            f"Duplicate player-date rows: "
            f"{duplicate_count:,}"
        )

    return (
        frame
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
# Resolve training-load source
#
# The paper defines Daily Load as the sum of session RPE
# values per day.
#
# In our reconstructed training table, srpe_total represents
# summed session RPE where:
#
#     sRPE = RPE * session duration
#
# This is therefore used as the paper-inspired daily-load
# source.
# ============================================================

def resolve_daily_load(
    frame: pd.DataFrame,
) -> pd.Series:

    if "srpe_total" not in frame.columns:

        raise ValueError(
            "Expected 'srpe_total' in merged player-day table."
        )

    daily_load = pd.to_numeric(
        frame[
            "srpe_total"
        ],
        errors="coerce",
    )

    # Important:
    #
    # Missing training records remain NaN.
    #
    # We do NOT assume that no training row automatically means
    # zero load at this stage.
    return daily_load


# ============================================================
# Build paper-derived features for one player
# ============================================================

def build_player_features(
    player_frame: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        player_frame
        .sort_values(
            "date"
        )
        .copy()
    )

    # --------------------------------------------------------
    # Daily load
    # --------------------------------------------------------

    result[
        "daily_load"
    ] = resolve_daily_load(
        result
    )

    # --------------------------------------------------------
    # ATL
    #
    # Paper:
    # average training load over previous 7 days.
    #
    # min_periods=1 preserves partial histories at this stage.
    # Missing values are not filled here.
    # --------------------------------------------------------

    result[
        "atl_7d"
    ] = (
        result[
            "daily_load"
        ]
        .rolling(
            window=ATL_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Weekly Load
    #
    # Paper:
    # sum of training load over previous 7 days.
    # --------------------------------------------------------

    result[
        "weekly_load_7d"
    ] = (
        result[
            "daily_load"
        ]
        .rolling(
            window=WEEKLY_LOAD_WINDOW,
            min_periods=1,
        )
        .sum()
    )

    # --------------------------------------------------------
    # Monotony
    #
    # Common training-load formulation:
    #
    #     7-day mean load / 7-day SD load
    #
    # The paper only describes monotony as day-to-day
    # variability and does not report its exact implementation.
    #
    # We therefore record this as a reconstruction assumption.
    # --------------------------------------------------------

    load_mean_7d = (
        result[
            "daily_load"
        ]
        .rolling(
            window=MONOTONY_WINDOW,
            min_periods=2,
        )
        .mean()
    )

    load_sd_7d = (
        result[
            "daily_load"
        ]
        .rolling(
            window=MONOTONY_WINDOW,
            min_periods=2,
        )
        .std()
    )

    result[
        "monotony_7d"
    ] = (
        load_mean_7d
        /
        load_sd_7d.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Strain
    #
    # Paper:
    # training load x monotony.
    #
    # We use weekly load as the 7-day training-load quantity.
    # --------------------------------------------------------

    result[
        "strain_7d"
    ] = (
        result[
            "weekly_load_7d"
        ]
        *
        result[
            "monotony_7d"
        ]
    )

    # --------------------------------------------------------
    # CTL28
    #
    # Paper:
    # sum of training load over previous 28 days.
    # --------------------------------------------------------

    result[
        "ctl28"
    ] = (
        result[
            "daily_load"
        ]
        .rolling(
            window=CTL28_WINDOW,
            min_periods=1,
        )
        .sum()
    )

    # --------------------------------------------------------
    # CTL42
    #
    # Paper:
    # sum of training load over previous 42 days.
    # --------------------------------------------------------

    result[
        "ctl42"
    ] = (
        result[
            "daily_load"
        ]
        .rolling(
            window=CTL42_WINDOW,
            min_periods=1,
        )
        .sum()
    )

    # --------------------------------------------------------
    # ACWR
    #
    # Paper:
    # acute:chronic workload ratio.
    #
    # The manuscript does not state the exact denominator.
    #
    # Reconstruction assumption:
    #
    #     7-day mean load
    #     ----------------
    #     28-day mean load
    #
    # --------------------------------------------------------

    chronic_mean_28d = (
        result[
            "daily_load"
        ]
        .rolling(
            window=CTL28_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    result[
        "acwr_7_28"
    ] = (
        result[
            "atl_7d"
        ]
        /
        chronic_mean_28d.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Subjective missingness
    #
    # Paper:
    # proportion of subjective questionnaires not completed in
    # the previous 7 days.
    #
    # has_subjective=True means questionnaire information was
    # actually observed for that player-day.
    # --------------------------------------------------------

    subjective_observed = (
        result[
            "has_subjective"
        ]
        .astype(int)
    )

    observed_subjective_7d = (
        subjective_observed
        .rolling(
            window=SUBJECTIVE_MISSINGNESS_WINDOW,
            min_periods=1,
        )
        .sum()
    )

    available_history_7d = (
        pd.Series(
            1,
            index=result.index,
            dtype=float,
        )
        .rolling(
            window=SUBJECTIVE_MISSINGNESS_WINDOW,
            min_periods=1,
        )
        .sum()
    )

    result[
        "subjective_missingness_7d"
    ] = (
        1.0
        -
        (
            observed_subjective_7d
            /
            available_history_7d
        )
    )

    # --------------------------------------------------------
    # Past injury count
    #
    # Cumulative injuries prior to the current day.
    #
    # shift(1) is critical:
    #
    # today's injury must not contribute to today's predictor,
    # otherwise we would leak outcome information.
    # --------------------------------------------------------

    result[
        "past_injury_count"
    ] = (
        result[
            "injury_event"
        ]
        .shift(1)
        .fillna(0)
        .cumsum()
        .astype(int)
    )

    return result


# ============================================================
# Build all player features
# ============================================================

def build_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    groups = []

    for (
        player_name,
        player_frame,
    ) in frame.groupby(
        "player_name",
        sort=True,
    ):

        print(
            f"Building features for "
            f"{player_name}..."
        )

        groups.append(
            build_player_features(
                player_frame
            )
        )

    return (
        pd.concat(
            groups,
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


# ============================================================
# Feature coverage summary
# ============================================================

def build_feature_coverage(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    derived_features = [
        "daily_load",
        "atl_7d",
        "weekly_load_7d",
        "monotony_7d",
        "strain_7d",
        "ctl28",
        "ctl42",
        "acwr_7_28",
        "subjective_missingness_7d",
        "past_injury_count",
    ]

    rows = []

    total_rows = len(
        frame
    )

    for feature in derived_features:

        observed = int(
            frame[
                feature
            ]
            .notna()
            .sum()
        )

        missing = int(
            frame[
                feature
            ]
            .isna()
            .sum()
        )

        rows.append(
            {
                "feature":
                    feature,

                "rows":
                    total_rows,

                "observed":
                    observed,

                "missing":
                    missing,

                "observed_fraction":
                    (
                        observed
                        / total_rows
                        if total_rows
                        else np.nan
                    ),

                "missing_fraction":
                    (
                        missing
                        / total_rows
                        if total_rows
                        else np.nan
                    ),

                "minimum":
                    frame[
                        feature
                    ].min(),

                "median":
                    frame[
                        feature
                    ].median(),

                "mean":
                    frame[
                        feature
                    ].mean(),

                "maximum":
                    frame[
                        feature
                    ].max(),
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
        "SoccerMon Paper-Inspired Feature Builder"
    )

    print(
        "=" * 80
    )

    print()

    player_day = (
        load_player_day()
    )

    print(
        f"Player-day rows loaded: "
        f"{len(player_day):,}"
    )

    print(
        f"Players: "
        f"{player_day['player_name'].nunique():,}"
    )

    print()

    features = (
        build_features(
            player_day
        )
    )

    coverage = (
        build_feature_coverage(
            features
        )
    )

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    duplicates = int(
        features.duplicated(
            subset=[
                "player_name",
                "date",
            ]
        ).sum()
    )

    if duplicates:

        raise ValueError(
            f"Duplicate player-date rows after "
            f"feature engineering: {duplicates:,}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    features.to_csv(
        FEATURE_FILE,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Paper-Inspired Feature Summary"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "Dataset"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Players: "
        f"{features['player_name'].nunique():,}"
    )

    lines.append(
        f"Player-day rows: "
        f"{len(features):,}"
    )

    lines.append("")

    lines.append(
        "Derived paper-inspired features"
    )

    lines.append(
        "-" * 80
    )

    for feature in [
        "daily_load",
        "atl_7d",
        "weekly_load_7d",
        "monotony_7d",
        "strain_7d",
        "ctl28",
        "ctl42",
        "acwr_7_28",
        "subjective_missingness_7d",
        "past_injury_count",
    ]:

        lines.append(
            feature
        )

    lines.append("")

    lines.append(
        "Reconstruction assumptions"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "1. srpe_total is used as Daily Load because the paper "
        "defines daily load as the sum of session RPE values "
        "across the day."
    )

    lines.append(
        "2. ATL is the rolling 7-day mean of Daily Load."
    )

    lines.append(
        "3. Weekly Load is the rolling 7-day sum of Daily Load."
    )

    lines.append(
        "4. Monotony is reconstructed as the rolling 7-day "
        "mean divided by rolling 7-day standard deviation."
    )

    lines.append(
        "5. Strain is reconstructed as Weekly Load multiplied "
        "by Monotony."
    )

    lines.append(
        "6. CTL28 and CTL42 are rolling sums over 28 and 42 "
        "calendar days, respectively."
    )

    lines.append(
        "7. ACWR is reconstructed as 7-day mean load divided "
        "by 28-day mean load because the manuscript does not "
        "specify the chronic denominator."
    )

    lines.append(
        "8. Missing training records remain missing rather than "
        "being automatically interpreted as zero-load days."
    )

    lines.append(
        "9. Past injury count uses only injuries occurring "
        "before the current player-day to prevent leakage."
    )

    lines.append(
        "10. No feature imputation is performed in this script."
    )

    SUMMARY_FILE.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(
            lines
        )
    )

    print()

    print(
        f"Feature table written to:\n"
        f"{FEATURE_FILE}"
    )

    print()

    print(
        f"Feature coverage written to:\n"
        f"{COVERAGE_FILE}"
    )

    print()

    print(
        f"Summary written to:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()