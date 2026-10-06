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

PLAYER_DAY_FILE = (
    BASE_DIR
    / "player_day"
    / "player_day_unimputed.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "player_day"
    / "audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


WINDOW_FILE = (
    OUTPUT_DIR
    / "player_day_window_eligibility.csv"
)

PLAYER_SUMMARY_FILE = (
    OUTPUT_DIR
    / "window_eligibility_by_player.csv"
)

INJURY_EVENT_FILE = (
    OUTPUT_DIR
    / "injury_event_window_eligibility.csv"
)

MODALITY_SUMMARY_FILE = (
    OUTPUT_DIR
    / "window_modality_coverage_summary.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "player_day_window_audit_summary.txt"
)


# ============================================================
# Reconstruction settings
# ============================================================

HISTORY_DAYS = 21
FOLLOWUP_DAYS = 7


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
# Load canonical player-day table
# ============================================================

def load_player_day() -> pd.DataFrame:

    require_file(
        PLAYER_DAY_FILE
    )

    frame = pd.read_csv(
        PLAYER_DAY_FILE
    )

    required = {
        "player_name",
        "team",
        "date",
        "has_subjective",
        "has_training",
        "has_objective",
        "has_any_model_data",
        "has_all_three_modalities",
        "available_modality_count",
        "injury_event",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:

        raise ValueError(
            "Player-day table missing required columns: "
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
        "has_any_model_data",
        "has_all_three_modalities",
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
            f"Found {duplicate_count:,} duplicate "
            f"player-date rows."
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
# Modality range helper
#
# This lets us distinguish:
#
#     no observation because the modality is structurally
#     outside that player's observed period
#
# from:
#
#     missing observation inside the player's observed period
# ============================================================

def get_modality_ranges(
    frame: pd.DataFrame,
) -> dict:

    ranges = {}

    for modality, column in [
        (
            "subjective",
            "has_subjective",
        ),
        (
            "training",
            "has_training",
        ),
        (
            "objective",
            "has_objective",
        ),
    ]:

        observed = frame[
            frame[
                column
            ]
        ]

        if observed.empty:

            ranges[
                modality
            ] = (
                pd.NaT,
                pd.NaT,
            )

        else:

            ranges[
                modality
            ] = (
                observed[
                    "date"
                ].min(),

                observed[
                    "date"
                ].max(),
            )

    return ranges


# ============================================================
# Count dates falling inside a modality's overall observed span
# ============================================================

def count_dates_inside_range(
    dates: pd.DatetimeIndex,
    start,
    end,
) -> int:

    if (
        pd.isna(start)
        or
        pd.isna(end)
    ):

        return 0

    return int(
        (
            (dates >= start)
            &
            (dates <= end)
        ).sum()
    )


# ============================================================
# Audit one player's possible windows
# ============================================================

def audit_player(
    player_frame: pd.DataFrame,
) -> list[dict]:

    player_frame = (
        player_frame
        .sort_values(
            "date"
        )
        .copy()
    )

    player_name = (
        player_frame[
            "player_name"
        ].iloc[0]
    )

    team = (
        player_frame[
            "team"
        ].iloc[0]
    )

    first_calendar_date = (
        player_frame[
            "date"
        ].min()
    )

    last_calendar_date = (
        player_frame[
            "date"
        ].max()
    )

    modality_ranges = (
        get_modality_ranges(
            player_frame
        )
    )

    # --------------------------------------------------------
    # Fast lookup table by calendar date
    # --------------------------------------------------------

    lookup = (
        player_frame
        .set_index(
            "date"
        )
        .sort_index()
    )

    rows = []

    for prediction_date in (
        player_frame[
            "date"
        ]
    ):

        # ----------------------------------------------------
        # History:
        #
        # t-21 ... t-1
        # ----------------------------------------------------

        history_dates = pd.date_range(
            prediction_date
            - pd.Timedelta(
                days=HISTORY_DAYS
            ),
            prediction_date
            - pd.Timedelta(
                days=1
            ),
            freq="D",
        )

        # ----------------------------------------------------
        # Future:
        #
        # t+1 ... t+7
        # ----------------------------------------------------

        future_dates = pd.date_range(
            prediction_date
            + pd.Timedelta(
                days=1
            ),
            prediction_date
            + pd.Timedelta(
                days=FOLLOWUP_DAYS
            ),
            freq="D",
        )

        # ----------------------------------------------------
        # Calendar eligibility
        # ----------------------------------------------------

        has_full_history_calendar = (
            history_dates.min()
            >= first_calendar_date
        )

        has_full_followup_calendar = (
            future_dates.max()
            <= last_calendar_date
        )

        # ----------------------------------------------------
        # Restrict to history rows.
        #
        # Because the merged table is a complete calendar,
        # all dates inside the player's calendar should exist.
        # ----------------------------------------------------

        history = lookup.reindex(
            history_dates
        )

        future = lookup.reindex(
            future_dates
        )

        # ----------------------------------------------------
        # History modality counts
        # ----------------------------------------------------

        subjective_days = int(
            history[
                "has_subjective"
            ]
            .fillna(False)
            .sum()
        )

        training_days = int(
            history[
                "has_training"
            ]
            .fillna(False)
            .sum()
        )

        objective_days = int(
            history[
                "has_objective"
            ]
            .fillna(False)
            .sum()
        )

        any_data_days = int(
            history[
                "has_any_model_data"
            ]
            .fillna(False)
            .sum()
        )

        all_three_days = int(
            history[
                "has_all_three_modalities"
            ]
            .fillna(False)
            .sum()
        )

        zero_modality_days = int(
            (
                history[
                    "available_modality_count"
                ]
                .fillna(0)
                == 0
            ).sum()
        )

        one_modality_days = int(
            (
                history[
                    "available_modality_count"
                ]
                .fillna(0)
                == 1
            ).sum()
        )

        two_modality_days = int(
            (
                history[
                    "available_modality_count"
                ]
                .fillna(0)
                == 2
            ).sum()
        )

        three_modality_days = int(
            (
                history[
                    "available_modality_count"
                ]
                .fillna(0)
                == 3
            ).sum()
        )

        # ----------------------------------------------------
        # Structural coverage
        #
        # How many of the 21 history calendar days even fall
        # inside the player's observed period for each modality?
        # ----------------------------------------------------

        subjective_range_days = (
            count_dates_inside_range(
                history_dates,
                *modality_ranges[
                    "subjective"
                ],
            )
        )

        training_range_days = (
            count_dates_inside_range(
                history_dates,
                *modality_ranges[
                    "training"
                ],
            )
        )

        objective_range_days = (
            count_dates_inside_range(
                history_dates,
                *modality_ranges[
                    "objective"
                ],
            )
        )

        # ----------------------------------------------------
        # Within-range missing days
        #
        # These are potentially imputation-type gaps.
        #
        # Days outside the observed range are structural
        # absence and should NOT be treated the same way.
        # ----------------------------------------------------

        subjective_within_range_missing = (
            subjective_range_days
            - subjective_days
        )

        training_within_range_missing = (
            training_range_days
            - training_days
        )

        objective_within_range_missing = (
            objective_range_days
            - objective_days
        )

        subjective_structural_absence = (
            HISTORY_DAYS
            - subjective_range_days
        )

        training_structural_absence = (
            HISTORY_DAYS
            - training_range_days
        )

        objective_structural_absence = (
            HISTORY_DAYS
            - objective_range_days
        )

        # ----------------------------------------------------
        # Future injury information
        # ----------------------------------------------------

        future_injury_events = int(
            future[
                "injury_event"
            ]
            .fillna(0)
            .sum()
        )

        injury_within_followup = (
            future_injury_events
            > 0
        )

        # ----------------------------------------------------
        # Days until first injury in future window
        # ----------------------------------------------------

        future_event_rows = future[
            future[
                "injury_event"
            ]
            .fillna(0)
            .astype(int)
            > 0
        ]

        if not future_event_rows.empty:

            first_future_injury_date = (
                future_event_rows.index.min()
            )

            days_until_injury = int(
                (
                    first_future_injury_date
                    - prediction_date
                ).days
            )

        else:

            first_future_injury_date = (
                pd.NaT
            )

            days_until_injury = (
                np.nan
            )

        # ----------------------------------------------------
        # Today's injury status
        #
        # We preserve this because prediction windows should
        # generally not use an already-injured day as the
        # beginning of a new risk window.
        # ----------------------------------------------------

        current_row = lookup.loc[
            prediction_date
        ]

        injury_on_prediction_day = (
            int(
                current_row[
                    "injury_event"
                ]
            )
            > 0
        )

        # ----------------------------------------------------
        # Eligibility definitions
        #
        # We intentionally create multiple definitions rather
        # than deciding on one prematurely.
        # ----------------------------------------------------

        calendar_eligible = (
            has_full_history_calendar
            and
            has_full_followup_calendar
            and
            not injury_on_prediction_day
        )

        # STRICT:
        #
        # Every history day has all 3 modalities.
        strict_multimodal_eligible = (
            calendar_eligible
            and
            all_three_days
            == HISTORY_DAYS
        )

        # RANGE:
        #
        # Entire 21-day history lies inside all three modality
        # observation spans.
        #
        # Missing individual observations can still exist.
        full_modality_range_eligible = (
            calendar_eligible
            and
            subjective_range_days
            == HISTORY_DAYS
            and
            training_range_days
            == HISTORY_DAYS
            and
            objective_range_days
            == HISTORY_DAYS
        )

        # PARTIAL:
        #
        # At least one actual observation from every modality
        # appears somewhere in the 21-day history.
        partial_multimodal_eligible = (
            calendar_eligible
            and
            subjective_days
            > 0
            and
            training_days
            > 0
            and
            objective_days
            > 0
        )

        rows.append(
            {
                "player_name":
                    player_name,

                "team":
                    team,

                "prediction_date":
                    prediction_date,

                "history_start":
                    history_dates.min(),

                "history_end":
                    history_dates.max(),

                "followup_start":
                    future_dates.min(),

                "followup_end":
                    future_dates.max(),

                # Calendar
                "has_full_history_calendar":
                    has_full_history_calendar,

                "has_full_followup_calendar":
                    has_full_followup_calendar,

                "injury_on_prediction_day":
                    injury_on_prediction_day,

                "calendar_eligible":
                    calendar_eligible,

                # Observed history counts
                "subjective_days_21d":
                    subjective_days,

                "training_days_21d":
                    training_days,

                "objective_days_21d":
                    objective_days,

                "any_data_days_21d":
                    any_data_days,

                "all_three_days_21d":
                    all_three_days,

                # Modality count breakdown
                "zero_modality_days_21d":
                    zero_modality_days,

                "one_modality_days_21d":
                    one_modality_days,

                "two_modality_days_21d":
                    two_modality_days,

                "three_modality_days_21d":
                    three_modality_days,

                # Coverage fractions
                "subjective_coverage_21d":
                    subjective_days
                    / HISTORY_DAYS,

                "training_coverage_21d":
                    training_days
                    / HISTORY_DAYS,

                "objective_coverage_21d":
                    objective_days
                    / HISTORY_DAYS,

                "all_three_coverage_21d":
                    all_three_days
                    / HISTORY_DAYS,

                "any_data_coverage_21d":
                    any_data_days
                    / HISTORY_DAYS,

                # Structural-range diagnostics
                "subjective_range_days_21d":
                    subjective_range_days,

                "training_range_days_21d":
                    training_range_days,

                "objective_range_days_21d":
                    objective_range_days,

                "subjective_within_range_missing_21d":
                    subjective_within_range_missing,

                "training_within_range_missing_21d":
                    training_within_range_missing,

                "objective_within_range_missing_21d":
                    objective_within_range_missing,

                "subjective_structural_absence_21d":
                    subjective_structural_absence,

                "training_structural_absence_21d":
                    training_structural_absence,

                "objective_structural_absence_21d":
                    objective_structural_absence,

                # Eligibility variants
                "strict_multimodal_eligible":
                    strict_multimodal_eligible,

                "full_modality_range_eligible":
                    full_modality_range_eligible,

                "partial_multimodal_eligible":
                    partial_multimodal_eligible,

                # Outcome / future
                "injury_within_7d":
                    injury_within_followup,

                "future_injury_event_count":
                    future_injury_events,

                "first_future_injury_date":
                    first_future_injury_date,

                "days_until_injury":
                    days_until_injury,
            }
        )

    return rows


# ============================================================
# Audit all players
# ============================================================

def build_window_audit(
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

        print(
            f"Auditing {player_name}..."
        )

        rows.extend(
            audit_player(
                frame
            )
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "player_name",
                "prediction_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Player summary
# ============================================================

def build_player_summary(
    windows: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        player_name,
        frame,
    ) in windows.groupby(
        "player_name",
        sort=True,
    ):

        calendar = frame[
            frame[
                "calendar_eligible"
            ]
        ]

        rows.append(
            {
                "player_name":
                    player_name,

                "team":
                    frame[
                        "team"
                    ].iloc[0],

                "candidate_prediction_days":
                    len(
                        frame
                    ),

                "calendar_eligible_days":
                    int(
                        frame[
                            "calendar_eligible"
                        ].sum()
                    ),

                "strict_multimodal_days":
                    int(
                        frame[
                            "strict_multimodal_eligible"
                        ].sum()
                    ),

                "full_modality_range_days":
                    int(
                        frame[
                            "full_modality_range_eligible"
                        ].sum()
                    ),

                "partial_multimodal_days":
                    int(
                        frame[
                            "partial_multimodal_eligible"
                        ].sum()
                    ),

                "positive_7d_windows":
                    int(
                        (
                            frame[
                                "calendar_eligible"
                            ]
                            &
                            frame[
                                "injury_within_7d"
                            ]
                        ).sum()
                    ),

                "strict_positive_7d_windows":
                    int(
                        (
                            frame[
                                "strict_multimodal_eligible"
                            ]
                            &
                            frame[
                                "injury_within_7d"
                            ]
                        ).sum()
                    ),

                "range_positive_7d_windows":
                    int(
                        (
                            frame[
                                "full_modality_range_eligible"
                            ]
                            &
                            frame[
                                "injury_within_7d"
                            ]
                        ).sum()
                    ),

                "partial_positive_7d_windows":
                    int(
                        (
                            frame[
                                "partial_multimodal_eligible"
                            ]
                            &
                            frame[
                                "injury_within_7d"
                            ]
                        ).sum()
                    ),

                "median_subjective_coverage":
                    (
                        calendar[
                            "subjective_coverage_21d"
                        ].median()
                        if not calendar.empty
                        else np.nan
                    ),

                "median_training_coverage":
                    (
                        calendar[
                            "training_coverage_21d"
                        ].median()
                        if not calendar.empty
                        else np.nan
                    ),

                "median_objective_coverage":
                    (
                        calendar[
                            "objective_coverage_21d"
                        ].median()
                        if not calendar.empty
                        else np.nan
                    ),

                "median_all_three_coverage":
                    (
                        calendar[
                            "all_three_coverage_21d"
                        ].median()
                        if not calendar.empty
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Injury-event eligibility table
#
# For every actual injury event on date I, inspect the
# prediction windows ending 1-7 days before I.
# ============================================================

def build_injury_event_summary(
    player_day: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:

    injury_events = player_day[
        player_day[
            "injury_event"
        ]
        > 0
    ][
        [
            "player_name",
            "team",
            "date",
            "injury_episode_count",
            "body_regions",
        ]
    ].copy()

    rows = []

    for _, event in (
        injury_events.iterrows()
    ):

        player = (
            event[
                "player_name"
            ]
        )

        injury_date = (
            event[
                "date"
            ]
        )

        candidate_windows = windows[
            (
                windows[
                    "player_name"
                ]
                == player
            )
            &
            (
                windows[
                    "prediction_date"
                ]
                >= injury_date
                - pd.Timedelta(
                    days=FOLLOWUP_DAYS
                )
            )
            &
            (
                windows[
                    "prediction_date"
                ]
                < injury_date
            )
        ].copy()

        rows.append(
            {
                "player_name":
                    player,

                "team":
                    event[
                        "team"
                    ],

                "injury_date":
                    injury_date,

                "injury_episode_count":
                    event.get(
                        "injury_episode_count",
                        np.nan,
                    ),

                "body_regions":
                    event.get(
                        "body_regions",
                        "",
                    ),

                "candidate_prediction_windows_prior_7d":
                    len(
                        candidate_windows
                    ),

                "calendar_eligible_windows":
                    int(
                        candidate_windows[
                            "calendar_eligible"
                        ].sum()
                    ),

                "strict_multimodal_windows":
                    int(
                        candidate_windows[
                            "strict_multimodal_eligible"
                        ].sum()
                    ),

                "full_modality_range_windows":
                    int(
                        candidate_windows[
                            "full_modality_range_eligible"
                        ].sum()
                    ),

                "partial_multimodal_windows":
                    int(
                        candidate_windows[
                            "partial_multimodal_eligible"
                        ].sum()
                    ),

                "best_subjective_days_21d":
                    (
                        candidate_windows[
                            "subjective_days_21d"
                        ].max()
                        if not candidate_windows.empty
                        else 0
                    ),

                "best_training_days_21d":
                    (
                        candidate_windows[
                            "training_days_21d"
                        ].max()
                        if not candidate_windows.empty
                        else 0
                    ),

                "best_objective_days_21d":
                    (
                        candidate_windows[
                            "objective_days_21d"
                        ].max()
                        if not candidate_windows.empty
                        else 0
                    ),

                "best_all_three_days_21d":
                    (
                        candidate_windows[
                            "all_three_days_21d"
                        ].max()
                        if not candidate_windows.empty
                        else 0
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Modality coverage distribution
# ============================================================

def build_modality_summary(
    windows: pd.DataFrame,
) -> pd.DataFrame:

    eligible = windows[
        windows[
            "calendar_eligible"
        ]
    ].copy()

    rows = []

    for modality in [
        "subjective",
        "training",
        "objective",
        "all_three",
        "any_data",
    ]:

        column = (
            f"{modality}_coverage_21d"
        )

        values = (
            eligible[
                column
            ]
            .dropna()
        )

        if values.empty:
            continue

        rows.append(
            {
                "modality":
                    modality,

                "windows":
                    len(
                        values
                    ),

                "minimum":
                    values.min(),

                "q25":
                    values.quantile(
                        0.25
                    ),

                "median":
                    values.median(),

                "mean":
                    values.mean(),

                "q75":
                    values.quantile(
                        0.75
                    ),

                "maximum":
                    values.max(),
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
        "SoccerMon Paper-Inspired Player-Day Window Audit"
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

    # --------------------------------------------------------
    # Window audit
    # --------------------------------------------------------

    windows = (
        build_window_audit(
            player_day
        )
    )

    # --------------------------------------------------------
    # Derived summaries
    # --------------------------------------------------------

    player_summary = (
        build_player_summary(
            windows
        )
    )

    injury_summary = (
        build_injury_event_summary(
            player_day,
            windows,
        )
    )

    modality_summary = (
        build_modality_summary(
            windows
        )
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    windows.to_csv(
        WINDOW_FILE,
        index=False,
    )

    player_summary.to_csv(
        PLAYER_SUMMARY_FILE,
        index=False,
    )

    injury_summary.to_csv(
        INJURY_EVENT_FILE,
        index=False,
    )

    modality_summary.to_csv(
        MODALITY_SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # Global summary
    # ========================================================

    total_windows = len(
        windows
    )

    calendar_eligible = int(
        windows[
            "calendar_eligible"
        ].sum()
    )

    strict_eligible = int(
        windows[
            "strict_multimodal_eligible"
        ].sum()
    )

    range_eligible = int(
        windows[
            "full_modality_range_eligible"
        ].sum()
    )

    partial_eligible = int(
        windows[
            "partial_multimodal_eligible"
        ].sum()
    )

    calendar_positive = int(
        (
            windows[
                "calendar_eligible"
            ]
            &
            windows[
                "injury_within_7d"
            ]
        ).sum()
    )

    strict_positive = int(
        (
            windows[
                "strict_multimodal_eligible"
            ]
            &
            windows[
                "injury_within_7d"
            ]
        ).sum()
    )

    range_positive = int(
        (
            windows[
                "full_modality_range_eligible"
            ]
            &
            windows[
                "injury_within_7d"
            ]
        ).sum()
    )

    partial_positive = int(
        (
            windows[
                "partial_multimodal_eligible"
            ]
            &
            windows[
                "injury_within_7d"
            ]
        ).sum()
    )

    total_injury_events = len(
        injury_summary
    )

    events_with_partial_window = int(
        (
            injury_summary[
                "partial_multimodal_windows"
            ]
            > 0
        ).sum()
    )

    events_with_range_window = int(
        (
            injury_summary[
                "full_modality_range_windows"
            ]
            > 0
        ).sum()
    )

    events_with_strict_window = int(
        (
            injury_summary[
                "strict_multimodal_windows"
            ]
            > 0
        ).sum()
    )

    # --------------------------------------------------------
    # Summary file
    # --------------------------------------------------------

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Paper-Inspired Player-Day Window Audit"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "Window definition"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"History window: {HISTORY_DAYS} prior calendar days"
    )

    lines.append(
        f"Future follow-up: {FOLLOWUP_DAYS} calendar days"
    )

    lines.append(
        "Prediction day itself is not included in either history or future outcome."
    )

    lines.append("")

    lines.append(
        "Candidate windows"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"All possible player-day windows: "
        f"{total_windows:,}"
    )

    lines.append(
        f"Calendar-eligible windows: "
        f"{calendar_eligible:,}"
    )

    lines.append(
        f"Strict complete multimodal windows: "
        f"{strict_eligible:,}"
    )

    lines.append(
        f"Full modality-range windows: "
        f"{range_eligible:,}"
    )

    lines.append(
        f"Partial multimodal windows: "
        f"{partial_eligible:,}"
    )

    lines.append("")

    lines.append(
        "7-day injury-positive windows"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Calendar-eligible positive windows: "
        f"{calendar_positive:,}"
    )

    lines.append(
        f"Strict multimodal positive windows: "
        f"{strict_positive:,}"
    )

    lines.append(
        f"Full modality-range positive windows: "
        f"{range_positive:,}"
    )

    lines.append(
        f"Partial multimodal positive windows: "
        f"{partial_positive:,}"
    )

    lines.append("")

    lines.append(
        "Unique injury-event support"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Observed injury-event days: "
        f"{total_injury_events:,}"
    )

    lines.append(
        f"Injury events with >=1 strict multimodal prediction window: "
        f"{events_with_strict_window:,}"
    )

    lines.append(
        f"Injury events with >=1 full-range prediction window: "
        f"{events_with_range_window:,}"
    )

    lines.append(
        f"Injury events with >=1 partial multimodal prediction window: "
        f"{events_with_partial_window:,}"
    )

    lines.append("")

    lines.append(
        "Eligibility definitions"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "Calendar eligible: full 21-day history, full 7-day future follow-up, "
        "and no injury event on prediction day."
    )

    lines.append(
        "Strict multimodal: all 21 history days contain subjective, training, "
        "and objective observations."
    )

    lines.append(
        "Full modality range: all 21 history days fall inside the observed "
        "date ranges of all three modalities, although individual values may "
        "still be missing."
    )

    lines.append(
        "Partial multimodal: at least one subjective, one training, and one "
        "objective observation occur somewhere within the 21-day history."
    )

    lines.append("")

    lines.append(
        "Important interpretation"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "This audit does not select the final DeepHit eligibility rule."
    )

    lines.append(
        "It separates structural modality absence from within-range missingness "
        "so that later imputation decisions do not manufacture data outside "
        "actual observation periods."
    )

    lines.append(
        "Positive-window counts are not equivalent to unique injuries because "
        "one injury can appear in several prediction windows during the "
        "7-day forecast horizon."
    )

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
        f"Window-level audit written to:\n"
        f"{WINDOW_FILE}"
    )

    print()

    print(
        f"Player summary written to:\n"
        f"{PLAYER_SUMMARY_FILE}"
    )

    print()

    print(
        f"Injury-event summary written to:\n"
        f"{INJURY_EVENT_FILE}"
    )

    print()

    print(
        f"Modality summary written to:\n"
        f"{MODALITY_SUMMARY_FILE}"
    )

    print()

    print(
        f"Text summary written to:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":

    main()