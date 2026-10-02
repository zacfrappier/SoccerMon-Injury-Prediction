from pathlib import Path
import math

import numpy as np
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

MANIFEST_FILE = (
    AUDIT_DIR
    / "objective_file_manifest.csv"
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
    / "objective"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Output files
# ============================================================

SESSION_FILE = (
    OUTPUT_DIR
    / "objective_session_features.csv"
)

DAILY_FILE = (
    OUTPUT_DIR
    / "objective_daily.csv"
)

QA_FILE = (
    OUTPUT_DIR
    / "objective_session_qa.csv"
)

COVERAGE_FILE = (
    OUTPUT_DIR
    / "objective_coverage.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "objective_daily_summary.txt"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR
    / "objective_session_checkpoint.csv"
)


# ============================================================
# Reconstruction settings
# ============================================================

TARGET_TEAM = "TeamB"


# ------------------------------------------------------------
# Speed units
#
# SoccerMon speed is treated as metres/second.
# ------------------------------------------------------------

MPS_TO_KMH = 3.6


# ------------------------------------------------------------
# Paper-inspired QA limits
#
# These limits are explicitly described in the DeepHit paper.
# ------------------------------------------------------------

MAX_SPEED_KMH = 32.0
MAX_DURATION_MIN = 200.0
MAX_DISTANCE_KM = 16.0


# ------------------------------------------------------------
# Paper-inspired running-speed zones
#
# IMPORTANT:
# These exact four boundaries were not published in the
# DeepHit manuscript.
#
# We therefore use documented women's-football / SoccerMon
# thresholds as a reconstruction choice.
#
# low:     < 3.33 m/s
# medium:  3.33 - 4.44 m/s
# high:    > 4.44 - 5.55 m/s
# sprint:  > 5.55 m/s
# ------------------------------------------------------------

MEDIUM_SPEED_MIN_MPS = 3.33
HIGH_SPEED_MIN_MPS = 4.44
SPRINT_SPEED_MIN_MPS = 5.55


# ------------------------------------------------------------
# Timestamp-gap handling
#
# Normal GNSS timestamps occur at approximately 0.1 sec.
#
# Large gaps should not be interpreted as continuous movement.
# We therefore integrate distance/time only across intervals
# <= 1 second.
#
# This is a documented reconstruction choice.
# ------------------------------------------------------------

MAX_VALID_INTERVAL_SECONDS = 1.0


# ------------------------------------------------------------
# Checkpoint frequency
# ------------------------------------------------------------

CHECKPOINT_EVERY_FILES = 50


# ============================================================
# Helpers
# ============================================================

def require_file(path: Path) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def parse_bool(
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
            }
        )
        .fillna(False)
        .astype(bool)
    )


# ============================================================
# Cohort
# ============================================================

def load_selected_cohort() -> pd.DataFrame:

    require_file(
        COHORT_FILE
    )

    frame = pd.read_csv(
        COHORT_FILE
    )

    required = {
        "player_name",
        "player_id",
        "team",
        "include_in_deephit",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:

        raise ValueError(
            "Cohort file missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    include = parse_bool(
        frame[
            "include_in_deephit"
        ]
    )

    result = (
        frame.loc[
            include,
            [
                "player_name",
                "player_id",
                "team",
            ],
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# Manifest
# ============================================================

def load_manifest(
    cohort: pd.DataFrame,
) -> pd.DataFrame:

    require_file(
        MANIFEST_FILE
    )

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    required = {
        "path",
        "filename",
        "team",
        "date",
        "player_id",
    }

    missing = (
        required
        - set(manifest.columns)
    )

    if missing:

        raise ValueError(
            "Manifest missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    selected_ids = set(
        cohort[
            "player_id"
        ]
    )

    result = manifest[
        (
            manifest[
                "team"
            ]
            == TARGET_TEAM
        )
        &
        (
            manifest[
                "player_id"
            ].isin(
                selected_ids
            )
        )
    ].copy()

    result[
        "date"
    ] = pd.to_datetime(
        result[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()

    result = (
        result
        .dropna(
            subset=[
                "date",
                "player_id",
                "path",
            ]
        )
        .sort_values(
            [
                "date",
                "player_id",
                "filename",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# Time parsing
# ============================================================

def parse_clock_seconds(
    series: pd.Series,
) -> pd.Series:
    """
    Convert strings like:

        11:29:49.7

    into seconds since midnight.

    pd.to_timedelta handles fractional seconds correctly.
    """

    parsed = pd.to_timedelta(
        series,
        errors="coerce",
    )

    return parsed.dt.total_seconds()


# ============================================================
# Collapse GPS repeated timestamps
# ============================================================

def collapse_timestamps(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    result = frame[
        [
            "time",
            "speed",
        ]
    ].copy()

    result[
        "speed"
    ] = pd.to_numeric(
        result[
            "speed"
        ],
        errors="coerce",
    )

    # GPS/speed values repeat inside timestamp groups.
    #
    # Audit 06 showed that speed does not vary within these
    # repeated timestamp groups, so keeping the first row is
    # sufficient.
    result = (
        result
        .drop_duplicates(
            subset=[
                "time"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "seconds"
    ] = parse_clock_seconds(
        result[
            "time"
        ]
    )

    result = result.dropna(
        subset=[
            "seconds",
            "speed",
        ]
    )

    return result


# ============================================================
# Session feature extraction
# ============================================================

def extract_session_features(
    path: Path,
) -> dict:

    # --------------------------------------------------------
    # Only load columns needed for the paper-inspired baseline.
    #
    # This is critical because individual objective files can
    # contain hundreds of thousands to >1 million raw rows.
    # --------------------------------------------------------

    raw = pd.read_parquet(
        path,
        columns=[
            "time",
            "speed",
        ],
    )

    raw_rows = len(
        raw
    )

    gps = collapse_timestamps(
        raw
    )

    unique_timestamps = len(
        gps
    )

    if unique_timestamps < 2:

        raise ValueError(
            "Fewer than two valid unique timestamps."
        )

    # --------------------------------------------------------
    # Intervals
    #
    # Each timestamp represents a speed observation.
    # dt is time to the next unique timestamp.
    # --------------------------------------------------------

    gps[
        "dt"
    ] = (
        gps[
            "seconds"
        ]
        .shift(-1)
        - gps[
            "seconds"
        ]
    )

    # --------------------------------------------------------
    # Handle possible midnight crossing.
    # --------------------------------------------------------

    midnight_mask = (
        gps[
            "dt"
        ]
        < 0
    )

    gps.loc[
        midnight_mask,
        "dt",
    ] = (
        gps.loc[
            midnight_mask,
            "dt",
        ]
        + 86400.0
    )

    # --------------------------------------------------------
    # Valid integration intervals
    #
    # Do not integrate over large timestamp gaps.
    # --------------------------------------------------------

    valid_interval = (
        gps[
            "dt"
        ].notna()
        &
        (
            gps[
                "dt"
            ]
            > 0
        )
        &
        (
            gps[
                "dt"
            ]
            <= MAX_VALID_INTERVAL_SECONDS
        )
    )

    gps[
        "valid_interval"
    ] = valid_interval

    valid = gps[
        valid_interval
    ].copy()

    if valid.empty:

        raise ValueError(
            "No valid timestamp intervals."
        )

    # --------------------------------------------------------
    # Session timing
    # --------------------------------------------------------

    first_second = (
        gps[
            "seconds"
        ]
        .iloc[0]
    )

    last_second = (
        gps[
            "seconds"
        ]
        .iloc[-1]
    )

    raw_clock_duration_sec = (
        last_second
        - first_second
    )

    if raw_clock_duration_sec < 0:

        raw_clock_duration_sec += (
            86400.0
        )

    # Valid sampled duration excludes long recording gaps.
    sampled_duration_sec = (
        valid[
            "dt"
        ].sum()
    )

    duration_min = (
        sampled_duration_sec
        / 60.0
    )


    # --------------------------------------------------------
    # Speed statistics
    #
    # Use unique timestamp-level observations rather than raw
    # repeated rows.
    # --------------------------------------------------------

    speed = gps[
        "speed"
    ]

    speed_mean_mps = float(
        speed.mean()
    )

    speed_max_mps = float(
        speed.max()
    )

    speed_std_mps = float(
        speed.std(
            ddof=0
        )
    )

    speed_mean_kmh = (
        speed_mean_mps
        * MPS_TO_KMH
    )

    speed_max_kmh = (
        speed_max_mps
        * MPS_TO_KMH
    )

    speed_std_kmh = (
        speed_std_mps
        * MPS_TO_KMH
    )


    # --------------------------------------------------------
    # Distance
    #
    # For each valid interval:
    #
    # distance = speed * elapsed time
    #
    # Speed is in m/s and dt is seconds.
    # --------------------------------------------------------

    valid[
        "distance_m"
    ] = (
        valid[
            "speed"
        ]
        * valid[
            "dt"
        ]
    )

    total_distance_m = float(
        valid[
            "distance_m"
        ]
        .sum()
    )

    total_distance_km = (
        total_distance_m
        / 1000.0
    )

    if duration_min > 0:

        distance_per_min_m = (
            total_distance_m
            / duration_min
        )

    else:

        distance_per_min_m = (
            float("nan")
        )


    # ========================================================
    # Running intensity zones
    # ========================================================

    speed_valid = valid[
        "speed"
    ]

    low_mask = (
        speed_valid
        < MEDIUM_SPEED_MIN_MPS
    )

    medium_mask = (
        (
            speed_valid
            >= MEDIUM_SPEED_MIN_MPS
        )
        &
        (
            speed_valid
            <= HIGH_SPEED_MIN_MPS
        )
    )

    high_mask = (
        (
            speed_valid
            > HIGH_SPEED_MIN_MPS
        )
        &
        (
            speed_valid
            <= SPRINT_SPEED_MIN_MPS
        )
    )

    sprint_mask = (
        speed_valid
        > SPRINT_SPEED_MIN_MPS
    )


    def zone_values(
        mask: pd.Series,
    ) -> tuple[
        float,
        float,
        float,
    ]:

        zone_time_sec = float(
            valid.loc[
                mask,
                "dt",
            ]
            .sum()
        )

        zone_distance_m = float(
            valid.loc[
                mask,
                "distance_m",
            ]
            .sum()
        )

        if sampled_duration_sec > 0:

            zone_proportion = (
                zone_time_sec
                / sampled_duration_sec
            )

        else:

            zone_proportion = (
                float("nan")
            )

        return (
            zone_proportion,
            zone_time_sec,
            zone_distance_m,
        )


    (
        low_p,
        low_t,
        low_d,
    ) = zone_values(
        low_mask
    )

    (
        medium_p,
        medium_t,
        medium_d,
    ) = zone_values(
        medium_mask
    )

    (
        high_p,
        high_t,
        high_d,
    ) = zone_values(
        high_mask
    )

    (
        sprint_p,
        sprint_t,
        sprint_d,
    ) = zone_values(
        sprint_mask
    )


    # --------------------------------------------------------
    # Gap diagnostics
    # --------------------------------------------------------

    interval_count = int(
        gps[
            "dt"
        ]
        .notna()
        .sum()
    )

    valid_interval_count = int(
        valid_interval.sum()
    )

    excluded_interval_count = (
        interval_count
        - valid_interval_count
    )

    if interval_count > 0:

        excluded_interval_fraction = (
            excluded_interval_count
            / interval_count
        )

    else:

        excluded_interval_fraction = (
            float("nan")
        )


    return {
        "raw_rows":
        raw_rows,

        "unique_timestamps":
        unique_timestamps,

        "raw_clock_duration_seconds":
        raw_clock_duration_sec,

        "sampled_duration_seconds":
        sampled_duration_sec,

        "duration_obj_min":
        duration_min,

        "speed_mean_mps":
        speed_mean_mps,

        "speed_max_mps":
        speed_max_mps,

        "speed_std_mps":
        speed_std_mps,

        "Speed_km_h_mean":
        speed_mean_kmh,

        "Speed_km_h_max":
        speed_max_kmh,

        "Speed_km_h_std":
        speed_std_kmh,

        "Distance_m":
        total_distance_m,

        "Distance":
        total_distance_km,

        "distance_per_min":
        distance_per_min_m,

        "sp_lir_p":
        low_p,

        "sp_lir_t":
        low_t,

        "sp_lir_d":
        low_d,

        "sp_mir_p":
        medium_p,

        "sp_mir_t":
        medium_t,

        "sp_mir_d":
        medium_d,

        "sp_hir_p":
        high_p,

        "sp_hir_t":
        high_t,

        "sp_hir_d":
        high_d,

        "sp_spr_p":
        sprint_p,

        "sp_spr_t":
        sprint_t,

        "sp_spr_d":
        sprint_d,

        "timestamp_intervals":
        interval_count,

        "valid_timestamp_intervals":
        valid_interval_count,

        "excluded_timestamp_intervals":
        excluded_interval_count,

        "excluded_interval_fraction":
        excluded_interval_fraction,
    }


# ============================================================
# QA classification
# ============================================================

def assign_qa(
    row: dict,
) -> dict:

    reasons = []

    if (
        row[
            "Speed_km_h_max"
        ]
        > MAX_SPEED_KMH
    ):

        reasons.append(
            "max_speed_over_32_kmh"
        )

    if (
        row[
            "duration_obj_min"
        ]
        > MAX_DURATION_MIN
    ):

        reasons.append(
            "duration_over_200_min"
        )

    if (
        row[
            "Distance"
        ]
        > MAX_DISTANCE_KM
    ):

        reasons.append(
            "distance_over_16_km"
        )

    row[
        "qa_pass"
    ] = (
        len(
            reasons
        )
        == 0
    )

    row[
        "qa_exclusion_reason"
    ] = ";".join(
        reasons
    )

    return row


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(
    rows: list[dict],
) -> None:

    if not rows:

        return

    pd.DataFrame(
        rows
    ).to_csv(
        CHECKPOINT_FILE,
        index=False,
    )


def load_completed_files() -> set[str]:

    if not CHECKPOINT_FILE.exists():

        return set()

    checkpoint = pd.read_csv(
        CHECKPOINT_FILE
    )

    if "filename" not in checkpoint:

        return set()

    return set(
        checkpoint[
            "filename"
        ]
    )


# ============================================================
# Process objective files
# ============================================================

def process_manifest(
    manifest: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    completed = (
        load_completed_files()
    )

    if CHECKPOINT_FILE.exists():

        existing = pd.read_csv(
            CHECKPOINT_FILE
        )

        rows = existing.to_dict(
            orient="records"
        )

        print(
            f"Resuming from checkpoint: "
            f"{len(rows):,} files already recorded."
        )

    remaining = manifest[
        ~manifest[
            "filename"
        ].isin(
            completed
        )
    ]

    total = len(
        manifest
    )

    print(
        f"Objective files in cohort: "
        f"{total:,}"
    )

    print(
        f"Files remaining: "
        f"{len(remaining):,}"
    )

    print()

    processed_now = 0

    for _, manifest_row in (
        remaining.iterrows()
    ):

        path = Path(
            manifest_row[
                "path"
            ]
        )

        output = {
            "path":
            str(path),

            "filename":
            manifest_row[
                "filename"
            ],

            "team":
            manifest_row[
                "team"
            ],

            "player_id":
            manifest_row[
                "player_id"
            ],

            "date":
            manifest_row[
                "date"
            ],

            "error":
            "",
        }

        try:

            features = (
                extract_session_features(
                    path
                )
            )

            output.update(
                features
            )

            output = assign_qa(
                output
            )

        except Exception as exc:

            output[
                "error"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            output[
                "qa_pass"
            ] = False

            output[
                "qa_exclusion_reason"
            ] = (
                "processing_error"
            )

        rows.append(
            output
        )

        processed_now += 1

        if (
            processed_now
            % CHECKPOINT_EVERY_FILES
            == 0
        ):

            save_checkpoint(
                rows
            )

            print(
                f"Processed "
                f"{len(rows):,}/{total:,} files..."
            )

    save_checkpoint(
        rows
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Add player_name
# ============================================================

def add_player_names(
    sessions: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:

    names = (
        cohort[
            [
                "player_id",
                "player_name",
            ]
        ]
        .drop_duplicates()
    )

    result = sessions.merge(
        names,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    return result


# ============================================================
# Build daily objective features
# ============================================================

def build_daily(
    sessions: pd.DataFrame,
) -> pd.DataFrame:

    valid = sessions[
        (
            sessions[
                "qa_pass"
            ]
            == True
        )
        &
        (
            sessions[
                "error"
            ]
            .fillna("")
            == ""
        )
    ].copy()

    if valid.empty:

        raise ValueError(
            "No objective sessions passed QA."
        )

    valid[
        "date"
    ] = pd.to_datetime(
        valid[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()


    # --------------------------------------------------------
    # Several objective files may occur on the same player-day.
    #
    # Aggregate the objective exposure for that day.
    # --------------------------------------------------------

    valid[
        "weighted_speed_sum"
    ] = (
        valid[
            "speed_mean_mps"
        ]
        * valid[
            "unique_timestamps"
        ]
    )

    valid[
        "speed_second_moment"
    ] = (
        (
            valid[
                "speed_std_mps"
            ]
            ** 2
        )
        +
        (
            valid[
                "speed_mean_mps"
            ]
            ** 2
        )
    )

    valid[
        "weighted_speed_second_moment"
    ] = (
        valid[
            "speed_second_moment"
        ]
        * valid[
            "unique_timestamps"
        ]
    )

    grouped = (
        valid
        .groupby(
            [
                "player_name",
                "team",
                "date",
            ],
            as_index=False,
        )
        .agg(
            objective_session_files=(
                "filename",
                "size",
            ),

            objective_timestamp_count=(
                "unique_timestamps",
                "sum",
            ),

            objective_duration_seconds=(
                "sampled_duration_seconds",
                "sum",
            ),

            objective_distance_m=(
                "Distance_m",
                "sum",
            ),

            speed_weighted_sum=(
                "weighted_speed_sum",
                "sum",
            ),

            speed_weighted_second_moment=(
                "weighted_speed_second_moment",
                "sum",
            ),

            Speed_km_h_max=(
                "Speed_km_h_max",
                "max",
            ),

            sp_lir_t=(
                "sp_lir_t",
                "sum",
            ),

            sp_lir_d=(
                "sp_lir_d",
                "sum",
            ),

            sp_mir_t=(
                "sp_mir_t",
                "sum",
            ),

            sp_mir_d=(
                "sp_mir_d",
                "sum",
            ),

            sp_hir_t=(
                "sp_hir_t",
                "sum",
            ),

            sp_hir_d=(
                "sp_hir_d",
                "sum",
            ),

            sp_spr_t=(
                "sp_spr_t",
                "sum",
            ),

            sp_spr_d=(
                "sp_spr_d",
                "sum",
            ),
        )
    )


    # --------------------------------------------------------
    # Daily mean speed
    # --------------------------------------------------------

    grouped[
        "speed_mean_mps"
    ] = (
        grouped[
            "speed_weighted_sum"
        ]
        / grouped[
            "objective_timestamp_count"
        ]
    )


    # --------------------------------------------------------
    # Daily pooled speed SD
    #
    # Var(X) = E[X^2] - E[X]^2
    # --------------------------------------------------------

    second_moment = (
        grouped[
            "speed_weighted_second_moment"
        ]
        / grouped[
            "objective_timestamp_count"
        ]
    )

    variance = (
        second_moment
        - (
            grouped[
                "speed_mean_mps"
            ]
            ** 2
        )
    )

    variance = variance.clip(
        lower=0
    )

    grouped[
        "speed_std_mps"
    ] = np.sqrt(
        variance
    )


    grouped[
        "Speed_km_h_mean"
    ] = (
        grouped[
            "speed_mean_mps"
        ]
        * MPS_TO_KMH
    )

    grouped[
        "Speed_km_h_std"
    ] = (
        grouped[
            "speed_std_mps"
        ]
        * MPS_TO_KMH
    )


    grouped[
        "duration_obj"
    ] = (
        grouped[
            "objective_duration_seconds"
        ]
        / 60.0
    )


    grouped[
        "Distance"
    ] = (
        grouped[
            "objective_distance_m"
        ]
        / 1000.0
    )


    grouped[
        "distance_per_min"
    ] = np.where(
        grouped[
            "duration_obj"
        ]
        > 0,

        grouped[
            "objective_distance_m"
        ]
        / grouped[
            "duration_obj"
        ],

        np.nan,
    )


    # --------------------------------------------------------
    # Daily zone proportions
    # --------------------------------------------------------

    duration_seconds = (
        grouped[
            "objective_duration_seconds"
        ]
    )

    for zone in [
        "lir",
        "mir",
        "hir",
        "spr",
    ]:

        grouped[
            f"sp_{zone}_p"
        ] = np.where(
            duration_seconds
            > 0,

            grouped[
                f"sp_{zone}_t"
            ]
            / duration_seconds,

            np.nan,
        )


    # --------------------------------------------------------
    # Final naming / order
    # --------------------------------------------------------

    final_columns = [
        "player_name",
        "team",
        "date",

        "objective_session_files",

        "duration_obj",

        "Speed_km_h_mean",
        "Speed_km_h_max",
        "Speed_km_h_std",

        "sp_lir_p",
        "sp_lir_t",
        "sp_lir_d",

        "sp_mir_p",
        "sp_mir_t",
        "sp_mir_d",

        "sp_hir_p",
        "sp_hir_t",
        "sp_hir_d",

        "sp_spr_p",
        "sp_spr_t",
        "sp_spr_d",

        "Distance",
        "distance_per_min",
    ]

    return (
        grouped[
            final_columns
        ]
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
# Coverage
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

        subset = daily[
            daily[
                "player_name"
            ]
            == player_name
        ]

        if subset.empty:

            rows.append(
                {
                    "player_name":
                    player_name,

                    "team":
                    team,

                    "first_objective_date":
                    pd.NaT,

                    "last_objective_date":
                    pd.NaT,

                    "objective_days":
                    0,

                    "objective_files":
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

                "first_objective_date":
                subset[
                    "date"
                ].min(),

                "last_objective_date":
                subset[
                    "date"
                ].max(),

                "objective_days":
                len(
                    subset
                ),

                "objective_files":
                int(
                    subset[
                        "objective_session_files"
                    ]
                    .sum()
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
        "SoccerMon Paper-Inspired Objective Daily Builder"
    )

    print(
        "=" * 80
    )

    print()

    cohort = (
        load_selected_cohort()
    )

    print(
        f"Selected cohort players: "
        f"{len(cohort):,}"
    )

    manifest = (
        load_manifest(
            cohort
        )
    )

    print(
        f"Objective files selected: "
        f"{len(manifest):,}"
    )

    print()

    sessions = (
        process_manifest(
            manifest
        )
    )

    sessions = (
        add_player_names(
            sessions,
            cohort,
        )
    )


    # --------------------------------------------------------
    # Write complete session results
    # --------------------------------------------------------

    sessions.to_csv(
        SESSION_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # QA table
    # --------------------------------------------------------

    qa_columns = [
        "filename",
        "player_name",
        "team",
        "date",

        "Speed_km_h_max",
        "duration_obj_min",
        "Distance",

        "raw_rows",
        "unique_timestamps",

        "excluded_timestamp_intervals",
        "excluded_interval_fraction",

        "qa_pass",
        "qa_exclusion_reason",
        "error",
    ]

    available_qa = [
        column
        for column
        in qa_columns
        if column
        in sessions.columns
    ]

    sessions[
        available_qa
    ].to_csv(
        QA_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # Daily aggregation
    # --------------------------------------------------------

    daily = (
        build_daily(
            sessions
        )
    )

    daily.to_csv(
        DAILY_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    coverage = (
        build_coverage(
            daily,
            cohort,
        )
    )

    coverage.to_csv(
        COVERAGE_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    processed_files = len(
        sessions
    )

    processing_errors = int(
        (
            sessions[
                "error"
            ]
            .fillna("")
            != ""
        ).sum()
    )

    qa_passed = int(
        (
            sessions[
                "qa_pass"
            ]
            == True
        ).sum()
    )

    qa_failed = (
        processed_files
        - qa_passed
    )

    speed_failures = int(
        sessions[
            "qa_exclusion_reason"
        ]
        .fillna("")
        .str.contains(
            "max_speed_over_32_kmh"
        )
        .sum()
    )

    duration_failures = int(
        sessions[
            "qa_exclusion_reason"
        ]
        .fillna("")
        .str.contains(
            "duration_over_200_min"
        )
        .sum()
    )

    distance_failures = int(
        sessions[
            "qa_exclusion_reason"
        ]
        .fillna("")
        .str.contains(
            "distance_over_16_km"
        )
        .sum()
    )

    players_daily = (
        daily[
            "player_name"
        ]
        .nunique()
    )

    objective_days = len(
        daily
    )

    multi_file_days = int(
        (
            daily[
                "objective_session_files"
            ]
            > 1
        ).sum()
    )


    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Paper-Inspired Objective Daily Summary"
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
        f"Players represented after QA: "
        f"{players_daily:,}"
    )

    lines.append("")

    lines.append(
        "Objective files"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Files selected: "
        f"{len(manifest):,}"
    )

    lines.append(
        f"Files processed: "
        f"{processed_files:,}"
    )

    lines.append(
        f"Processing errors: "
        f"{processing_errors:,}"
    )

    lines.append("")

    lines.append(
        "Paper-inspired QA"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Sessions passing QA: "
        f"{qa_passed:,}"
    )

    lines.append(
        f"Sessions failing QA: "
        f"{qa_failed:,}"
    )

    lines.append(
        f"Max-speed exclusions: "
        f"{speed_failures:,}"
    )

    lines.append(
        f"Duration exclusions: "
        f"{duration_failures:,}"
    )

    lines.append(
        f"Distance exclusions: "
        f"{distance_failures:,}"
    )

    lines.append("")

    lines.append(
        "Daily objective table"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Player-days with objective data: "
        f"{objective_days:,}"
    )

    lines.append(
        f"Player-days containing >1 objective file: "
        f"{multi_file_days:,}"
    )

    lines.append("")

    lines.append(
        "Reconstruction assumptions"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "1. SoccerMon speed is interpreted as metres per second."
    )

    lines.append(
        "2. Repeated GPS/speed rows are collapsed to one observation "
        "per unique timestamp."
    )

    lines.append(
        "3. Timestamp intervals longer than "
        f"{MAX_VALID_INTERVAL_SECONDS:.1f} second are excluded from "
        "distance and zone-time integration."
    )

    lines.append(
        "4. Speed zones use reconstruction thresholds:"
    )

    lines.append(
        "   low < 3.33 m/s"
    )

    lines.append(
        "   medium 3.33-4.44 m/s"
    )

    lines.append(
        "   high >4.44-5.55 m/s"
    )

    lines.append(
        "   sprint >5.55 m/s"
    )

    lines.append(
        "5. Heart rate is not used in the initial paper-inspired "
        "objective feature set."
    )

    lines.append(
        "6. Accelerometer and gyroscope signals are intentionally "
        "reserved for later improved-model experiments."
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
        f"Session features written to:\n"
        f"{SESSION_FILE}"
    )

    print()

    print(
        f"QA results written to:\n"
        f"{QA_FILE}"
    )

    print()

    print(
        f"Daily objective table written to:\n"
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