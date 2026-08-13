from __future__ import annotations

# Lines 3-9: Standard-library imports
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Lines 11-12: Third-party imports
import pandas as pd


# Line 16: Resolve repository root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingLoadValidationRecord:
    player_name: str
    date: str
    session_count: int
    raw_srpe_sum: float
    deduplicated_srpe_sum: float
    provided_daily_load: float | None
    raw_difference: float | None
    deduplicated_difference: float | None
    raw_matches_provided: bool
    deduplicated_matches_provided: bool


def find_subjective_root() -> Path:
    """Locate the extracted SoccerMon subjective-data root."""
    root = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "subjective"
        / "subjective"
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Subjective dataset not found: {root}"
        )

    return root


def load_session_json(
    session_path: Path,
) -> pd.DataFrame:
    """
    Normalize session.json into one row per player session.

    Expected structure:
    player_name -> list of session records
    """
    with session_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected session.json top level to be a dictionary."
        )

    frames: list[pd.DataFrame] = []

    for player_name, sessions in data.items():
        if not isinstance(sessions, list):
            raise ValueError(
                f"Expected a list for player {player_name}"
            )

        if not sessions:
            continue

        player_frame = pd.json_normalize(
            sessions
        )

        player_frame.insert(
            0,
            "player_name",
            player_name,
        )

        frames.append(
            player_frame
        )

    if not frames:
        raise RuntimeError(
            "No session records were found."
        )

    frame = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    required_columns = {
        "player_name",
        "srpe",
        "rpe",
        "duration",
        "date",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "session.json is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return frame


def clean_sessions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Parse dates and numeric session fields."""
    cleaned = frame.copy()

    cleaned["date"] = pd.to_datetime(
        cleaned["date"],
        format="%d.%m.%Y",
        errors="coerce",
    )

    for column in [
        "srpe",
        "rpe",
        "duration",
    ]:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    if cleaned["date"].isna().any():
        bad_rows = int(
            cleaned["date"].isna().sum()
        )

        raise ValueError(
            f"{bad_rows} session rows contain invalid dates."
        )

    return cleaned


def validate_srpe_formula(
    sessions: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    """
    Verify whether sRPE equals RPE multiplied by duration.

    Does not modify source data.
    """
    result = sessions[
        [
            "player_name",
            "date",
            "srpe",
            "rpe",
            "duration",
        ]
    ].copy()

    result["calculated_srpe"] = (
        result["rpe"]
        * result["duration"]
    )

    result["srpe_difference"] = (
        result["srpe"]
        - result["calculated_srpe"]
    )

    result["formula_matches"] = (
        result[
            "srpe_difference"
        ].abs()
        <= tolerance
    )

    return result


def reshape_daily_load(
    path: Path,
) -> pd.DataFrame:
    """
    Convert daily_load.csv from wide format to long format.

    Output:
    date | player_name | provided_daily_load
    """
    frame = pd.read_csv(
        path
    )

    date_column = str(
        frame.columns[0]
    )

    frame[date_column] = pd.to_datetime(
        frame[date_column],
        errors="coerce",
    )

    player_columns = [
        column
        for column in frame.columns
        if str(column).startswith(
            ("TeamA-", "TeamB-")
        )
    ]

    long_frame = frame.melt(
        id_vars=[date_column],
        value_vars=player_columns,
        var_name="player_name",
        value_name="provided_daily_load",
    )

    long_frame = long_frame.rename(
        columns={
            date_column: "date"
        }
    )

    long_frame[
        "provided_daily_load"
    ] = pd.to_numeric(
        long_frame[
            "provided_daily_load"
        ],
        errors="coerce",
    )

    return long_frame


def aggregate_raw_sessions(
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """Sum all session sRPE values by player and date."""
    return (
        sessions
        .groupby(
            [
                "player_name",
                "date",
            ],
            as_index=False,
        )
        .agg(
            session_count=(
                "srpe",
                "size",
            ),
            raw_srpe_sum=(
                "srpe",
                "sum",
            ),
        )
    )


def aggregate_deduplicated_sessions(
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drop exact duplicate session rows before summing sRPE.

    This is for comparison only. We are NOT yet deciding that duplicates
    should actually be removed.
    """
    deduplicated = (
        sessions
        .drop_duplicates(
            subset=[
                "player_name",
                "srpe",
                "rpe",
                "duration",
                "date",
            ]
        )
        .copy()
    )

    return (
        deduplicated
        .groupby(
            [
                "player_name",
                "date",
            ],
            as_index=False,
        )
        .agg(
            deduplicated_srpe_sum=(
                "srpe",
                "sum",
            )
        )
    )


def compare_daily_load(
    raw_daily: pd.DataFrame,
    deduplicated_daily: pd.DataFrame,
    provided_daily: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    """Compare both reconstructed versions with provided daily_load."""
    comparison = raw_daily.merge(
        deduplicated_daily,
        on=[
            "player_name",
            "date",
        ],
        how="outer",
    )

    comparison = comparison.merge(
        provided_daily,
        on=[
            "player_name",
            "date",
        ],
        how="left",
    )

    comparison[
        "session_count"
    ] = (
        comparison["session_count"]
        .fillna(0)
        .astype(int)
    )

    comparison[
        "raw_srpe_sum"
    ] = (
        comparison["raw_srpe_sum"]
        .fillna(0.0)
    )

    comparison[
        "deduplicated_srpe_sum"
    ] = (
        comparison[
            "deduplicated_srpe_sum"
        ]
        .fillna(0.0)
    )

    comparison[
        "raw_difference"
    ] = (
        comparison[
            "raw_srpe_sum"
        ]
        - comparison[
            "provided_daily_load"
        ]
    )

    comparison[
        "deduplicated_difference"
    ] = (
        comparison[
            "deduplicated_srpe_sum"
        ]
        - comparison[
            "provided_daily_load"
        ]
    )

    comparison[
        "raw_matches_provided"
    ] = (
        comparison[
            "raw_difference"
        ].abs()
        <= tolerance
    )

    comparison[
        "deduplicated_matches_provided"
    ] = (
        comparison[
            "deduplicated_difference"
        ].abs()
        <= tolerance
    )

    return comparison


def write_summary(
    sessions: pd.DataFrame,
    srpe_validation: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a readable validation summary."""
    formula_matches = int(
        srpe_validation[
            "formula_matches"
        ].sum()
    )

    formula_failures = int(
        (
            ~srpe_validation[
                "formula_matches"
            ]
        ).sum()
    )

    valid_comparisons = comparison[
        comparison[
            "provided_daily_load"
        ].notna()
    ]

    raw_matches = int(
        valid_comparisons[
            "raw_matches_provided"
        ].sum()
    )

    dedup_matches = int(
        valid_comparisons[
            "deduplicated_matches_provided"
        ].sum()
    )

    duplicate_session_rows = int(
        sessions.duplicated(
            subset=[
                "player_name",
                "srpe",
                "rpe",
                "duration",
                "date",
            ]
        ).sum()
    )

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append(
        "SoccerMon Training Load Validation"
    )
    lines.append("=" * 80)

    lines.append(
        f"Session records: "
        f"{len(sessions):,}"
    )

    lines.append(
        f"Exact duplicate session rows "
        f"(excluding first occurrence): "
        f"{duplicate_session_rows:,}"
    )

    lines.append("")
    lines.append(
        "sRPE formula validation"
    )
    lines.append("-" * 80)

    lines.append(
        f"sRPE == RPE × duration: "
        f"{formula_matches:,}/"
        f"{len(srpe_validation):,}"
    )

    lines.append(
        f"Formula mismatches: "
        f"{formula_failures:,}"
    )

    lines.append("")
    lines.append(
        "Daily-load comparison"
    )
    lines.append("-" * 80)

    lines.append(
        f"Player-days compared with provided daily_load: "
        f"{len(valid_comparisons):,}"
    )

    lines.append(
        f"Raw session sums matching provided daily_load: "
        f"{raw_matches:,}/"
        f"{len(valid_comparisons):,}"
    )

    if len(valid_comparisons) > 0:
        lines.append(
            f"Raw match percentage: "
            f"{raw_matches / len(valid_comparisons) * 100:.2f}%"
        )

    lines.append(
        f"Deduplicated sums matching provided daily_load: "
        f"{dedup_matches:,}/"
        f"{len(valid_comparisons):,}"
    )

    if len(valid_comparisons) > 0:
        lines.append(
            f"Deduplicated match percentage: "
            f"{dedup_matches / len(valid_comparisons) * 100:.2f}%"
        )

    lines.append("")
    lines.append(
        "Interpretation"
    )
    lines.append("-" * 80)

    if raw_matches > dedup_matches:
        lines.append(
            "The provided daily_load agrees more often with the raw "
            "session data than with exact duplicates removed."
        )

    elif dedup_matches > raw_matches:
        lines.append(
            "The provided daily_load agrees more often after exact "
            "duplicate session rows are removed."
        )

    else:
        lines.append(
            "Raw and deduplicated reconstructions match the provided "
            "daily_load equally often."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(lines)
    )

    print()
    print(
        f"Summary written to: "
        f"{output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate SoccerMon sRPE calculations and compare "
            "session-derived daily load with provided daily_load.csv."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "audit"
        ),
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help=(
            "Absolute tolerance when comparing numeric values."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subjective_root = (
        find_subjective_root()
    )

    session_path = (
        subjective_root
        / "training-load"
        / "session.json"
    )

    daily_load_path = (
        subjective_root
        / "training-load"
        / "daily_load.csv"
    )

    print(
        f"Loading sessions: "
        f"{session_path}"
    )

    sessions = load_session_json(
        session_path
    )

    sessions = clean_sessions(
        sessions
    )

    print(
        f"Loaded {len(sessions):,} session records."
    )

    srpe_validation = (
        validate_srpe_formula(
            sessions,
            args.tolerance,
        )
    )

    raw_daily = (
        aggregate_raw_sessions(
            sessions
        )
    )

    deduplicated_daily = (
        aggregate_deduplicated_sessions(
            sessions
        )
    )

    provided_daily = (
        reshape_daily_load(
            daily_load_path
        )
    )

    comparison = compare_daily_load(
        raw_daily=raw_daily,
        deduplicated_daily=(
            deduplicated_daily
        ),
        provided_daily=(
            provided_daily
        ),
        tolerance=args.tolerance,
    )

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    srpe_path = (
        output_dir
        / "srpe_formula_validation.csv"
    )

    daily_comparison_path = (
        output_dir
        / "daily_load_validation.csv"
    )

    mismatch_path = (
        output_dir
        / "daily_load_mismatches.csv"
    )

    summary_path = (
        output_dir
        / "training_load_validation_summary.txt"
    )

    srpe_validation.to_csv(
        srpe_path,
        index=False,
    )

    comparison.to_csv(
        daily_comparison_path,
        index=False,
    )

    comparison[
        (
            ~comparison[
                "raw_matches_provided"
            ]
        )
        |
        (
            ~comparison[
                "deduplicated_matches_provided"
            ]
        )
    ].to_csv(
        mismatch_path,
        index=False,
    )

    write_summary(
        sessions=sessions,
        srpe_validation=(
            srpe_validation
        ),
        comparison=comparison,
        output_path=(
            summary_path
        ),
    )

    print(
        f"sRPE validation written to: "
        f"{srpe_path}"
    )

    print(
        f"Daily-load comparison written to: "
        f"{daily_comparison_path}"
    )

    print(
        f"Mismatches written to: "
        f"{mismatch_path}"
    )


if __name__ == "__main__":
    main()