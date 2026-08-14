from __future__ import annotations

# Lines 3-9: Standard-library imports
import argparse
import json
from collections import Counter
from pathlib import Path

# Lines 11-12: Third-party imports
import pandas as pd


# Line 16: Resolve repository root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_subjective_root() -> Path:
    """Locate the SoccerMon subjective-data directory."""
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


def load_sessions(
    session_path: Path,
) -> pd.DataFrame:
    """Normalize session.json into one row per session."""
    with session_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    frames: list[pd.DataFrame] = []

    for player_name, sessions in data.items():
        if not isinstance(sessions, list):
            continue

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

    frame["date"] = pd.to_datetime(
        frame["date"],
        format="%d.%m.%Y",
        errors="coerce",
    )

    for column in [
        "srpe",
        "rpe",
        "duration",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    return frame


def load_daily_load(
    daily_load_path: Path,
) -> pd.DataFrame:
    """Convert provided daily_load.csv from wide to long format."""
    frame = pd.read_csv(
        daily_load_path
    )

    date_column = str(
        frame.columns[0]
    )

    frame[date_column] = pd.to_datetime(
        frame[date_column],
        format="%d.%m.%Y",
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


def build_daily_session_summary(
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """Create rich session-level summaries for each player-day."""
    rows: list[dict[str, object]] = []

    grouped = sessions.groupby(
        [
            "player_name",
            "date",
        ],
        sort=False,
    )

    for (
        player_name,
        date,
    ), group in grouped:

        raw_sum = float(
            group["srpe"].sum()
        )

        deduplicated = (
            group.drop_duplicates(
                subset=[
                    "srpe",
                    "rpe",
                    "duration",
                ]
            )
        )

        deduplicated_sum = float(
            deduplicated[
                "srpe"
            ].sum()
        )

        session_values = (
            group["srpe"]
            .dropna()
            .astype(float)
            .tolist()
        )

        duplicate_count = int(
            group.duplicated(
                subset=[
                    "srpe",
                    "rpe",
                    "duration",
                ]
            ).sum()
        )

        rows.append(
            {
                "player_name": player_name,
                "date": date,
                "session_count": len(group),
                "raw_srpe_sum": raw_sum,
                "deduplicated_srpe_sum": (
                    deduplicated_sum
                ),
                "duplicate_session_count": (
                    duplicate_count
                ),
                "session_srpe_values": ";".join(
                    str(value)
                    for value in session_values
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def classify_row(
    row: pd.Series,
    tolerance: float,
) -> str:
    """Classify the reason for a daily-load mismatch."""
    provided = row[
        "provided_daily_load"
    ]

    raw_sum = row[
        "raw_srpe_sum"
    ]

    dedup_sum = row[
        "deduplicated_srpe_sum"
    ]

    duplicate_count = row[
        "duplicate_session_count"
    ]

    session_values = [
        float(value)
        for value in str(
            row["session_srpe_values"]
        ).split(";")
        if value != ""
    ]

    if pd.isna(provided):
        return "provided_missing"

    if abs(
        raw_sum
        - provided
    ) <= tolerance:
        return "exact_match"

    if (
        duplicate_count > 0
        and abs(
            dedup_sum
            - provided
        ) <= tolerance
    ):
        return "explained_by_duplicate_removal"

    if (
        provided == 0
        and raw_sum > 0
    ):
        return (
            "provided_zero_reconstructed_positive"
        )

    if (
        raw_sum == 0
        and provided > 0
    ):
        return (
            "reconstructed_zero_provided_positive"
        )

    if any(
        abs(
            session_value
            - provided
        ) <= tolerance
        for session_value in session_values
    ):
        return "provided_equals_single_session"

    difference = (
        raw_sum
        - provided
    )

    if any(
        abs(
            abs(difference)
            - session_value
        ) <= tolerance
        for session_value in session_values
    ):
        return "difference_equals_one_session"

    if abs(
        difference
    ) <= 1:
        return "small_rounding_difference"

    if abs(
        difference
    ) <= 10:
        return "small_difference_under_10"

    return "unexplained"


def write_summary(
    mismatches: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write human-readable mismatch-category report."""
    counts = Counter(
        mismatches[
            "mismatch_category"
        ]
    )

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append(
        "SoccerMon Daily Load Mismatch Classification"
    )
    lines.append("=" * 80)

    lines.append(
        f"Mismatch player-days analyzed: "
        f"{len(mismatches):,}"
    )

    lines.append("")
    lines.append(
        "Category counts"
    )
    lines.append("-" * 80)

    for category, count in (
        counts.most_common()
    ):
        percentage = (
            count
            / len(mismatches)
            * 100
            if len(mismatches) > 0
            else 0
        )

        lines.append(
            f"{category}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    lines.append("")
    lines.append(
        "Interpretation"
    )
    lines.append("-" * 80)

    duplicate_explained = counts.get(
        "explained_by_duplicate_removal",
        0,
    )

    lines.append(
        "Mismatches fully explained by removing "
        f"exact duplicate sessions: "
        f"{duplicate_explained:,}"
    )

    zero_provided = counts.get(
        "provided_zero_reconstructed_positive",
        0,
    )

    lines.append(
        "Provided load is zero despite positive "
        f"session-derived load: "
        f"{zero_provided:,}"
    )

    single_session = counts.get(
        "provided_equals_single_session",
        0,
    )

    lines.append(
        "Provided load equals one individual session "
        f"instead of the daily sum: "
        f"{single_session:,}"
    )

    unexplained = counts.get(
        "unexplained",
        0,
    )

    lines.append(
        f"Still unexplained: "
        f"{unexplained:,}"
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
            "Categorize SoccerMon player-days where "
            "session-derived daily load differs from "
            "the provided daily_load.csv value."
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
        "Loading session.json..."
    )

    sessions = load_sessions(
        session_path
    )

    print(
        f"Session records loaded: "
        f"{len(sessions):,}"
    )

    session_daily = (
        build_daily_session_summary(
            sessions
        )
    )

    provided = load_daily_load(
        daily_load_path
    )

    comparison = session_daily.merge(
        provided,
        on=[
            "player_name",
            "date",
        ],
        how="left",
    )

    comparison[
        "difference"
    ] = (
        comparison[
            "raw_srpe_sum"
        ]
        - comparison[
            "provided_daily_load"
        ]
    )

    comparison[
        "absolute_difference"
    ] = (
        comparison[
            "difference"
        ].abs()
    )

    comparison[
        "mismatch_category"
    ] = comparison.apply(
        classify_row,
        axis=1,
        tolerance=args.tolerance,
    )

    mismatches = comparison[
        comparison[
            "mismatch_category"
        ]
        != "exact_match"
    ].copy()

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_path = (
        output_dir
        / "daily_load_mismatch_classification.csv"
    )

    unexplained_path = (
        output_dir
        / "daily_load_unexplained_mismatches.csv"
    )

    summary_path = (
        output_dir
        / "daily_load_mismatch_summary.txt"
    )

    mismatches.to_csv(
        all_path,
        index=False,
    )

    mismatches[
        mismatches[
            "mismatch_category"
        ]
        == "unexplained"
    ].to_csv(
        unexplained_path,
        index=False,
    )

    write_summary(
        mismatches,
        summary_path,
    )

    print(
        f"Mismatch classification written to: "
        f"{all_path}"
    )

    print(
        f"Unexplained mismatches written to: "
        f"{unexplained_path}"
    )


if __name__ == "__main__":
    main()