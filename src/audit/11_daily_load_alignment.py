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
class AlignmentSummary:
    alignment_name: str
    compared_rows: int
    exact_matches: int
    match_percentage: float
    mean_absolute_difference: float | None
    median_absolute_difference: float | None


def find_subjective_root() -> Path:
    """Locate the extracted SoccerMon subjective root."""
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
    """Normalize session.json into one row per player session."""
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

    frame["srpe"] = pd.to_numeric(
        frame["srpe"],
        errors="coerce",
    )

    return frame


def aggregate_sessions(
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """Sum raw sRPE values by player and date."""
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
            reconstructed_daily_load=(
                "srpe",
                "sum",
            ),
            session_count=(
                "srpe",
                "size",
            ),
        )
    )


def load_provided_daily_load(
    path: Path,
) -> pd.DataFrame:
    """Convert provided daily_load.csv from wide to long format."""
    frame = pd.read_csv(
        path
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


def compare_fixed_offset(
    reconstructed: pd.DataFrame,
    provided: pd.DataFrame,
    offset_days: int,
    tolerance: float,
) -> tuple[pd.DataFrame, AlignmentSummary]:
    """
    Compare reconstructed session load against provided daily load
    after shifting the provided date by a fixed number of days.
    """
    shifted = provided.copy()

    shifted["comparison_date"] = (
        shifted["date"]
        + pd.to_timedelta(
            offset_days,
            unit="D",
        )
    )

    merged = reconstructed.merge(
        shifted[
            [
                "player_name",
                "comparison_date",
                "provided_daily_load",
                "date",
            ]
        ],
        left_on=[
            "player_name",
            "date",
        ],
        right_on=[
            "player_name",
            "comparison_date",
        ],
        how="inner",
        suffixes=(
            "_reconstructed",
            "_provided",
        ),
    )

    merged = merged[
        merged[
            "provided_daily_load"
        ].notna()
    ].copy()

    merged["difference"] = (
        merged[
            "reconstructed_daily_load"
        ]
        - merged[
            "provided_daily_load"
        ]
    )

    merged["absolute_difference"] = (
        merged["difference"].abs()
    )

    merged["matches"] = (
        merged[
            "absolute_difference"
        ]
        <= tolerance
    )

    compared_rows = len(
        merged
    )

    exact_matches = int(
        merged["matches"].sum()
    )

    match_percentage = (
        exact_matches
        / compared_rows
        * 100
        if compared_rows > 0
        else 0.0
    )

    mean_absolute_difference = (
        float(
            merged[
                "absolute_difference"
            ].mean()
        )
        if compared_rows > 0
        else None
    )

    median_absolute_difference = (
        float(
            merged[
                "absolute_difference"
            ].median()
        )
        if compared_rows > 0
        else None
    )

    if offset_days == 0:
        alignment_name = "same_date"
    elif offset_days > 0:
        alignment_name = (
            f"provided_date_plus_{offset_days}_days"
        )
    else:
        alignment_name = (
            f"provided_date_minus_{abs(offset_days)}_days"
        )

    summary = AlignmentSummary(
        alignment_name=alignment_name,
        compared_rows=compared_rows,
        exact_matches=exact_matches,
        match_percentage=match_percentage,
        mean_absolute_difference=(
            mean_absolute_difference
        ),
        median_absolute_difference=(
            median_absolute_difference
        ),
    )

    return merged, summary


def compare_available_day_alignment(
    reconstructed: pd.DataFrame,
    provided: pd.DataFrame,
    direction: str,
    tolerance: float,
) -> tuple[pd.DataFrame, AlignmentSummary]:
    """
    Compare each reconstructed session day with the nearest available
    provided day for the same player.

    direction:
        backward -> previous available provided date
        forward  -> next available provided date
    """
    left = (
        reconstructed
        .sort_values(
            [
                "player_name",
                "date",
            ]
        )
        .copy()
    )

    right = (
        provided[
            provided[
                "provided_daily_load"
            ].notna()
            & provided[
                "date"
            ].notna()
        ]
        .sort_values(
            [
                "player_name",
                "date",
            ]
        )
        .copy()
    )

    right = right.rename(
        columns={
            "date": "provided_date"
        }
    )

    merged_parts: list[
        pd.DataFrame
    ] = []

    for player_name, left_player in (
        left.groupby(
            "player_name",
            sort=False,
        )
    ):
        right_player = right[
            right[
                "player_name"
            ]
            == player_name
        ].copy()

        if right_player.empty:
            continue

        left_player = (
            left_player
            .sort_values("date")
            .copy()
        )

        right_player = (
            right_player
            .sort_values(
                "provided_date"
            )
            .copy()
        )

        merged_player = pd.merge_asof(
            left_player,
            right_player,
            left_on="date",
            right_on="provided_date",
            direction=direction,
            allow_exact_matches=True,
        )

        merged_parts.append(
            merged_player
        )

    if merged_parts:
        merged = pd.concat(
            merged_parts,
            ignore_index=True,
        )
    else:
        merged = pd.DataFrame()

    if not merged.empty:
        merged = merged[
            merged[
                "provided_daily_load"
            ].notna()
        ].copy()

        merged[
            "difference"
        ] = (
            merged[
                "reconstructed_daily_load"
            ]
            - merged[
                "provided_daily_load"
            ]
        )

        merged[
            "absolute_difference"
        ] = (
            merged[
                "difference"
            ].abs()
        )

        merged[
            "matches"
        ] = (
            merged[
                "absolute_difference"
            ]
            <= tolerance
        )

        merged[
            "date_gap_days"
        ] = (
            merged[
                "provided_date"
            ]
            - merged[
                "date"
            ]
        ).dt.days

    compared_rows = len(
        merged
    )

    exact_matches = (
        int(
            merged[
                "matches"
            ].sum()
        )
        if compared_rows > 0
        else 0
    )

    match_percentage = (
        exact_matches
        / compared_rows
        * 100
        if compared_rows > 0
        else 0.0
    )

    summary = AlignmentSummary(
        alignment_name=(
            "previous_available_provided_day"
            if direction == "backward"
            else "next_available_provided_day"
        ),
        compared_rows=compared_rows,
        exact_matches=exact_matches,
        match_percentage=match_percentage,
        mean_absolute_difference=(
            float(
                merged[
                    "absolute_difference"
                ].mean()
            )
            if compared_rows > 0
            else None
        ),
        median_absolute_difference=(
            float(
                merged[
                    "absolute_difference"
                ].median()
            )
            if compared_rows > 0
            else None
        ),
    )

    return merged, summary


def write_summary(
    summaries: list[AlignmentSummary],
    output_path: Path,
) -> None:
    """Write readable alignment results."""
    ranked = sorted(
        summaries,
        key=lambda record: (
            record.match_percentage,
            -(
                record.mean_absolute_difference
                if record.mean_absolute_difference
                is not None
                else float("inf")
            ),
        ),
        reverse=True,
    )

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append(
        "SoccerMon Daily Load Date-Alignment Audit"
    )
    lines.append("=" * 80)

    lines.append("")
    lines.append(
        "Alignment results"
    )
    lines.append("-" * 80)

    for record in ranked:
        lines.append(
            f"{record.alignment_name} | "
            f"matches: "
            f"{record.exact_matches:,}/"
            f"{record.compared_rows:,} | "
            f"{record.match_percentage:.2f}% | "
            f"mean abs diff: "
            f"{record.mean_absolute_difference} | "
            f"median abs diff: "
            f"{record.median_absolute_difference}"
        )

    if ranked:
        best = ranked[0]

        lines.append("")
        lines.append(
            "Best alignment"
        )
        lines.append("-" * 80)

        lines.append(
            f"{best.alignment_name}"
        )

        lines.append(
            f"Exact match percentage: "
            f"{best.match_percentage:.2f}%"
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
            "Test whether provided SoccerMon daily_load values "
            "are shifted relative to session-derived sRPE sums."
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
        "Loading session data..."
    )

    sessions = load_session_json(
        session_path
    )

    reconstructed = (
        aggregate_sessions(
            sessions
        )
    )

    provided = (
        load_provided_daily_load(
            daily_load_path
        )
    )

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[
        AlignmentSummary
    ] = []

    # Test fixed calendar-day offsets.
    for offset_days in [
        -2,
        -1,
        0,
        1,
        2,
    ]:
        comparison, summary = (
            compare_fixed_offset(
                reconstructed=(
                    reconstructed
                ),
                provided=(
                    provided
                ),
                offset_days=(
                    offset_days
                ),
                tolerance=(
                    args.tolerance
                ),
            )
        )

        summaries.append(
            summary
        )

        comparison.to_csv(
            output_dir
            / (
                "daily_load_alignment_"
                f"{summary.alignment_name}.csv"
            ),
            index=False,
        )

    # Test nearest available provided days.
    for direction in [
        "backward",
        "forward",
    ]:
        comparison, summary = (
            compare_available_day_alignment(
                reconstructed=(
                    reconstructed
                ),
                provided=(
                    provided
                ),
                direction=(
                    direction
                ),
                tolerance=(
                    args.tolerance
                ),
            )
        )

        summaries.append(
            summary
        )

        comparison.to_csv(
            output_dir
            / (
                "daily_load_alignment_"
                f"{summary.alignment_name}.csv"
            ),
            index=False,
        )

    summary_csv_path = (
        output_dir
        / "daily_load_alignment_summary.csv"
    )

    summary_txt_path = (
        output_dir
        / "daily_load_alignment_summary.txt"
    )

    pd.DataFrame(
        [
            asdict(record)
            for record in summaries
        ]
    ).to_csv(
        summary_csv_path,
        index=False,
    )

    write_summary(
        summaries,
        summary_txt_path,
    )

    print(
        f"Alignment summary CSV written to: "
        f"{summary_csv_path}"
    )


if __name__ == "__main__":
    main()