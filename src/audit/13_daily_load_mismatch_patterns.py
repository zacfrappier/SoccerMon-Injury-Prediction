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
    """Convert daily_load.csv from wide to long format."""
    frame = pd.read_csv(
        daily_load_path
    )

    date_column = str(
        frame.columns[0]
    )

    # Important:
    # SoccerMon uses DD.MM.YYYY.
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


def build_session_daily(
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate session.json to one row per player-day."""
    rows: list[dict[str, object]] = []

    grouped = sessions.groupby(
        [
            "player_name",
            "date",
        ],
        sort=True,
    )

    for (
        player_name,
        date,
    ), group in grouped:

        srpe_values = (
            group["srpe"]
            .dropna()
            .astype(float)
            .tolist()
        )

        rows.append(
            {
                "player_name": player_name,
                "date": date,
                "session_count": len(group),
                "reconstructed_daily_load": float(
                    group["srpe"].sum()
                ),
                "session_srpe_values": ";".join(
                    str(value)
                    for value in srpe_values
                ),
                "session_rpe_values": ";".join(
                    str(value)
                    for value in (
                        group["rpe"]
                        .dropna()
                        .tolist()
                    )
                ),
                "session_duration_values": ";".join(
                    str(value)
                    for value in (
                        group["duration"]
                        .dropna()
                        .tolist()
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_comparison(
    session_daily: pd.DataFrame,
    provided: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    """Join reconstructed and provided daily load."""
    comparison = session_daily.merge(
        provided,
        on=[
            "player_name",
            "date",
        ],
        how="left",
    )

    comparison["difference"] = (
        comparison[
            "provided_daily_load"
        ]
        - comparison[
            "reconstructed_daily_load"
        ]
    )

    comparison[
        "absolute_difference"
    ] = (
        comparison[
            "difference"
        ].abs()
    )

    comparison["matches"] = (
        comparison[
            "absolute_difference"
        ]
        <= tolerance
    )

    comparison["weekday"] = (
        comparison[
            "date"
        ].dt.day_name()
    )

    comparison["year"] = (
        comparison[
            "date"
        ].dt.year
    )

    comparison["month"] = (
        comparison[
            "date"
        ].dt.to_period("M").astype(str)
    )

    return comparison


def create_mismatch_windows(
    comparison: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """
    Create a +/- N-day context window around every mismatch.

    This lets us inspect nearby provided and reconstructed values.
    """
    mismatches = comparison[
        ~comparison["matches"]
        & comparison[
            "provided_daily_load"
        ].notna()
    ].copy()

    affected_players = (
        mismatches[
            "player_name"
        ]
        .drop_duplicates()
        .tolist()
    )

    rows: list[
        pd.DataFrame
    ] = []

    for player_name in affected_players:

        player_all = comparison[
            comparison[
                "player_name"
            ]
            == player_name
        ].copy()

        player_mismatches = mismatches[
            mismatches[
                "player_name"
            ]
            == player_name
        ]

        for mismatch_date in (
            player_mismatches[
                "date"
            ]
        ):
            start_date = (
                mismatch_date
                - pd.Timedelta(
                    days=window_days
                )
            )

            end_date = (
                mismatch_date
                + pd.Timedelta(
                    days=window_days
                )
            )

            window = player_all[
                (
                    player_all[
                        "date"
                    ]
                    >= start_date
                )
                &
                (
                    player_all[
                        "date"
                    ]
                    <= end_date
                )
            ].copy()

            window.insert(
                0,
                "mismatch_date",
                mismatch_date,
            )

            window[
                "days_from_mismatch"
            ] = (
                window["date"]
                - mismatch_date
            ).dt.days

            rows.append(
                window
            )

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


def create_player_summary(
    mismatches: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize mismatch behavior by affected player."""
    rows: list[
        dict[str, object]
    ] = []

    for (
        player_name,
        group,
    ) in mismatches.groupby(
        "player_name"
    ):

        differences = (
            group["difference"]
            .dropna()
        )

        rows.append(
            {
                "player_name": player_name,
                "mismatch_count": len(group),
                "first_mismatch": (
                    group["date"].min()
                ),
                "last_mismatch": (
                    group["date"].max()
                ),
                "mean_difference": (
                    float(
                        differences.mean()
                    )
                    if not differences.empty
                    else None
                ),
                "median_difference": (
                    float(
                        differences.median()
                    )
                    if not differences.empty
                    else None
                ),
                "minimum_difference": (
                    float(
                        differences.min()
                    )
                    if not differences.empty
                    else None
                ),
                "maximum_difference": (
                    float(
                        differences.max()
                    )
                    if not differences.empty
                    else None
                ),
                "provided_greater_count": int(
                    (
                        differences > 0
                    ).sum()
                ),
                "reconstructed_greater_count": int(
                    (
                        differences < 0
                    ).sum()
                ),
                "equal_count": int(
                    (
                        differences == 0
                    ).sum()
                ),
                "most_common_weekday": (
                    group[
                        "weekday"
                    ]
                    .value_counts()
                    .index[0]
                    if not group.empty
                    else ""
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def create_weekday_summary(
    mismatches: pd.DataFrame,
) -> pd.DataFrame:
    """Count mismatches by player and weekday."""
    return (
        mismatches
        .groupby(
            [
                "player_name",
                "weekday",
            ],
            as_index=False,
        )
        .agg(
            mismatch_count=(
                "date",
                "size",
            ),
            mean_difference=(
                "difference",
                "mean",
            ),
            median_difference=(
                "difference",
                "median",
            ),
        )
        .sort_values(
            [
                "player_name",
                "mismatch_count",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )


def create_difference_frequency(
    mismatches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine whether the same unexplained load differences recur.
    """
    frequency = (
        mismatches
        .groupby(
            [
                "player_name",
                "difference",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "occurrences"
            }
        )
        .sort_values(
            [
                "player_name",
                "occurrences",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    return frequency


def write_summary(
    mismatches: pd.DataFrame,
    player_summary: pd.DataFrame,
    weekday_summary: pd.DataFrame,
    difference_frequency: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write human-readable mismatch-pattern findings."""
    lines: list[str] = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Daily Load Mismatch Pattern Audit"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Mismatch player-days: "
        f"{len(mismatches):,}"
    )

    lines.append(
        f"Affected players: "
        f"{mismatches['player_name'].nunique():,}"
    )

    lines.append("")
    lines.append(
        "Player-level summary"
    )
    lines.append(
        "-" * 80
    )

    for _, row in (
        player_summary.iterrows()
    ):
        lines.append(
            f"{row['player_name']} | "
            f"mismatches: "
            f"{int(row['mismatch_count'])} | "
            f"range: "
            f"{row['first_mismatch']} "
            f"to "
            f"{row['last_mismatch']} | "
            f"median difference: "
            f"{row['median_difference']} | "
            f"provided greater: "
            f"{int(row['provided_greater_count'])} | "
            f"reconstructed greater: "
            f"{int(row['reconstructed_greater_count'])} | "
            f"most common weekday: "
            f"{row['most_common_weekday']}"
        )

    lines.append("")
    lines.append(
        "Weekday concentrations"
    )
    lines.append(
        "-" * 80
    )

    for _, row in (
        weekday_summary.iterrows()
    ):
        lines.append(
            f"{row['player_name']} | "
            f"{row['weekday']} | "
            f"{int(row['mismatch_count'])} mismatches | "
            f"median difference: "
            f"{row['median_difference']}"
        )

    lines.append("")
    lines.append(
        "Most frequently repeated differences"
    )
    lines.append(
        "-" * 80
    )

    for player_name in (
        mismatches[
            "player_name"
        ].unique()
    ):
        lines.append(
            f"{player_name}:"
        )

        player_differences = (
            difference_frequency[
                difference_frequency[
                    "player_name"
                ]
                == player_name
            ]
            .head(10)
        )

        for _, row in (
            player_differences.iterrows()
        ):
            lines.append(
                f"  difference "
                f"{row['difference']}: "
                f"{int(row['occurrences'])} times"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines)
        + "\n",
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
            "Analyze temporal and player-specific patterns "
            "in SoccerMon daily-load mismatches."
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
        "--window-days",
        type=int,
        default=3,
        help=(
            "Days before and after each mismatch "
            "included in the context window."
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
        "Loading SoccerMon session and "
        "daily-load data..."
    )

    sessions = load_sessions(
        session_path
    )

    session_daily = (
        build_session_daily(
            sessions
        )
    )

    provided = load_daily_load(
        daily_load_path
    )

    comparison = (
        build_comparison(
            session_daily,
            provided,
            args.tolerance,
        )
    )

    mismatches = comparison[
        ~comparison["matches"]
        & comparison[
            "provided_daily_load"
        ].notna()
    ].copy()

    print(
        f"Found {len(mismatches):,} "
        f"mismatches across "
        f"{mismatches['player_name'].nunique():,} "
        "players."
    )

    player_summary = (
        create_player_summary(
            mismatches
        )
    )

    weekday_summary = (
        create_weekday_summary(
            mismatches
        )
    )

    difference_frequency = (
        create_difference_frequency(
            mismatches
        )
    )

    windows = (
        create_mismatch_windows(
            comparison,
            args.window_days,
        )
    )

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mismatch_path = (
        output_dir
        / "daily_load_pattern_mismatches.csv"
    )

    player_path = (
        output_dir
        / "daily_load_mismatch_by_player.csv"
    )

    weekday_path = (
        output_dir
        / "daily_load_mismatch_by_weekday.csv"
    )

    difference_path = (
        output_dir
        / "daily_load_mismatch_difference_frequency.csv"
    )

    window_path = (
        output_dir
        / "daily_load_mismatch_context_windows.csv"
    )

    summary_path = (
        output_dir
        / "daily_load_mismatch_pattern_summary.txt"
    )

    mismatches.to_csv(
        mismatch_path,
        index=False,
    )

    player_summary.to_csv(
        player_path,
        index=False,
    )

    weekday_summary.to_csv(
        weekday_path,
        index=False,
    )

    difference_frequency.to_csv(
        difference_path,
        index=False,
    )

    windows.to_csv(
        window_path,
        index=False,
    )

    write_summary(
        mismatches=mismatches,
        player_summary=player_summary,
        weekday_summary=weekday_summary,
        difference_frequency=(
            difference_frequency
        ),
        output_path=summary_path,
    )


if __name__ == "__main__":
    main()