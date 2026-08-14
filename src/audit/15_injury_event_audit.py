from __future__ import annotations

# Lines 3-10: Standard-library imports
import argparse
from pathlib import Path

# Lines 12-13: Third-party imports
import pandas as pd


# Line 17: Resolve repository root automatically
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


def load_injuries(
    injury_path: Path,
) -> pd.DataFrame:
    """Load and clean injury.csv."""
    frame = pd.read_csv(
        injury_path
    )

    required_columns = {
        "player_name",
        "type",
        "timestamp",
    }

    missing = (
        required_columns
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "injury.csv is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    frame = frame.copy()

    frame["player_name"] = (
        frame["player_name"]
        .astype(str)
        .str.strip()
    )

    frame["type"] = (
        frame["type"]
        .astype(str)
        .str.strip()
    )

    # SoccerMon subjective dates use DD.MM.YYYY.
    frame["injury_date"] = pd.to_datetime(
        frame["timestamp"],
        format="%d.%m.%Y",
        errors="coerce",
    )

    frame["team"] = (
        frame["player_name"]
        .str.extract(
            r"^(Team[A-Za-z0-9]+)-",
            expand=False,
        )
    )

    return frame


def load_objective_manifest(
    manifest_path: Path,
) -> pd.DataFrame:
    """Load objective file manifest from Script 1."""
    frame = pd.read_csv(
        manifest_path
    )

    required_columns = {
        "team",
        "player_id",
        "date",
    }

    missing = (
        required_columns
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Objective manifest is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    frame = frame.copy()

    frame["player_name"] = (
        frame["team"].astype(str)
        + "-"
        + frame["player_id"].astype(str)
    )

    frame["objective_date"] = pd.to_datetime(
        frame["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    return frame


def load_overlap_table(
    overlap_path: Path,
) -> pd.DataFrame:
    """Load player-overlap output from Script 14."""
    frame = pd.read_csv(
        overlap_path
    )

    if "player_name" not in frame.columns:
        raise ValueError(
            "Player overlap table does not contain player_name."
        )

    return frame


def build_player_injury_summary(
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize injury records by player."""
    rows: list[dict[str, object]] = []

    for player_name, group in injuries.groupby(
        "player_name"
    ):
        valid_dates = (
            group["injury_date"]
            .dropna()
            .sort_values()
        )

        rows.append(
            {
                "player_name": player_name,
                "team": group[
                    "team"
                ].iloc[0],
                "injury_records": len(group),
                "unique_injury_dates": int(
                    valid_dates.nunique()
                ),
                "unique_injury_types": int(
                    group["type"]
                    .dropna()
                    .nunique()
                ),
                "first_injury_date": (
                    valid_dates.min()
                    if not valid_dates.empty
                    else pd.NaT
                ),
                "last_injury_date": (
                    valid_dates.max()
                    if not valid_dates.empty
                    else pd.NaT
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "injury_records",
            ascending=False,
        )
    )


def build_injury_type_summary(
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """Count injury type values."""
    return (
        injuries
        .groupby(
            "type",
            dropna=False,
            as_index=False,
        )
        .agg(
            injury_records=(
                "player_name",
                "size",
            ),
            unique_players=(
                "player_name",
                "nunique",
            ),
            unique_dates=(
                "injury_date",
                "nunique",
            ),
        )
        .sort_values(
            "injury_records",
            ascending=False,
        )
    )


def build_same_day_summary(
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find player-days containing more than one injury record.
    """
    grouped = (
        injuries
        .groupby(
            [
                "player_name",
                "injury_date",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            injury_records=(
                "type",
                "size",
            ),
            injury_types=(
                "type",
                lambda values: ";".join(
                    sorted(
                        set(
                            str(value)
                            for value in values
                        )
                    )
                ),
            ),
        )
    )

    return grouped[
        grouped[
            "injury_records"
        ]
        > 1
    ].copy()


def build_recurrence_table(
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate days between consecutive injury records for each player.

    This is descriptive only. It does not assume that consecutive
    records represent distinct clinical injury episodes.
    """
    frame = (
        injuries[
            injuries[
                "injury_date"
            ].notna()
        ]
        .sort_values(
            [
                "player_name",
                "injury_date",
            ]
        )
        .copy()
    )

    frame[
        "previous_injury_date"
    ] = (
        frame.groupby(
            "player_name"
        )[
            "injury_date"
        ]
        .shift(1)
    )

    frame[
        "days_since_previous_injury_record"
    ] = (
        frame[
            "injury_date"
        ]
        - frame[
            "previous_injury_date"
        ]
    ).dt.days

    return frame


def build_objective_date_sets(
    objective: pd.DataFrame,
) -> dict[str, set[pd.Timestamp]]:
    """Build player -> set of objective recording dates."""
    result: dict[
        str,
        set[pd.Timestamp]
    ] = {}

    for player_name, group in (
        objective.groupby(
            "player_name"
        )
    ):
        result[
            player_name
        ] = set(
            group[
                "objective_date"
            ]
            .dropna()
            .tolist()
        )

    return result


def build_objective_ranges(
    objective: pd.DataFrame,
) -> dict[
    str,
    tuple[
        pd.Timestamp,
        pd.Timestamp,
    ],
]:
    """Build first/last objective date for each player."""
    result = {}

    for player_name, group in (
        objective.groupby(
            "player_name"
        )
    ):
        dates = (
            group[
                "objective_date"
            ]
            .dropna()
        )

        if dates.empty:
            continue

        result[player_name] = (
            dates.min(),
            dates.max(),
        )

    return result


def count_prior_objective_days(
    objective_dates: set[pd.Timestamp],
    injury_date: pd.Timestamp,
    window_days: int,
) -> int:
    """
    Count objective recording dates strictly before an injury,
    within a specified lookback window.
    """
    start = (
        injury_date
        - pd.Timedelta(
            days=window_days
        )
    )

    return sum(
        start
        <= date
        < injury_date
        for date in objective_dates
    )


def build_injury_objective_alignment(
    injuries: pd.DataFrame,
    objective: pd.DataFrame,
    overlap: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure objective-data availability around every injury record.
    """
    objective_dates = (
        build_objective_date_sets(
            objective
        )
    )

    objective_ranges = (
        build_objective_ranges(
            objective
        )
    )

    core_players = set(
        overlap.loc[
            overlap[
                "in_all_core_sources"
            ]
            == True,
            "player_name",
        ]
    )

    rows: list[
        dict[str, object]
    ] = []

    for _, injury in (
        injuries.iterrows()
    ):
        player_name = (
            injury[
                "player_name"
            ]
        )

        injury_date = (
            injury[
                "injury_date"
            ]
        )

        player_dates = (
            objective_dates.get(
                player_name,
                set(),
            )
        )

        player_range = (
            objective_ranges.get(
                player_name
            )
        )

        if (
            pd.notna(injury_date)
            and player_range
            is not None
        ):
            within_objective_range = (
                player_range[0]
                <= injury_date
                <= player_range[1]
            )
        else:
            within_objective_range = False

        if pd.notna(
            injury_date
        ):
            exact_objective_date = (
                injury_date
                in player_dates
            )

            prior_1 = (
                count_prior_objective_days(
                    player_dates,
                    injury_date,
                    1,
                )
            )

            prior_3 = (
                count_prior_objective_days(
                    player_dates,
                    injury_date,
                    3,
                )
            )

            prior_7 = (
                count_prior_objective_days(
                    player_dates,
                    injury_date,
                    7,
                )
            )

            prior_14 = (
                count_prior_objective_days(
                    player_dates,
                    injury_date,
                    14,
                )
            )

            prior_28 = (
                count_prior_objective_days(
                    player_dates,
                    injury_date,
                    28,
                )
            )
        else:
            exact_objective_date = False
            prior_1 = 0
            prior_3 = 0
            prior_7 = 0
            prior_14 = 0
            prior_28 = 0

        rows.append(
            {
                "player_name": player_name,
                "team": injury["team"],
                "injury_date": injury_date,
                "injury_type": (
                    injury["type"]
                ),
                "in_core_population": (
                    player_name
                    in core_players
                ),
                "objective_first_date": (
                    player_range[0]
                    if player_range
                    else pd.NaT
                ),
                "objective_last_date": (
                    player_range[1]
                    if player_range
                    else pd.NaT
                ),
                "injury_within_objective_range": (
                    within_objective_range
                ),
                "objective_on_injury_date": (
                    exact_objective_date
                ),
                "objective_days_prior_1d": (
                    prior_1
                ),
                "objective_days_prior_3d": (
                    prior_3
                ),
                "objective_days_prior_7d": (
                    prior_7
                ),
                "objective_days_prior_14d": (
                    prior_14
                ),
                "objective_days_prior_28d": (
                    prior_28
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def write_summary(
    injuries: pd.DataFrame,
    deduplicated: pd.DataFrame,
    player_summary: pd.DataFrame,
    type_summary: pd.DataFrame,
    same_day: pd.DataFrame,
    recurrence: pd.DataFrame,
    alignment: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write human-readable injury audit findings."""
    exact_duplicate_count = int(
        injuries.duplicated().sum()
    )

    valid_dates = (
        injuries[
            "injury_date"
        ].dropna()
    )

    recurrence_days = (
        recurrence[
            "days_since_previous_injury_record"
        ]
        .dropna()
    )

    lines: list[str] = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Injury Event Audit"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Raw injury records: "
        f"{len(injuries):,}"
    )

    lines.append(
        f"Exact duplicate rows "
        f"(excluding first occurrence): "
        f"{exact_duplicate_count:,}"
    )

    lines.append(
        f"Unique rows after exact deduplication: "
        f"{len(deduplicated):,}"
    )

    lines.append(
        f"Players with injury records: "
        f"{injuries['player_name'].nunique():,}"
    )

    lines.append(
        f"Unique injury dates: "
        f"{valid_dates.nunique():,}"
    )

    if not valid_dates.empty:
        lines.append(
            f"Injury date range: "
            f"{valid_dates.min()} "
            f"to "
            f"{valid_dates.max()}"
        )

    lines.append("")
    lines.append(
        "Team distribution"
    )
    lines.append(
        "-" * 80
    )

    for team, count in (
        injuries[
            "team"
        ]
        .value_counts(
            dropna=False
        )
        .items()
    ):
        lines.append(
            f"{team}: {count:,}"
        )

    lines.append("")
    lines.append(
        "Injury types"
    )
    lines.append(
        "-" * 80
    )

    for _, row in (
        type_summary.iterrows()
    ):
        lines.append(
            f"{row['type']} | "
            f"records: "
            f"{int(row['injury_records'])} | "
            f"players: "
            f"{int(row['unique_players'])}"
        )

    lines.append("")
    lines.append(
        "Player injury counts"
    )
    lines.append(
        "-" * 80
    )

    for _, row in (
        player_summary.iterrows()
    ):
        lines.append(
            f"{row['player_name']} | "
            f"records: "
            f"{int(row['injury_records'])} | "
            f"unique dates: "
            f"{int(row['unique_injury_dates'])} | "
            f"types: "
            f"{int(row['unique_injury_types'])}"
        )

    lines.append("")
    lines.append(
        "Same-player same-day records"
    )
    lines.append(
        "-" * 80
    )

    lines.append(
        f"Player-days with multiple injury records: "
        f"{len(same_day):,}"
    )

    if not recurrence_days.empty:
        lines.append("")
        lines.append(
            "Spacing between consecutive injury records"
        )
        lines.append(
            "-" * 80
        )

        lines.append(
            f"Minimum days: "
            f"{int(recurrence_days.min())}"
        )

        lines.append(
            f"Median days: "
            f"{float(recurrence_days.median()):.2f}"
        )

        lines.append(
            f"Maximum days: "
            f"{int(recurrence_days.max())}"
        )

        lines.append(
            "Consecutive records on same date: "
            f"{int((recurrence_days == 0).sum()):,}"
        )

    lines.append("")
    lines.append(
        "Objective-data coverage around injury records"
    )
    lines.append(
        "-" * 80
    )

    lines.append(
        "Injury records belonging to core players: "
        f"{int(alignment['in_core_population'].sum()):,}/"
        f"{len(alignment):,}"
    )

    lines.append(
        "Injuries inside player's objective date range: "
        f"{int(alignment['injury_within_objective_range'].sum()):,}/"
        f"{len(alignment):,}"
    )

    lines.append(
        "Objective recording on exact injury date: "
        f"{int(alignment['objective_on_injury_date'].sum()):,}/"
        f"{len(alignment):,}"
    )

    for window in [
        1,
        3,
        7,
        14,
        28,
    ]:
        column = (
            f"objective_days_prior_{window}d"
        )

        records_with_prior_data = int(
            (
                alignment[column]
                > 0
            ).sum()
        )

        lines.append(
            f"Injury records with >=1 objective day "
            f"in prior {window}d: "
            f"{records_with_prior_data:,}/"
            f"{len(alignment):,}"
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
            "Audit SoccerMon injury records, recurrence, "
            "duplicates, types, and objective-data coverage."
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subjective_root = (
        find_subjective_root()
    )

    injury_path = (
        subjective_root
        / "injury"
        / "injury.csv"
    )

    objective_manifest_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "audit"
        / "objective_file_manifest.csv"
    )

    overlap_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "audit"
        / "player_overlap_audit.csv"
    )

    print(
        "Loading injury records..."
    )

    injuries = load_injuries(
        injury_path
    )

    print(
        f"Loaded {len(injuries):,} "
        "injury records."
    )

    objective = (
        load_objective_manifest(
            objective_manifest_path
        )
    )

    overlap = (
        load_overlap_table(
            overlap_path
        )
    )

    deduplicated = (
        injuries
        .drop_duplicates()
        .copy()
    )

    player_summary = (
        build_player_injury_summary(
            injuries
        )
    )

    type_summary = (
        build_injury_type_summary(
            injuries
        )
    )

    same_day = (
        build_same_day_summary(
            injuries
        )
    )

    recurrence = (
        build_recurrence_table(
            injuries
        )
    )

    alignment = (
        build_injury_objective_alignment(
            injuries=injuries,
            objective=objective,
            overlap=overlap,
        )
    )

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    injuries.to_csv(
        output_dir
        / "injury_records_cleaned.csv",
        index=False,
    )

    deduplicated.to_csv(
        output_dir
        / "injury_records_exact_deduplicated.csv",
        index=False,
    )

    player_summary.to_csv(
        output_dir
        / "injury_player_summary.csv",
        index=False,
    )

    type_summary.to_csv(
        output_dir
        / "injury_type_summary.csv",
        index=False,
    )

    same_day.to_csv(
        output_dir
        / "injury_same_day_records.csv",
        index=False,
    )

    recurrence.to_csv(
        output_dir
        / "injury_recurrence_audit.csv",
        index=False,
    )

    alignment.to_csv(
        output_dir
        / "injury_objective_alignment.csv",
        index=False,
    )

    summary_path = (
        output_dir
        / "injury_event_summary.txt"
    )

    write_summary(
        injuries=injuries,
        deduplicated=deduplicated,
        player_summary=player_summary,
        type_summary=type_summary,
        same_day=same_day,
        recurrence=recurrence,
        alignment=alignment,
        output_path=summary_path,
    )


if __name__ == "__main__":
    main()