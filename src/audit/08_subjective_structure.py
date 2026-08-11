from __future__ import annotations

# Lines 3-9: Standard-library imports
import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

# Lines 11-12: Third-party imports
import pandas as pd


# Line 16: Resolve repository root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Lines 20-32: Known subjective table groups
WELLNESS_FILES = [
    "fatigue.csv",
    "mood.csv",
    "readiness.csv",
    "sleep_duration.csv",
    "sleep_quality.csv",
    "soreness.csv",
    "stress.csv",
]

TRAINING_LOAD_FILES = [
    "acwr.csv",
    "atl.csv",
    "ctl28.csv",
    "ctl42.csv",
    "daily_load.csv",
    "monotony.csv",
    "strain.csv",
    "weekly_load.csv",
]


@dataclass
class WideTableRecord:
    category: str
    filename: str
    rows: int
    player_columns: int
    first_column_name: str
    first_date: str
    last_date: str
    unique_dates: int
    duplicate_dates: int
    missing_player_cells: int
    missing_player_fraction: float
    minimum_value: float | None
    maximum_value: float | None
    mean_value: float | None
    median_value: float | None
    player_set_matches_reference: bool
    date_index_matches_reference: bool
    error: str


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


def detect_date_column(frame: pd.DataFrame) -> str:
    """
    Treat the first column as the date column for wide tables.

    This handles labels such as:
    Date
    Fatigue Data
    Mood Data
    Readiness Data
    """
    if frame.empty and len(frame.columns) == 0:
        raise ValueError("Table has no columns")

    return str(frame.columns[0])


def get_player_columns(frame: pd.DataFrame) -> list[str]:
    """Return all team-prefixed player columns."""
    return [
        str(column)
        for column in frame.columns
        if str(column).startswith("TeamA-")
        or str(column).startswith("TeamB-")
    ]


def parse_dates(
    frame: pd.DataFrame,
    date_column: str,
) -> pd.Series:
    """Parse date values from a wide subjective table."""
    return pd.to_datetime(
        frame[date_column],
        errors="coerce",
    )


def summarize_numeric_values(
    frame: pd.DataFrame,
    player_columns: list[str],
) -> dict[str, float | None]:
    """Summarize numeric values across player-value cells only."""
    if not player_columns:
        return {
            "minimum_value": None,
            "maximum_value": None,
            "mean_value": None,
            "median_value": None,
        }

    numeric = (
        frame[player_columns]
        .apply(pd.to_numeric, errors="coerce")
        .stack()
    )

    if numeric.empty:
        return {
            "minimum_value": None,
            "maximum_value": None,
            "mean_value": None,
            "median_value": None,
        }

    return {
        "minimum_value": float(numeric.min()),
        "maximum_value": float(numeric.max()),
        "mean_value": float(numeric.mean()),
        "median_value": float(numeric.median()),
    }


def inspect_wide_table(
    path: Path,
    category: str,
    reference_players: set[str] | None,
    reference_dates: list[pd.Timestamp] | None,
) -> tuple[
    WideTableRecord,
    set[str],
    list[pd.Timestamp],
    list[dict[str, object]],
]:
    """
    Inspect one wellness or training-load table.

    Returns:
    - file summary
    - player set
    - ordered date index
    - per-player missingness rows
    """
    try:
        frame = pd.read_csv(path)

        date_column = detect_date_column(
            frame
        )

        player_columns = get_player_columns(
            frame
        )

        player_set = set(
            player_columns
        )

        dates = parse_dates(
            frame,
            date_column,
        )

        valid_dates = dates.dropna()

        ordered_dates = valid_dates.tolist()

        duplicate_dates = int(
            valid_dates.duplicated().sum()
        )

        player_values = (
            frame[player_columns]
            if player_columns
            else pd.DataFrame()
        )

        total_player_cells = (
            len(frame)
            * len(player_columns)
        )

        missing_player_cells = (
            int(player_values.isna().sum().sum())
            if player_columns
            else 0
        )

        missing_player_fraction = (
            missing_player_cells
            / total_player_cells
            if total_player_cells > 0
            else 0.0
        )

        numeric_summary = (
            summarize_numeric_values(
                frame,
                player_columns,
            )
        )

        if reference_players is None:
            player_set_matches_reference = True
        else:
            player_set_matches_reference = (
                player_set == reference_players
            )

        if reference_dates is None:
            date_index_matches_reference = True
        else:
            date_index_matches_reference = (
                ordered_dates
                == reference_dates
            )

        per_player_rows: list[
            dict[str, object]
        ] = []

        for player in player_columns:
            series = frame[player]

            per_player_rows.append(
                {
                    "category": category,
                    "filename": path.name,
                    "player_name": player,
                    "rows": len(series),
                    "missing_count": int(
                        series.isna().sum()
                    ),
                    "missing_fraction": float(
                        series.isna().mean()
                    ),
                    "non_missing_count": int(
                        series.notna().sum()
                    ),
                }
            )

        record = WideTableRecord(
            category=category,
            filename=path.name,
            rows=len(frame),
            player_columns=len(player_columns),
            first_column_name=date_column,
            first_date=(
                str(valid_dates.min())
                if not valid_dates.empty
                else ""
            ),
            last_date=(
                str(valid_dates.max())
                if not valid_dates.empty
                else ""
            ),
            unique_dates=int(
                valid_dates.nunique()
            ),
            duplicate_dates=duplicate_dates,
            missing_player_cells=(
                missing_player_cells
            ),
            missing_player_fraction=(
                missing_player_fraction
            ),
            minimum_value=(
                numeric_summary[
                    "minimum_value"
                ]
            ),
            maximum_value=(
                numeric_summary[
                    "maximum_value"
                ]
            ),
            mean_value=(
                numeric_summary[
                    "mean_value"
                ]
            ),
            median_value=(
                numeric_summary[
                    "median_value"
                ]
            ),
            player_set_matches_reference=(
                player_set_matches_reference
            ),
            date_index_matches_reference=(
                date_index_matches_reference
            ),
            error="",
        )

        return (
            record,
            player_set,
            ordered_dates,
            per_player_rows,
        )

    except Exception as exc:
        record = WideTableRecord(
            category=category,
            filename=path.name,
            rows=0,
            player_columns=0,
            first_column_name="",
            first_date="",
            last_date="",
            unique_dates=0,
            duplicate_dates=0,
            missing_player_cells=0,
            missing_player_fraction=0.0,
            minimum_value=None,
            maximum_value=None,
            mean_value=None,
            median_value=None,
            player_set_matches_reference=False,
            date_index_matches_reference=False,
            error=str(exc),
        )

        return (
            record,
            set(),
            [],
            [],
        )


def inspect_injury_duplicates(
    injury_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Identify exact duplicate rows in injury.csv."""
    frame = pd.read_csv(
        injury_path
    )

    duplicate_mask = frame.duplicated(
        keep=False
    )

    duplicate_rows = (
        frame[duplicate_mask]
        .sort_values(
            list(frame.columns)
        )
        .copy()
    )

    unique_frame = (
        frame.drop_duplicates()
        .copy()
    )

    return (
        duplicate_rows,
        unique_frame,
    )


def write_csv_records(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write dictionaries to CSV."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        output_path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fieldnames = list(
        rows[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    records: list[WideTableRecord],
    player_sets: dict[str, set[str]],
    injury_duplicates: pd.DataFrame,
    injury_unique: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write human-readable subjective-structure findings."""
    successful = [
        record
        for record in records
        if not record.error
    ]

    failed = [
        record
        for record in records
        if record.error
    ]

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append(
        "SoccerMon Subjective Structure Audit"
    )
    lines.append("=" * 80)

    lines.append(
        f"Wide tables inspected: "
        f"{len(records):,}"
    )

    lines.append(
        f"Successful reads: "
        f"{len(successful):,}"
    )

    lines.append(
        f"Failed reads: "
        f"{len(failed):,}"
    )

    lines.append("")
    lines.append(
        "Wide-table structure"
    )
    lines.append("-" * 80)

    for record in successful:
        lines.append(
            f"{record.category}/{record.filename} | "
            f"players: {record.player_columns} | "
            f"dates: {record.unique_dates} | "
            f"first column: {record.first_column_name} | "
            f"range: {record.first_date} "
            f"to {record.last_date} | "
            f"missing player fraction: "
            f"{record.missing_player_fraction:.4f} | "
            f"value range: "
            f"{record.minimum_value} "
            f"to {record.maximum_value}"
        )

    lines.append("")
    lines.append(
        "Player-set consistency"
    )
    lines.append("-" * 80)

    for filename, players in sorted(
        player_sets.items()
    ):
        lines.append(
            f"{filename}: "
            f"{len(players):,} players"
        )

    lines.append("")
    lines.append(
        "Injury duplicate audit"
    )
    lines.append("-" * 80)

    lines.append(
        f"Original injury rows: "
        f"{len(injury_unique) + len(injury_duplicates) // 2:,}"
    )

    lines.append(
        f"Rows participating in duplicate groups: "
        f"{len(injury_duplicates):,}"
    )

    lines.append(
        f"Unique injury rows after drop_duplicates(): "
        f"{len(injury_unique):,}"
    )

    if failed:
        lines.append("")
        lines.append(
            "Failed tables"
        )
        lines.append("-" * 80)

        for record in failed:
            lines.append(
                f"{record.filename}: "
                f"{record.error}"
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
            "Validate SoccerMon subjective wide tables, "
            "player sets, date alignment, missingness, "
            "value ranges, and injury duplicates."
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

    records: list[
        WideTableRecord
    ] = []

    per_player_rows: list[
        dict[str, object]
    ] = []

    player_sets: dict[
        str,
        set[str]
    ] = {}

    reference_players: (
        set[str] | None
    ) = None

    reference_dates: (
        list[pd.Timestamp] | None
    ) = None

    table_specs: list[
        tuple[str, Path]
    ] = []

    for filename in TRAINING_LOAD_FILES:
        table_specs.append(
            (
                "training-load",
                subjective_root
                / "training-load"
                / filename,
            )
        )

    for filename in WELLNESS_FILES:
        table_specs.append(
            (
                "wellness",
                subjective_root
                / "wellness"
                / filename,
            )
        )

    print(
        f"Inspecting "
        f"{len(table_specs):,} wide subjective tables..."
    )

    for index, (
        category,
        path,
    ) in enumerate(
        table_specs,
        start=1,
    ):
        (
            record,
            player_set,
            ordered_dates,
            player_rows,
        ) = inspect_wide_table(
            path=path,
            category=category,
            reference_players=(
                reference_players
            ),
            reference_dates=(
                reference_dates
            ),
        )

        if (
            reference_players is None
            and not record.error
        ):
            reference_players = (
                player_set.copy()
            )

        if (
            reference_dates is None
            and not record.error
        ):
            reference_dates = list(
                ordered_dates
            )

        records.append(
            record
        )

        player_sets[
            path.name
        ] = player_set

        per_player_rows.extend(
            player_rows
        )

        print(
            f"Inspected {index:,}/"
            f"{len(table_specs):,}: "
            f"{path.name}"
        )

    injury_path = (
        subjective_root
        / "injury"
        / "injury.csv"
    )

    (
        injury_duplicates,
        injury_unique,
    ) = inspect_injury_duplicates(
        injury_path
    )

    structure_path = (
        args.output_dir
        / "subjective_structure_tables.csv"
    )

    missingness_path = (
        args.output_dir
        / "subjective_player_missingness.csv"
    )

    duplicate_path = (
        args.output_dir
        / "subjective_injury_duplicates.csv"
    )

    deduplicated_injury_path = (
        args.output_dir
        / "subjective_injury_deduplicated_preview.csv"
    )

    summary_path = (
        args.output_dir
        / "subjective_structure_summary.txt"
    )

    structure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            asdict(record)
            for record in records
        ]
    ).to_csv(
        structure_path,
        index=False,
    )

    write_csv_records(
        per_player_rows,
        missingness_path,
    )

    injury_duplicates.to_csv(
        duplicate_path,
        index=False,
    )

    injury_unique.to_csv(
        deduplicated_injury_path,
        index=False,
    )

    write_summary(
        records=records,
        player_sets=player_sets,
        injury_duplicates=(
            injury_duplicates
        ),
        injury_unique=(
            injury_unique
        ),
        output_path=(
            summary_path
        ),
    )

    print(
        f"Structure audit written to: "
        f"{structure_path}"
    )

    print(
        f"Player missingness written to: "
        f"{missingness_path}"
    )

    print(
        f"Injury duplicates written to: "
        f"{duplicate_path}"
    )


if __name__ == "__main__":
    main()