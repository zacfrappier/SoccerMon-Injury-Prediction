from __future__ import annotations

# Lines 3-10: Standard-library imports
import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

# Lines 12-13: Third-party imports
import pandas as pd


# Line 17: Resolve the repository root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Lines 21-37: One summary record per subjective CSV file
@dataclass
class SubjectiveFileRecord:
    path: str
    category: str
    filename: str
    table_name: str
    rows: int
    columns: int
    column_names: str
    player_column: str
    unique_players: int | None
    date_column: str
    first_date: str
    last_date: str
    total_missing_cells: int
    missing_fraction: float
    duplicate_rows: int
    error: str


def find_subjective_root(raw_dir: Path) -> Path:
    """
    Locate the actual subjective-data root.

    Your extraction currently contains:
    data/raw/subjective/subjective/
    """
    direct = raw_dir / "subjective"

    nested = (
        raw_dir
        / "subjective"
        / "subjective"
    )

    if nested.exists():
        return nested

    if direct.exists():
        return direct

    raise FileNotFoundError(
        "Could not locate the subjective dataset under "
        f"{raw_dir}"
    )


def find_player_column(columns: list[str]) -> str:
    """
    Find a likely player identifier column.

    Returns an empty string when no likely player column exists.
    """
    candidates = [
        "player_name",
        "player",
        "player_id",
        "athlete",
        "athlete_id",
    ]

    lower_map = {
        column.lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    return ""


def find_date_column(columns: list[str]) -> str:
    """
    Find a likely timestamp/date column.
    """
    candidates = [
        "timestamp",
        "date",
        "datetime",
        "time",
        "created_at",
    ]

    lower_map = {
        column.lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    return ""


def summarize_dates(
    frame: pd.DataFrame,
    date_column: str,
) -> tuple[str, str]:
    """
    Parse the likely date column and return first/last valid dates.
    """
    if not date_column:
        return "", ""

    parsed = pd.to_datetime(
        frame[date_column],
        errors="coerce",
    )

    valid = parsed.dropna()

    if valid.empty:
        return "", ""

    return (
        str(valid.min()),
        str(valid.max()),
    )


def inspect_csv(
    path: Path,
    subjective_root: Path,
) -> tuple[
    SubjectiveFileRecord,
    list[dict[str, object]],
]:
    """
    Inspect one subjective CSV and return:
    1. a file-level summary
    2. a per-column missing-value summary
    """
    try:
        frame = pd.read_csv(path)

        category = (
            path.parent.relative_to(
                subjective_root
            ).as_posix()
        )

        columns = [
            str(column)
            for column in frame.columns
        ]

        player_column = find_player_column(
            columns
        )

        date_column = find_date_column(
            columns
        )

        if player_column:
            unique_players = int(
                frame[player_column]
                .dropna()
                .astype(str)
                .nunique()
            )
        else:
            unique_players = None

        first_date, last_date = summarize_dates(
            frame,
            date_column,
        )

        total_cells = (
            len(frame)
            * len(frame.columns)
        )

        total_missing = int(
            frame.isna()
            .sum()
            .sum()
        )

        if total_cells > 0:
            missing_fraction = (
                total_missing
                / total_cells
            )
        else:
            missing_fraction = 0.0

        duplicate_rows = int(
            frame.duplicated().sum()
        )

        record = SubjectiveFileRecord(
            path=str(path),
            category=category,
            filename=path.name,
            table_name=path.stem,
            rows=len(frame),
            columns=len(frame.columns),
            column_names=";".join(columns),
            player_column=player_column,
            unique_players=unique_players,
            date_column=date_column,
            first_date=first_date,
            last_date=last_date,
            total_missing_cells=total_missing,
            missing_fraction=missing_fraction,
            duplicate_rows=duplicate_rows,
            error="",
        )

        column_rows: list[
            dict[str, object]
        ] = []

        for column in frame.columns:
            series = frame[column]

            column_rows.append(
                {
                    "category": category,
                    "filename": path.name,
                    "table_name": path.stem,
                    "column": column,
                    "dtype": str(series.dtype),
                    "rows": len(series),
                    "missing_count": int(
                        series.isna().sum()
                    ),
                    "missing_fraction": (
                        float(
                            series.isna().mean()
                        )
                        if len(series) > 0
                        else 0.0
                    ),
                    "unique_non_null_values": int(
                        series.dropna().nunique()
                    ),
                }
            )

        return record, column_rows

    except Exception as exc:
        record = SubjectiveFileRecord(
            path=str(path),
            category="",
            filename=path.name,
            table_name=path.stem,
            rows=0,
            columns=0,
            column_names="",
            player_column="",
            unique_players=None,
            date_column="",
            first_date="",
            last_date="",
            total_missing_cells=0,
            missing_fraction=0.0,
            duplicate_rows=0,
            error=str(exc),
        )

        return record, []


def write_file_inventory(
    records: list[SubjectiveFileRecord],
    output_path: Path,
) -> None:
    """Write one row per subjective CSV."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        SubjectiveFileRecord
        .__dataclass_fields__
        .keys()
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

        for record in records:
            writer.writerow(
                asdict(record)
            )


def write_column_inventory(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write one row per column per subjective table."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "category",
        "filename",
        "table_name",
        "column",
        "dtype",
        "rows",
        "missing_count",
        "missing_fraction",
        "unique_non_null_values",
    ]

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
    records: list[SubjectiveFileRecord],
    output_path: Path,
) -> None:
    """Write a human-readable subjective-data summary."""
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

    categories = sorted(
        {
            record.category
            for record in successful
        }
    )

    total_rows = sum(
        record.rows
        for record in successful
    )

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append(
        "SoccerMon Subjective Dataset Inventory"
    )
    lines.append("=" * 80)

    lines.append(
        f"CSV files discovered: {len(records):,}"
    )

    lines.append(
        f"Successful reads: {len(successful):,}"
    )

    lines.append(
        f"Failed reads: {len(failed):,}"
    )

    lines.append(
        f"Logical categories: {len(categories):,}"
    )

    lines.append(
        f"Total rows across CSV files: "
        f"{total_rows:,}"
    )

    lines.append("")
    lines.append("Categories")
    lines.append("-" * 80)

    for category in categories:
        category_records = [
            record
            for record in successful
            if record.category == category
        ]

        lines.append(
            f"{category}: "
            f"{len(category_records)} files"
        )

    lines.append("")
    lines.append("File-level details")
    lines.append("-" * 80)

    for record in successful:
        lines.append(
            f"{record.category}/{record.filename} | "
            f"rows: {record.rows:,} | "
            f"columns: {record.columns:,} | "
            f"players: {record.unique_players} | "
            f"date column: {record.date_column or 'NONE'} | "
            f"range: "
            f"{record.first_date or 'UNKNOWN'} "
            f"to "
            f"{record.last_date or 'UNKNOWN'} | "
            f"missing fraction: "
            f"{record.missing_fraction:.4f} | "
            f"duplicate rows: "
            f"{record.duplicate_rows:,}"
        )

    if failed:
        lines.append("")
        lines.append(
            "Files that could not be read"
        )
        lines.append("-" * 80)

        for record in failed:
            lines.append(
                f"{record.path}: {record.error}"
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
        f"Summary written to: {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory all SoccerMon subjective CSV files, "
            "their schemas, players, dates, and missing values."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "raw"
        ),
        help=(
            "Path to the raw SoccerMon data directory."
        ),
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
        help=(
            "Directory for subjective audit outputs."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subjective_root = find_subjective_root(
        args.raw_dir
    )

    print(
        f"Scanning subjective data: "
        f"{subjective_root.resolve()}"
    )

    csv_files = sorted(
        subjective_root.rglob("*.csv")
    )

    if not csv_files:
        raise RuntimeError(
            "No subjective CSV files were found."
        )

    print(
        f"Found {len(csv_files):,} CSV files."
    )

    file_records: list[
        SubjectiveFileRecord
    ] = []

    column_records: list[
        dict[str, object]
    ] = []

    for index, path in enumerate(
        csv_files,
        start=1,
    ):
        record, column_rows = inspect_csv(
            path,
            subjective_root,
        )

        file_records.append(record)
        column_records.extend(
            column_rows
        )

        print(
            f"Inspected {index:,}/"
            f"{len(csv_files):,}: "
            f"{path.name}"
        )

    file_inventory_path = (
        args.output_dir
        / "subjective_file_inventory.csv"
    )

    column_inventory_path = (
        args.output_dir
        / "subjective_column_inventory.csv"
    )

    summary_path = (
        args.output_dir
        / "subjective_dataset_summary.txt"
    )

    write_file_inventory(
        file_records,
        file_inventory_path,
    )

    write_column_inventory(
        column_records,
        column_inventory_path,
    )

    write_summary(
        file_records,
        summary_path,
    )

    print(
        f"File inventory written to: "
        f"{file_inventory_path}"
    )

    print(
        f"Column inventory written to: "
        f"{column_inventory_path}"
    )


if __name__ == "__main__":
    main()