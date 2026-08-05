from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
# this lets script be called from anywhere 
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Expected objective filename:
# 2020-06-01-TeamA-1846d424-c17c-6279-23c6-612f48268673.parquet
FILE_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})-"
    r"(?P<team>Team[A-Za-z0-9]+)-"
    r"(?P<player_id>.+)\.parquet$"
)


@dataclass
class ObjectiveFileRecord:
    path: str
    filename: str
    team: str
    year: int
    month: str
    date: str
    player_id: str
    size_bytes: int


def human_readable_size(size_bytes: int) -> str:
    """Convert a byte count into a readable size."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"


def parse_objective_file(path: Path) -> ObjectiveFileRecord | None:
    """
    Parse metadata from an objective Parquet filename.

    Returns None when the filename does not match the expected format.
    """
    match = FILE_PATTERN.match(path.name)

    if match is None:
        return None

    date = match.group("date")
    year = int(date[:4])
    month = date[:7]

    return ObjectiveFileRecord(
        path=str(path),
        filename=path.name,
        team=match.group("team"),
        year=year,
        month=month,
        date=date,
        player_id=match.group("player_id"),
        size_bytes=path.stat().st_size,
    )


def collect_objective_records(raw_dir: Path) -> tuple[list[ObjectiveFileRecord], list[Path]]:
    """Find and parse all objective Parquet files."""
    records: list[ObjectiveFileRecord] = []
    unmatched_files: list[Path] = []

    for path in raw_dir.rglob("*.parquet"):
        record = parse_objective_file(path)

        if record is None:
            unmatched_files.append(path)
        else:
            records.append(record)

    records.sort(
        key=lambda record: (
            record.team,
            record.year,
            record.date,
            record.player_id,
        )
    )

    return records, unmatched_files


def write_manifest(
    records: list[ObjectiveFileRecord],
    output_path: Path,
) -> None:
    """Write one row per objective Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "filename",
        "team",
        "year",
        "month",
        "date",
        "player_id",
        "size_bytes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def write_summary(
    records: list[ObjectiveFileRecord],
    unmatched_files: list[Path],
    raw_dir: Path,
    output_path: Path,
) -> None:
    """Write a text summary of the objective dataset structure."""
    teams = sorted({record.team for record in records})
    years = sorted({record.year for record in records})
    months = sorted({record.month for record in records})
    dates = sorted({record.date for record in records})
    players = sorted({record.player_id for record in records})

    total_size = sum(record.size_bytes for record in records)

    team_counts = Counter(record.team for record in records)
    year_counts = Counter(record.year for record in records)
    month_counts = Counter(record.month for record in records)
    team_year_counts = Counter(
        (record.team, record.year) for record in records
    )
    player_counts = Counter(record.player_id for record in records)

    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("SoccerMon Objective Dataset Structure Summary")
    lines.append("=" * 70)
    lines.append(f"Raw data directory: {raw_dir.resolve()}")
    lines.append(f"Objective Parquet files: {len(records):,}")
    lines.append(f"Total Parquet size: {human_readable_size(total_size)}")
    lines.append(f"Teams: {len(teams)} ({', '.join(teams)})")
    lines.append(
        f"Years: {len(years)} ({', '.join(str(year) for year in years)})"
    )
    lines.append(f"Unique months: {len(months):,}")
    lines.append(f"Unique dates: {len(dates):,}")
    lines.append(f"Unique player IDs: {len(players):,}")
    lines.append(f"Unmatched Parquet filenames: {len(unmatched_files):,}")

    lines.append("")
    lines.append("Files by team")
    lines.append("-" * 70)

    for team in teams:
        lines.append(f"{team}: {team_counts[team]:,}")

    lines.append("")
    lines.append("Files by year")
    lines.append("-" * 70)

    for year in years:
        lines.append(f"{year}: {year_counts[year]:,}")

    lines.append("")
    lines.append("Files by team and year")
    lines.append("-" * 70)

    for team, year in sorted(team_year_counts):
        count = team_year_counts[(team, year)]
        lines.append(f"{team} {year}: {count:,}")

    lines.append("")
    lines.append("Files by month")
    lines.append("-" * 70)

    for month in months:
        lines.append(f"{month}: {month_counts[month]:,}")

    lines.append("")
    lines.append("Player session-file counts")
    lines.append("-" * 70)

    for player_id, count in player_counts.most_common():
        lines.append(f"{player_id}: {count:,}")

    if unmatched_files:
        lines.append("")
        lines.append("Unmatched Parquet files")
        lines.append("-" * 70)

        for path in unmatched_files:
            lines.append(str(path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:15]))
    print()
    print(f"Full summary written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the SoccerMon raw objective-data directory and create "
            "a file manifest and structure summary."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Path to the raw SoccerMon data directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "audit",
        help="Directory where audit outputs will be written.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {args.raw_dir}"
        )

    print(f"Scanning: {args.raw_dir.resolve()}")

    records, unmatched_files = collect_objective_records(args.raw_dir)

    if not records:
        raise RuntimeError(
            f"No matching objective Parquet files were found under "
            f"{args.raw_dir}"
        )

    manifest_path = args.output_dir / "objective_file_manifest.csv"
    summary_path = args.output_dir / "objective_dataset_summary.txt"

    write_manifest(records, manifest_path)

    write_summary(
        records=records,
        unmatched_files=unmatched_files,
        raw_dir=args.raw_dir,
        output_path=summary_path,
    )

    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()