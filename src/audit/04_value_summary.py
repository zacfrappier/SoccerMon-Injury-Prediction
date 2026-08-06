from __future__ import annotations

# Lines 3-10: Standard-library imports
import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import median

# Lines 12-14: Third-party imports
import pandas as pd
import pyarrow.parquet as pq


# Line 18: Resolve the repository root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Lines 22-36: Columns expected in every objective Parquet file
EXPECTED_COLUMNS = [
    "player_name",
    "time",
    "lat",
    "lon",
    "speed",
    "heart_rate",
    "hacc",
    "hdop",
    "signal_quality",
    "num_satellites",
    "inst_acc_impulse",
    "accl_x",
    "accl_y",
    "accl_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


# Lines 40-58: One summary record per inspected file
@dataclass
class FileValueRecord:
    path: str
    filename: str
    team: str
    year: str
    player_id: str
    rows_in_file: int
    rows_sampled: int
    time_parse_rate: float
    median_interval_seconds: float | None
    player_name_matches_filename: bool
    missing_cells: int
    suspicious_latitude_rows: int
    suspicious_longitude_rows: int
    negative_speed_rows: int
    suspicious_speed_rows: int
    suspicious_heart_rate_rows: int
    error: str


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Load the objective file manifest created by Script 1."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest does not exist: {manifest_path}\n"
            "Run src/audit/01_dataset_tree.py first."
        )

    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_sample_rows(
    rows: list[dict[str, str]],
    sample_per_team_year: int,
) -> list[dict[str, str]]:
    """
    Select representative files from each team-year combination.

    Sampling is deterministic so repeated runs use the same files.
    """
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[(row["team"], row["year"])].append(row)

    selected: list[dict[str, str]] = []

    for key in sorted(grouped):
        group_rows = sorted(
            grouped[key],
            key=lambda row: (
                row["date"],
                row["player_id"],
                row["filename"],
            ),
        )

        if len(group_rows) <= sample_per_team_year:
            selected.extend(group_rows)
            continue

        if sample_per_team_year == 1:
            indices = [0]
        else:
            last_index = len(group_rows) - 1
            indices = [
                round(index * last_index / (sample_per_team_year - 1))
                for index in range(sample_per_team_year)
            ]

        selected.extend(group_rows[index] for index in indices)

    return selected


def evenly_spaced_indices(total_rows: int, sample_rows: int) -> list[int]:
    """Return evenly spaced row positions across a file."""
    if total_rows <= 0:
        return []

    if total_rows <= sample_rows:
        return list(range(total_rows))

    if sample_rows == 1:
        return [0]

    last_index = total_rows - 1

    return sorted(
        {
            round(index * last_index / (sample_rows - 1))
            for index in range(sample_rows)
        }
    )


def read_sample(path: Path, sample_rows: int) -> tuple[pd.DataFrame, int]:
    """
    Read one Parquet file and keep evenly spaced sample rows.

    Each SoccerMon file contains one row group, so the file is read once
    and immediately reduced to the requested sample.
    """
    parquet_file = pq.ParquetFile(path)
    total_rows = parquet_file.metadata.num_rows

    table = parquet_file.read(columns=EXPECTED_COLUMNS)
    frame = table.to_pandas()

    indices = evenly_spaced_indices(total_rows, sample_rows)
    sampled = frame.iloc[indices].copy()

    return sampled, total_rows


def calculate_sampling_interval_seconds(
    parsed_times: pd.Series,
) -> float | None:
    """Estimate the median interval between sampled timestamps."""
    valid_times = parsed_times.dropna().sort_values()

    if len(valid_times) < 2:
        return None

    differences = valid_times.diff().dt.total_seconds().dropna()
    positive_differences = differences[differences > 0]

    if positive_differences.empty:
        return None

    return float(positive_differences.median())


def summarize_numeric_column(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, object]:
    """Calculate descriptive statistics for one numeric column."""
    numeric = pd.to_numeric(frame[column], errors="coerce")

    return {
        "column": column,
        "count": int(numeric.notna().sum()),
        "missing": int(numeric.isna().sum()),
        "minimum": float(numeric.min()) if numeric.notna().any() else None,
        "maximum": float(numeric.max()) if numeric.notna().any() else None,
        "mean": float(numeric.mean()) if numeric.notna().any() else None,
        "median": float(numeric.median()) if numeric.notna().any() else None,
        "standard_deviation": (
            float(numeric.std()) if numeric.notna().sum() > 1 else None
        ),
        "zero_count": int((numeric == 0).sum()),
        "negative_count": int((numeric < 0).sum()),
    }


def inspect_file(
    row: dict[str, str],
    sample_rows: int,
) -> tuple[FileValueRecord, list[dict[str, object]], pd.DataFrame | None]:
    """Inspect sampled values from one objective Parquet file."""
    path = Path(row["path"])

    try:
        frame, total_rows = read_sample(path, sample_rows)

        missing_columns = [
            column for column in EXPECTED_COLUMNS
            if column not in frame.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing expected columns: "
                + ", ".join(missing_columns)
            )

        parsed_times = pd.to_datetime(
            frame["time"],
            errors="coerce",
            utc=True,
        )

        time_parse_rate = float(parsed_times.notna().mean())

        median_interval = calculate_sampling_interval_seconds(
            parsed_times
        )

        player_names = (
            frame["player_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        player_name_matches = (
            len(player_names) > 0
            and all(name == row["player_id"] for name in player_names)
        )

        lat = pd.to_numeric(frame["lat"], errors="coerce")
        lon = pd.to_numeric(frame["lon"], errors="coerce")
        speed = pd.to_numeric(frame["speed"], errors="coerce")
        heart_rate = pd.to_numeric(
            frame["heart_rate"],
            errors="coerce",
        )

        # Lines 224-229: Broad diagnostic thresholds.
        # These are flags for inspection, not automatic deletion rules.
        suspicious_latitude = int(
            ((lat < -90) | (lat > 90)).sum()
        )
        suspicious_longitude = int(
            ((lon < -180) | (lon > 180)).sum()
        )
        negative_speed = int((speed < 0).sum())
        suspicious_speed = int((speed > 15).sum())
        suspicious_heart_rate = int(
            ((heart_rate < 30) | (heart_rate > 240)).sum()
        )

        numeric_summaries = [
            summarize_numeric_column(frame, column)
            for column in EXPECTED_COLUMNS
            if column not in {"player_name", "time"}
        ]

        record = FileValueRecord(
            path=str(path),
            filename=path.name,
            team=row["team"],
            year=row["year"],
            player_id=row["player_id"],
            rows_in_file=total_rows,
            rows_sampled=len(frame),
            time_parse_rate=time_parse_rate,
            median_interval_seconds=median_interval,
            player_name_matches_filename=player_name_matches,
            missing_cells=int(frame.isna().sum().sum()),
            suspicious_latitude_rows=suspicious_latitude,
            suspicious_longitude_rows=suspicious_longitude,
            negative_speed_rows=negative_speed,
            suspicious_speed_rows=suspicious_speed,
            suspicious_heart_rate_rows=suspicious_heart_rate,
            error="",
        )

        return record, numeric_summaries, frame

    except Exception as exc:
        record = FileValueRecord(
            path=str(path),
            filename=path.name,
            team=row["team"],
            year=row["year"],
            player_id=row["player_id"],
            rows_in_file=0,
            rows_sampled=0,
            time_parse_rate=0.0,
            median_interval_seconds=None,
            player_name_matches_filename=False,
            missing_cells=0,
            suspicious_latitude_rows=0,
            suspicious_longitude_rows=0,
            negative_speed_rows=0,
            suspicious_speed_rows=0,
            suspicious_heart_rate_rows=0,
            error=str(exc),
        )

        return record, [], None


def write_file_audit(
    records: list[FileValueRecord],
    output_path: Path,
) -> None:
    """Write one row per inspected Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "filename",
        "team",
        "year",
        "player_id",
        "rows_in_file",
        "rows_sampled",
        "time_parse_rate",
        "median_interval_seconds",
        "player_name_matches_filename",
        "missing_cells",
        "suspicious_latitude_rows",
        "suspicious_longitude_rows",
        "negative_speed_rows",
        "suspicious_speed_rows",
        "suspicious_heart_rate_rows",
        "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def write_column_audit(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write descriptive statistics by file and column."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "team",
        "year",
        "player_id",
        "column",
        "count",
        "missing",
        "minimum",
        "maximum",
        "mean",
        "median",
        "standard_deviation",
        "zero_count",
        "negative_count",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_example_rows(
    example_frames: list[pd.DataFrame],
    output_path: Path,
) -> None:
    """Write a small collection of sampled rows for visual inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not example_frames:
        output_path.write_text("", encoding="utf-8")
        return

    combined = pd.concat(example_frames, ignore_index=True)
    combined.to_csv(output_path, index=False)


def write_summary(
    records: list[FileValueRecord],
    output_path: Path,
) -> None:
    """Write a readable summary of the value audit."""
    successful = [
        record for record in records
        if not record.error
    ]

    failed = [
        record for record in records
        if record.error
    ]

    intervals = [
        record.median_interval_seconds
        for record in successful
        if record.median_interval_seconds is not None
    ]

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("SoccerMon Objective Sampled-Value Audit")
    lines.append("=" * 80)
    lines.append(f"Files inspected: {len(records):,}")
    lines.append(f"Successful reads: {len(successful):,}")
    lines.append(f"Failed reads: {len(failed):,}")

    if successful:
        lines.append(
            "Files with fully parseable timestamps: "
            f"{sum(record.time_parse_rate == 1.0 for record in successful):,}"
        )
        lines.append(
            "Files whose player_name matched the filename ID: "
            f"{sum(record.player_name_matches_filename for record in successful):,}"
        )
        lines.append(
            "Files containing sampled missing values: "
            f"{sum(record.missing_cells > 0 for record in successful):,}"
        )
        lines.append(
            "Files containing suspicious latitude values: "
            f"{sum(record.suspicious_latitude_rows > 0 for record in successful):,}"
        )
        lines.append(
            "Files containing suspicious longitude values: "
            f"{sum(record.suspicious_longitude_rows > 0 for record in successful):,}"
        )
        lines.append(
            "Files containing negative speed values: "
            f"{sum(record.negative_speed_rows > 0 for record in successful):,}"
        )
        lines.append(
            "Files containing sampled speed above 15 m/s: "
            f"{sum(record.suspicious_speed_rows > 0 for record in successful):,}"
        )
        lines.append(
            "Files containing sampled heart rate outside 30-240: "
            f"{sum(record.suspicious_heart_rate_rows > 0 for record in successful):,}"
        )

    if intervals:
        lines.append("")
        lines.append("Estimated sampled timestamp intervals")
        lines.append("-" * 80)
        lines.append(f"Minimum: {min(intervals):.6f} seconds")
        lines.append(f"Maximum: {max(intervals):.6f} seconds")
        lines.append(f"Median: {median(intervals):.6f} seconds")

    if failed:
        lines.append("")
        lines.append("Files that could not be inspected")
        lines.append("-" * 80)

        for record in failed:
            lines.append(f"{record.path}: {record.error}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    print()
    print(f"Summary written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect sampled values from representative SoccerMon "
            "objective Parquet files."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "audit"
            / "objective_file_manifest.csv"
        ),
        help="Path to the objective-file manifest.",
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
        help="Directory for value-audit outputs.",
    )

    parser.add_argument(
        "--sample-per-team-year",
        type=int,
        default=5,
        help=(
            "Number of files to inspect from each team-year group. "
            "Default: 5, for approximately 20 files total."
        ),
    )

    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=2000,
        help=(
            "Number of evenly spaced rows retained from each file. "
            "Default: 2000."
        ),
    )

    parser.add_argument(
        "--example-rows-per-file",
        type=int,
        default=5,
        help=(
            "Number of sampled rows from each file written to the "
            "example-row CSV."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample_per_team_year < 1:
        raise ValueError(
            "--sample-per-team-year must be at least 1"
        )

    if args.rows_per_file < 2:
        raise ValueError(
            "--rows-per-file must be at least 2"
        )

    print(f"Reading manifest: {args.manifest.resolve()}")

    rows = load_manifest(args.manifest)

    selected_rows = select_sample_rows(
        rows,
        args.sample_per_team_year,
    )

    print(
        f"Inspecting values from {len(selected_rows):,} "
        "representative Parquet files..."
    )

    file_records: list[FileValueRecord] = []
    column_records: list[dict[str, object]] = []
    example_frames: list[pd.DataFrame] = []

    for index, row in enumerate(selected_rows, start=1):
        record, summaries, frame = inspect_file(
            row,
            args.rows_per_file,
        )

        file_records.append(record)

        for summary in summaries:
            column_records.append(
                {
                    "filename": record.filename,
                    "team": record.team,
                    "year": record.year,
                    "player_id": record.player_id,
                    **summary,
                }
            )

        if frame is not None:
            example = frame.head(
                args.example_rows_per_file
            ).copy()

            example.insert(0, "source_filename", record.filename)
            example.insert(1, "source_team", record.team)
            example.insert(2, "source_year", record.year)

            example_frames.append(example)

        print(
            f"Inspected {index:,}/"
            f"{len(selected_rows):,} files"
        )

    file_audit_path = (
        args.output_dir
        / "objective_value_file_audit.csv"
    )

    column_audit_path = (
        args.output_dir
        / "objective_value_column_audit.csv"
    )

    examples_path = (
        args.output_dir
        / "objective_sample_rows.csv"
    )

    summary_path = (
        args.output_dir
        / "objective_value_summary.txt"
    )

    write_file_audit(file_records, file_audit_path)
    write_column_audit(column_records, column_audit_path)
    write_example_rows(example_frames, examples_path)
    write_summary(file_records, summary_path)

    print(f"File audit written to: {file_audit_path}")
    print(f"Column audit written to: {column_audit_path}")
    print(f"Example rows written to: {examples_path}")


if __name__ == "__main__":
    main()