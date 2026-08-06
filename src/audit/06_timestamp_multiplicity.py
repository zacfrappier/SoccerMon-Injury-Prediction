from __future__ import annotations

# Lines 3-10: Standard-library imports
import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

# Lines 12-14: Third-party imports
import pandas as pd
import pyarrow.parquet as pq


# Line 18: Resolve the repository root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Lines 22-35: Columns used in the timestamp-group audit
VALUE_COLUMNS = [
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


# Lines 39-68: One summary record per inspected file
@dataclass
class MultiplicityRecord:
    path: str
    filename: str
    team: str
    year: str
    player_id: str
    total_rows: int
    rows_inspected: int
    unique_timestamps: int
    duplicate_timestamp_groups: int
    minimum_rows_per_timestamp: int
    maximum_rows_per_timestamp: int
    median_rows_per_timestamp: float
    mode_rows_per_timestamp: int
    mean_rows_per_timestamp: float
    first_time: str
    last_time: str
    block_duration_seconds: float | None
    unique_timestamps_per_second: float | None
    estimated_rows_per_second: float | None
    invalid_timestamp_rows: int
    heart_rate_zero_fraction: float
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
    Select representative files from every team-year group.

    Selection is deterministic and distributed across the season.
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


def parse_time_column(time_series: pd.Series) -> pd.Series:
    """Parse SoccerMon time-of-day strings such as 13:48:39.6."""
    return pd.to_datetime(
        time_series.astype("string").str.strip(),
        format="%H:%M:%S.%f",
        errors="coerce",
    )


def mode_integer(values: pd.Series) -> int:
    """Return the most common integer value."""
    counts = Counter(int(value) for value in values)

    if not counts:
        return 0

    return counts.most_common(1)[0][0]


def read_contiguous_block(
    path: Path,
    start_row: int,
    rows_to_read: int,
) -> tuple[pd.DataFrame, int]:
    """
    Read selected columns and retain one contiguous block.

    SoccerMon files contain one row group, so the required columns are
    read once and then sliced in pandas.
    """
    parquet_file = pq.ParquetFile(path)
    total_rows = parquet_file.metadata.num_rows

    columns = [
        "player_name",
        "time",
        *VALUE_COLUMNS,
    ]

    table = parquet_file.read(columns=columns)
    frame = table.to_pandas()

    if start_row >= total_rows:
        raise ValueError(
            f"start_row {start_row:,} exceeds file length "
            f"{total_rows:,}"
        )

    end_row = min(
        start_row + rows_to_read,
        total_rows,
    )

    block = frame.iloc[start_row:end_row].copy()

    return block, total_rows


def summarize_column_within_timestamps(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, object]:
    """
    Measure how often a column varies within duplicate timestamp groups.

    If GPS and heart-rate values repeat while accelerometer and gyroscope
    values change, this suggests multiple IMU samples per timestamp.
    """
    grouped = frame.groupby(
        "_parsed_time",
        dropna=True,
        sort=False,
    )[column]

    unique_counts = grouped.nunique(dropna=False)

    group_count = len(unique_counts)

    varying_groups = int(
        (unique_counts > 1).sum()
    )

    constant_groups = int(
        (unique_counts <= 1).sum()
    )

    if group_count > 0:
        varying_fraction = varying_groups / group_count
        mean_unique_values = float(unique_counts.mean())
        median_unique_values = float(unique_counts.median())
        maximum_unique_values = int(unique_counts.max())
    else:
        varying_fraction = 0.0
        mean_unique_values = 0.0
        median_unique_values = 0.0
        maximum_unique_values = 0

    numeric = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    return {
        "column": column,
        "timestamp_groups": group_count,
        "constant_groups": constant_groups,
        "varying_groups": varying_groups,
        "varying_group_fraction": varying_fraction,
        "mean_unique_values_per_timestamp": mean_unique_values,
        "median_unique_values_per_timestamp": median_unique_values,
        "maximum_unique_values_per_timestamp": maximum_unique_values,
        "zero_rows": int((numeric == 0).sum()),
        "missing_rows": int(numeric.isna().sum()),
    }


def inspect_file(
    row: dict[str, str],
    start_row: int,
    rows_to_read: int,
) -> tuple[
    MultiplicityRecord,
    list[dict[str, object]],
    pd.DataFrame | None,
]:
    """Inspect timestamp multiplicity and within-group sensor variation."""
    path = Path(row["path"])

    try:
        frame, total_rows = read_contiguous_block(
            path=path,
            start_row=start_row,
            rows_to_read=rows_to_read,
        )

        parsed_times = parse_time_column(
            frame["time"]
        )

        frame["_parsed_time"] = parsed_times

        valid_frame = frame[
            frame["_parsed_time"].notna()
        ].copy()

        invalid_timestamp_rows = int(
            frame["_parsed_time"].isna().sum()
        )

        timestamp_counts = (
            valid_frame
            .groupby("_parsed_time", sort=False)
            .size()
        )

        unique_timestamps = int(
            len(timestamp_counts)
        )

        duplicate_timestamp_groups = int(
            (timestamp_counts > 1).sum()
        )

        if not timestamp_counts.empty:
            minimum_rows = int(
                timestamp_counts.min()
            )
            maximum_rows = int(
                timestamp_counts.max()
            )
            median_rows = float(
                timestamp_counts.median()
            )
            mode_rows = mode_integer(
                timestamp_counts
            )
            mean_rows = float(
                timestamp_counts.mean()
            )
        else:
            minimum_rows = 0
            maximum_rows = 0
            median_rows = 0.0
            mode_rows = 0
            mean_rows = 0.0

        valid_times = (
            valid_frame["_parsed_time"]
            .dropna()
            .sort_values()
        )

        if len(valid_times) >= 2:
            first_timestamp = (
                valid_times.iloc[0]
            )
            last_timestamp = (
                valid_times.iloc[-1]
            )

            block_duration = float(
                (
                    last_timestamp
                    - first_timestamp
                ).total_seconds()
            )
        else:
            first_timestamp = None
            last_timestamp = None
            block_duration = None

        if (
            block_duration is not None
            and block_duration > 0
        ):
            unique_timestamps_per_second = (
                unique_timestamps
                / block_duration
            )

            estimated_rows_per_second = (
                len(valid_frame)
                / block_duration
            )
        else:
            unique_timestamps_per_second = None
            estimated_rows_per_second = None

        heart_rate = pd.to_numeric(
            frame["heart_rate"],
            errors="coerce",
        )

        if len(frame) > 0:
            heart_rate_zero_fraction = float(
                (heart_rate == 0).mean()
            )
        else:
            heart_rate_zero_fraction = 0.0

        column_summaries: list[
            dict[str, object]
        ] = []

        for column in VALUE_COLUMNS:
            summary = (
                summarize_column_within_timestamps(
                    valid_frame,
                    column,
                )
            )

            column_summaries.append(
                {
                    "filename": path.name,
                    "team": row["team"],
                    "year": row["year"],
                    "player_id": row["player_id"],
                    **summary,
                }
            )

        record = MultiplicityRecord(
            path=str(path),
            filename=path.name,
            team=row["team"],
            year=row["year"],
            player_id=row["player_id"],
            total_rows=total_rows,
            rows_inspected=len(frame),
            unique_timestamps=unique_timestamps,
            duplicate_timestamp_groups=(
                duplicate_timestamp_groups
            ),
            minimum_rows_per_timestamp=(
                minimum_rows
            ),
            maximum_rows_per_timestamp=(
                maximum_rows
            ),
            median_rows_per_timestamp=(
                median_rows
            ),
            mode_rows_per_timestamp=mode_rows,
            mean_rows_per_timestamp=mean_rows,
            first_time=(
                str(frame["time"].iloc[0])
                if not frame.empty
                else ""
            ),
            last_time=(
                str(frame["time"].iloc[-1])
                if not frame.empty
                else ""
            ),
            block_duration_seconds=(
                block_duration
            ),
            unique_timestamps_per_second=(
                unique_timestamps_per_second
            ),
            estimated_rows_per_second=(
                estimated_rows_per_second
            ),
            invalid_timestamp_rows=(
                invalid_timestamp_rows
            ),
            heart_rate_zero_fraction=(
                heart_rate_zero_fraction
            ),
            error="",
        )

        example_columns = [
            "player_name",
            "time",
            *VALUE_COLUMNS,
        ]

        example = (
            valid_frame[example_columns]
            .head(100)
            .copy()
        )

        example.insert(
            0,
            "source_filename",
            path.name,
        )

        return (
            record,
            column_summaries,
            example,
        )

    except Exception as exc:
        record = MultiplicityRecord(
            path=str(path),
            filename=path.name,
            team=row["team"],
            year=row["year"],
            player_id=row["player_id"],
            total_rows=0,
            rows_inspected=0,
            unique_timestamps=0,
            duplicate_timestamp_groups=0,
            minimum_rows_per_timestamp=0,
            maximum_rows_per_timestamp=0,
            median_rows_per_timestamp=0.0,
            mode_rows_per_timestamp=0,
            mean_rows_per_timestamp=0.0,
            first_time="",
            last_time="",
            block_duration_seconds=None,
            unique_timestamps_per_second=None,
            estimated_rows_per_second=None,
            invalid_timestamp_rows=0,
            heart_rate_zero_fraction=0.0,
            error=str(exc),
        )

        return record, [], None


def write_file_audit(
    records: list[MultiplicityRecord],
    output_path: Path,
) -> None:
    """Write one timestamp-multiplicity row per file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        MultiplicityRecord.__dataclass_fields__.keys()
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


def write_column_audit(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write within-timestamp variation summaries by column."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "filename",
        "team",
        "year",
        "player_id",
        "column",
        "timestamp_groups",
        "constant_groups",
        "varying_groups",
        "varying_group_fraction",
        "mean_unique_values_per_timestamp",
        "median_unique_values_per_timestamp",
        "maximum_unique_values_per_timestamp",
        "zero_rows",
        "missing_rows",
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


def write_examples(
    frames: list[pd.DataFrame],
    output_path: Path,
) -> None:
    """Write representative duplicate-timestamp rows."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not frames:
        output_path.write_text(
            "",
            encoding="utf-8",
        )
        return

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined.to_csv(
        output_path,
        index=False,
    )


def write_summary(
    records: list[MultiplicityRecord],
    column_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write a readable timestamp-multiplicity report."""
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

    mode_rows = [
        record.mode_rows_per_timestamp
        for record in successful
        if record.mode_rows_per_timestamp > 0
    ]

    unique_timestamp_rates = [
        record.unique_timestamps_per_second
        for record in successful
        if record.unique_timestamps_per_second
        is not None
    ]

    row_rates = [
        record.estimated_rows_per_second
        for record in successful
        if record.estimated_rows_per_second
        is not None
    ]

    column_groups: dict[
        str,
        list[float],
    ] = defaultdict(list)

    for row in column_rows:
        column_groups[
            str(row["column"])
        ].append(
            float(
                row[
                    "varying_group_fraction"
                ]
            )
        )

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append(
        "SoccerMon Objective Timestamp Multiplicity Audit"
    )
    lines.append("=" * 80)

    lines.append(
        f"Files inspected: {len(records):,}"
    )
    lines.append(
        f"Successful reads: {len(successful):,}"
    )
    lines.append(
        f"Failed reads: {len(failed):,}"
    )

    if successful:
        lines.append(
            "Files containing duplicate timestamp groups: "
            f"{sum(record.duplicate_timestamp_groups > 0 for record in successful):,}"
            f"/{len(successful):,}"
        )

        lines.append(
            "Files containing invalid timestamps: "
            f"{sum(record.invalid_timestamp_rows > 0 for record in successful):,}"
        )

    if mode_rows:
        lines.append("")
        lines.append(
            "Rows per timestamp"
        )
        lines.append("-" * 80)

        counts = Counter(mode_rows)

        for value, count in counts.most_common():
            lines.append(
                f"Mode {value} rows per timestamp: "
                f"{count:,} files"
            )

        lines.append(
            "Median file-level mode: "
            f"{median(mode_rows):.2f}"
        )

    if unique_timestamp_rates:
        lines.append("")
        lines.append(
            "Unique timestamps per second"
        )
        lines.append("-" * 80)

        lines.append(
            f"Minimum: "
            f"{min(unique_timestamp_rates):.3f}"
        )
        lines.append(
            f"Maximum: "
            f"{max(unique_timestamp_rates):.3f}"
        )
        lines.append(
            f"Median: "
            f"{median(unique_timestamp_rates):.3f}"
        )

    if row_rates:
        lines.append("")
        lines.append(
            "Estimated rows per second"
        )
        lines.append("-" * 80)

        lines.append(
            f"Minimum: {min(row_rates):.3f}"
        )
        lines.append(
            f"Maximum: {max(row_rates):.3f}"
        )
        lines.append(
            f"Median: {median(row_rates):.3f}"
        )

    if column_groups:
        lines.append("")
        lines.append(
            "Average fraction of timestamp groups "
            "where each column varies"
        )
        lines.append("-" * 80)

        for column in VALUE_COLUMNS:
            fractions = column_groups.get(
                column,
                [],
            )

            if fractions:
                lines.append(
                    f"{column}: "
                    f"{sum(fractions) / len(fractions):.4f}"
                )

    lines.append("")
    lines.append("Per-file results")
    lines.append("-" * 80)

    for record in successful:
        lines.append(
            f"{record.team} {record.year} | "
            f"{record.filename} | "
            f"mode rows/timestamp: "
            f"{record.mode_rows_per_timestamp} | "
            f"timestamps/sec: "
            f"{record.unique_timestamps_per_second} | "
            f"rows/sec: "
            f"{record.estimated_rows_per_second}"
        )

    if failed:
        lines.append("")
        lines.append(
            "Files that could not be inspected"
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

    print("\n".join(lines))
    print()
    print(
        f"Summary written to: {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure rows per timestamp and sensor variation "
            "within SoccerMon objective-data timestamp groups."
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
        help=(
            "Path to the objective-file manifest."
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
            "Directory for multiplicity-audit outputs."
        ),
    )

    parser.add_argument(
        "--sample-per-team-year",
        type=int,
        default=3,
        help=(
            "Files inspected per team-year group. "
            "Default: 3, approximately 12 total."
        ),
    )

    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help=(
            "Starting row of the contiguous block. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=10000,
        help=(
            "Number of contiguous rows inspected per file. "
            "Default: 10000."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample_per_team_year < 1:
        raise ValueError(
            "--sample-per-team-year must be at least 1"
        )

    if args.start_row < 0:
        raise ValueError(
            "--start-row cannot be negative"
        )

    if args.rows_per_file < 2:
        raise ValueError(
            "--rows-per-file must be at least 2"
        )

    print(
        f"Reading manifest: "
        f"{args.manifest.resolve()}"
    )

    rows = load_manifest(
        args.manifest
    )

    selected_rows = select_sample_rows(
        rows,
        args.sample_per_team_year,
    )

    print(
        f"Inspecting timestamp multiplicity in "
        f"{len(selected_rows):,} Parquet files..."
    )

    file_records: list[
        MultiplicityRecord
    ] = []

    column_records: list[
        dict[str, object]
    ] = []

    example_frames: list[
        pd.DataFrame
    ] = []

    for index, row in enumerate(
        selected_rows,
        start=1,
    ):
        (
            record,
            summaries,
            example,
        ) = inspect_file(
            row=row,
            start_row=args.start_row,
            rows_to_read=args.rows_per_file,
        )

        file_records.append(record)
        column_records.extend(summaries)

        if example is not None:
            example_frames.append(example)

        print(
            f"Inspected {index:,}/"
            f"{len(selected_rows):,} files"
        )

    file_audit_path = (
        args.output_dir
        / "objective_timestamp_multiplicity_file_audit.csv"
    )

    column_audit_path = (
        args.output_dir
        / "objective_timestamp_column_variation.csv"
    )

    examples_path = (
        args.output_dir
        / "objective_timestamp_examples.csv"
    )

    summary_path = (
        args.output_dir
        / "objective_timestamp_multiplicity_summary.txt"
    )

    write_file_audit(
        file_records,
        file_audit_path,
    )

    write_column_audit(
        column_records,
        column_audit_path,
    )

    write_examples(
        example_frames,
        examples_path,
    )

    write_summary(
        records=file_records,
        column_rows=column_records,
        output_path=summary_path,
    )

    print(
        f"File audit written to: "
        f"{file_audit_path}"
    )

    print(
        f"Column audit written to: "
        f"{column_audit_path}"
    )

    print(
        f"Example rows written to: "
        f"{examples_path}"
    )


if __name__ == "__main__":
    main()