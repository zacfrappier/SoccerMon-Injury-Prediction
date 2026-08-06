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


# Lines 22-42: One record for each inspected Parquet file
@dataclass
class SamplingRecord:
    path: str
    filename: str
    team: str
    year: str
    player_id: str
    expected_player_name: str
    observed_player_names: str
    player_name_matches: bool
    total_rows: int
    rows_inspected: int
    first_time: str
    last_time: str
    block_duration_seconds: float | None
    valid_timestamp_rows: int
    invalid_timestamp_rows: int
    duplicate_intervals: int
    backward_intervals: int
    positive_intervals: int
    median_interval_seconds: float | None
    mode_interval_seconds: float | None
    estimated_frequency_hz: float | None
    minimum_interval_seconds: float | None
    maximum_interval_seconds: float | None
    large_gap_count: int
    largest_gap_seconds: float | None
    heart_rate_zero_rows: int
    suspicious_nonzero_heart_rate_rows: int
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
    Select deterministic files from each team-year group.

    Files are distributed across the available dates rather than selected
    only from the beginning of each season.
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
    """
    Parse SoccerMon time values such as 13:48:39.6.

    The objective files contain time-of-day strings rather than complete
    timestamps with calendar dates.
    """
    cleaned = time_series.astype("string").str.strip()

    return pd.to_datetime(
        cleaned,
        format="%H:%M:%S.%f",
        errors="coerce",
    )


def interval_mode(intervals: pd.Series) -> float | None:
    """
    Return the most common positive interval.

    Intervals are rounded to milliseconds before counting to avoid tiny
    floating-point differences.
    """
    if intervals.empty:
        return None

    rounded = intervals.round(3)
    counts = Counter(rounded.tolist())

    if not counts:
        return None

    return float(counts.most_common(1)[0][0])


def read_contiguous_block(
    path: Path,
    start_row: int,
    rows_to_read: int,
) -> tuple[pd.DataFrame, int]:
    """
    Read a Parquet file, then retain one contiguous block.

    SoccerMon files have one row group, so PyArrow reads the selected
    columns from that row group and pandas slices the contiguous rows.
    """
    parquet_file = pq.ParquetFile(path)
    total_rows = parquet_file.metadata.num_rows

    columns = [
        "player_name",
        "time",
        "heart_rate",
    ]

    table = parquet_file.read(columns=columns)
    frame = table.to_pandas()

    if start_row >= total_rows:
        raise ValueError(
            f"start_row {start_row:,} exceeds file length "
            f"{total_rows:,}"
        )

    end_row = min(start_row + rows_to_read, total_rows)

    block = frame.iloc[start_row:end_row].copy()

    return block, total_rows


def inspect_file(
    row: dict[str, str],
    start_row: int,
    rows_to_read: int,
    large_gap_multiplier: float,
) -> tuple[SamplingRecord, pd.DataFrame | None]:
    """Inspect timestamp cadence in one contiguous file block."""
    path = Path(row["path"])

    try:
        frame, total_rows = read_contiguous_block(
            path=path,
            start_row=start_row,
            rows_to_read=rows_to_read,
        )

        parsed_times = parse_time_column(frame["time"])

        valid_timestamp_rows = int(parsed_times.notna().sum())
        invalid_timestamp_rows = int(parsed_times.isna().sum())

        valid_times = parsed_times.dropna()

        differences = valid_times.diff().dt.total_seconds().dropna()

        duplicate_intervals = int((differences == 0).sum())
        backward_intervals = int((differences < 0).sum())

        positive_intervals = differences[differences > 0]

        median_interval = (
            float(positive_intervals.median())
            if not positive_intervals.empty
            else None
        )

        mode_interval = interval_mode(positive_intervals)

        estimated_frequency = (
            1.0 / mode_interval
            if mode_interval is not None and mode_interval > 0
            else None
        )

        minimum_interval = (
            float(positive_intervals.min())
            if not positive_intervals.empty
            else None
        )

        maximum_interval = (
            float(positive_intervals.max())
            if not positive_intervals.empty
            else None
        )

        if median_interval is not None:
            large_gap_threshold = (
                median_interval * large_gap_multiplier
            )

            large_gaps = positive_intervals[
                positive_intervals > large_gap_threshold
            ]
        else:
            large_gaps = pd.Series(dtype="float64")

        large_gap_count = int(len(large_gaps))

        largest_gap = (
            float(large_gaps.max())
            if not large_gaps.empty
            else None
        )

        if len(valid_times) >= 2:
            block_duration = float(
                (
                    valid_times.iloc[-1]
                    - valid_times.iloc[0]
                ).total_seconds()
            )
        else:
            block_duration = None

        observed_player_names = sorted(
            frame["player_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        expected_player_name = (
            f"{row['team']}-{row['player_id']}"
        )

        player_name_matches = (
            len(observed_player_names) == 1
            and observed_player_names[0] == expected_player_name
        )

        heart_rate = pd.to_numeric(
            frame["heart_rate"],
            errors="coerce",
        )

        heart_rate_zero_rows = int(
            (heart_rate == 0).sum()
        )

        suspicious_nonzero_heart_rate_rows = int(
            (
                ((heart_rate > 0) & (heart_rate < 30))
                | (heart_rate > 240)
            ).sum()
        )

        record = SamplingRecord(
            path=str(path),
            filename=path.name,
            team=row["team"],
            year=row["year"],
            player_id=row["player_id"],
            expected_player_name=expected_player_name,
            observed_player_names=";".join(observed_player_names),
            player_name_matches=player_name_matches,
            total_rows=total_rows,
            rows_inspected=len(frame),
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
            block_duration_seconds=block_duration,
            valid_timestamp_rows=valid_timestamp_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            duplicate_intervals=duplicate_intervals,
            backward_intervals=backward_intervals,
            positive_intervals=int(len(positive_intervals)),
            median_interval_seconds=median_interval,
            mode_interval_seconds=mode_interval,
            estimated_frequency_hz=estimated_frequency,
            minimum_interval_seconds=minimum_interval,
            maximum_interval_seconds=maximum_interval,
            large_gap_count=large_gap_count,
            largest_gap_seconds=largest_gap,
            heart_rate_zero_rows=heart_rate_zero_rows,
            suspicious_nonzero_heart_rate_rows=(
                suspicious_nonzero_heart_rate_rows
            ),
            error="",
        )

        interval_frame = pd.DataFrame(
            {
                "source_filename": path.name,
                "source_team": row["team"],
                "source_year": row["year"],
                "row_number": frame.index,
                "raw_time": frame["time"].values,
                "parsed_time": parsed_times.astype("string").values,
                "interval_seconds": (
                    parsed_times.diff()
                    .dt.total_seconds()
                    .values
                ),
                "heart_rate": heart_rate.values,
            }
        )

        return record, interval_frame

    except Exception as exc:
        record = SamplingRecord(
            path=str(path),
            filename=path.name,
            team=row["team"],
            year=row["year"],
            player_id=row["player_id"],
            expected_player_name=(
                f"{row['team']}-{row['player_id']}"
            ),
            observed_player_names="",
            player_name_matches=False,
            total_rows=0,
            rows_inspected=0,
            first_time="",
            last_time="",
            block_duration_seconds=None,
            valid_timestamp_rows=0,
            invalid_timestamp_rows=0,
            duplicate_intervals=0,
            backward_intervals=0,
            positive_intervals=0,
            median_interval_seconds=None,
            mode_interval_seconds=None,
            estimated_frequency_hz=None,
            minimum_interval_seconds=None,
            maximum_interval_seconds=None,
            large_gap_count=0,
            largest_gap_seconds=None,
            heart_rate_zero_rows=0,
            suspicious_nonzero_heart_rate_rows=0,
            error=str(exc),
        )

        return record, None


def write_file_audit(
    records: list[SamplingRecord],
    output_path: Path,
) -> None:
    """Write one sampling summary row per inspected file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        field.name
        for field in SamplingRecord.__dataclass_fields__.values()
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

        for record in records:
            writer.writerow(asdict(record))


def write_interval_samples(
    frames: list[pd.DataFrame],
    output_path: Path,
) -> None:
    """Write contiguous timestamp rows for manual inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        output_path.write_text("", encoding="utf-8")
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
    records: list[SamplingRecord],
    output_path: Path,
) -> None:
    """Write a readable sampling-frequency report."""
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

    frequencies = [
        record.estimated_frequency_hz
        for record in successful
        if record.estimated_frequency_hz is not None
    ]

    median_intervals = [
        record.median_interval_seconds
        for record in successful
        if record.median_interval_seconds is not None
    ]

    mode_intervals = [
        record.mode_interval_seconds
        for record in successful
        if record.mode_interval_seconds is not None
    ]

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("SoccerMon Objective Sampling-Frequency Audit")
    lines.append("=" * 80)
    lines.append(f"Files inspected: {len(records):,}")
    lines.append(f"Successful reads: {len(successful):,}")
    lines.append(f"Failed reads: {len(failed):,}")

    if successful:
        lines.append(
            "Player names matching team-prefixed filename ID: "
            f"{sum(record.player_name_matches for record in successful):,}"
            f"/{len(successful):,}"
        )

        lines.append(
            "Files with invalid timestamps: "
            f"{sum(record.invalid_timestamp_rows > 0 for record in successful):,}"
        )

        lines.append(
            "Files with duplicate timestamp intervals: "
            f"{sum(record.duplicate_intervals > 0 for record in successful):,}"
        )

        lines.append(
            "Files with backward timestamp intervals: "
            f"{sum(record.backward_intervals > 0 for record in successful):,}"
        )

        lines.append(
            "Files with large timestamp gaps: "
            f"{sum(record.large_gap_count > 0 for record in successful):,}"
        )

        lines.append(
            "Files containing heart-rate zero values: "
            f"{sum(record.heart_rate_zero_rows > 0 for record in successful):,}"
        )

        lines.append(
            "Files with suspicious nonzero heart rates: "
            f"{sum(record.suspicious_nonzero_heart_rate_rows > 0 for record in successful):,}"
        )

    if median_intervals:
        lines.append("")
        lines.append("Median consecutive intervals")
        lines.append("-" * 80)
        lines.append(
            f"Minimum: {min(median_intervals):.6f} seconds"
        )
        lines.append(
            f"Maximum: {max(median_intervals):.6f} seconds"
        )
        lines.append(
            f"Overall median: {median(median_intervals):.6f} seconds"
        )

    if mode_intervals:
        lines.append("")
        lines.append("Most common consecutive intervals")
        lines.append("-" * 80)

        interval_counts = Counter(
            round(value, 6)
            for value in mode_intervals
        )

        for interval, count in interval_counts.most_common():
            lines.append(
                f"{interval:.6f} seconds: {count:,} files"
            )

    if frequencies:
        lines.append("")
        lines.append("Estimated recording frequencies")
        lines.append("-" * 80)
        lines.append(
            f"Minimum: {min(frequencies):.3f} Hz"
        )
        lines.append(
            f"Maximum: {max(frequencies):.3f} Hz"
        )
        lines.append(
            f"Median: {median(frequencies):.3f} Hz"
        )

    lines.append("")
    lines.append("Per-file results")
    lines.append("-" * 80)

    for record in successful:
        lines.append(
            f"{record.team} {record.year} | "
            f"{record.filename} | "
            f"median interval: "
            f"{record.median_interval_seconds} s | "
            f"mode interval: "
            f"{record.mode_interval_seconds} s | "
            f"estimated frequency: "
            f"{record.estimated_frequency_hz} Hz | "
            f"large gaps: {record.large_gap_count}"
        )

    if failed:
        lines.append("")
        lines.append("Files that could not be inspected")
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
    print(f"Summary written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect contiguous objective-data rows to estimate "
            "SoccerMon recording frequency and timestamp quality."
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
        help="Directory for sampling-audit outputs.",
    )

    parser.add_argument(
        "--sample-per-team-year",
        type=int,
        default=3,
        help=(
            "Files inspected per team-year group. "
            "Default: 3, for approximately 12 files total."
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

    parser.add_argument(
        "--large-gap-multiplier",
        type=float,
        default=10.0,
        help=(
            "A gap larger than this multiple of the median interval "
            "is flagged. Default: 10."
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

    if args.start_row < 0:
        raise ValueError(
            "--start-row cannot be negative"
        )

    if args.large_gap_multiplier <= 1:
        raise ValueError(
            "--large-gap-multiplier must be greater than 1"
        )

    print(f"Reading manifest: {args.manifest.resolve()}")

    rows = load_manifest(args.manifest)

    selected_rows = select_sample_rows(
        rows,
        args.sample_per_team_year,
    )

    print(
        f"Inspecting contiguous blocks from "
        f"{len(selected_rows):,} Parquet files..."
    )

    records: list[SamplingRecord] = []
    interval_frames: list[pd.DataFrame] = []

    for index, row in enumerate(
        selected_rows,
        start=1,
    ):
        record, interval_frame = inspect_file(
            row=row,
            start_row=args.start_row,
            rows_to_read=args.rows_per_file,
            large_gap_multiplier=args.large_gap_multiplier,
        )

        records.append(record)

        if interval_frame is not None:
            interval_frames.append(
                interval_frame.head(200)
            )

        print(
            f"Inspected {index:,}/"
            f"{len(selected_rows):,} files"
        )

    file_audit_path = (
        args.output_dir
        / "objective_sampling_file_audit.csv"
    )

    interval_samples_path = (
        args.output_dir
        / "objective_sampling_interval_samples.csv"
    )

    summary_path = (
        args.output_dir
        / "objective_sampling_summary.txt"
    )

    write_file_audit(
        records,
        file_audit_path,
    )

    write_interval_samples(
        interval_frames,
        interval_samples_path,
    )

    write_summary(
        records,
        summary_path,
    )

    print(
        f"File audit written to: {file_audit_path}"
    )

    print(
        "Interval samples written to: "
        f"{interval_samples_path}"
    )


if __name__ == "__main__":
    main()