from __future__ import annotations

# Lines 3-9: Standard-library imports
import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

# Lines 11-12: PyArrow reads Parquet metadata efficiently
import pyarrow.parquet as pq


# Line 16: Resolve the repository root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SchemaRecord:
    path: str
    filename: str
    team: str
    year: str
    player_id: str
    row_count: int
    column_count: int
    row_groups: int
    schema_id: str
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


def schema_signature(schema_text: str) -> str:
    """Create a short, stable identifier for a Parquet schema."""
    return hashlib.sha256(
        schema_text.encode("utf-8")
    ).hexdigest()[:12]


def select_sample_rows(
    rows: list[dict[str, str]],
    sample_per_team_year: int,
) -> list[dict[str, str]]:
    """
    Select a spread of files from each team-year group.

    The sampling is deterministic so repeated runs inspect the same files.
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
                round(
                    index * last_index / (sample_per_team_year - 1)
                )
                for index in range(sample_per_team_year)
            ]

        selected.extend(group_rows[index] for index in indices)

    return selected


def inspect_parquet_file(
    row: dict[str, str],
) -> tuple[SchemaRecord, str | None]:
    """Read metadata and schema from one Parquet file."""
    path = Path(row["path"])

    try:
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        schema_text = str(parquet_file.schema_arrow)
        schema_id = schema_signature(schema_text)

        record = SchemaRecord(
            path=str(path),
            filename=path.name,
            team=row["team"],
            year=row["year"],
            player_id=row["player_id"],
            row_count=metadata.num_rows,
            column_count=metadata.num_columns,
            row_groups=metadata.num_row_groups,
            schema_id=schema_id,
            error="",
        )

        return record, schema_text

    except Exception as exc:
        record = SchemaRecord(
            path=str(path),
            filename=path.name,
            team=row["team"],
            year=row["year"],
            player_id=row["player_id"],
            row_count=0,
            column_count=0,
            row_groups=0,
            schema_id="ERROR",
            error=str(exc),
        )

        return record, None


def write_file_audit(
    records: list[SchemaRecord],
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
        "row_count",
        "column_count",
        "row_groups",
        "schema_id",
        "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def write_column_summary(
    schema_examples: dict[str, str],
    output_path: Path,
) -> None:
    """Write the full PyArrow schema for each unique schema ID."""
    lines: list[str] = []

    for schema_id in sorted(schema_examples):
        lines.append("=" * 80)
        lines.append(f"Schema ID: {schema_id}")
        lines.append("=" * 80)
        lines.append(schema_examples[schema_id])
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_summary(
    records: list[SchemaRecord],
    schema_examples: dict[str, str],
    output_path: Path,
) -> None:
    """Write a readable schema-audit summary."""
    successful = [
        record for record in records
        if not record.error
    ]

    failed = [
        record for record in records
        if record.error
    ]

    schema_counts = Counter(
        record.schema_id for record in successful
    )

    row_counts = [
        record.row_count for record in successful
    ]

    column_counts = [
        record.column_count for record in successful
    ]

    row_group_counts = [
        record.row_groups for record in successful
    ]

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("SoccerMon Objective Parquet Schema Audit")
    lines.append("=" * 80)
    lines.append(f"Files inspected: {len(records):,}")
    lines.append(f"Successful reads: {len(successful):,}")
    lines.append(f"Failed reads: {len(failed):,}")
    lines.append(f"Unique schemas: {len(schema_counts):,}")

    if row_counts:
        lines.append("")
        lines.append("Row-count statistics")
        lines.append("-" * 80)
        lines.append(f"Minimum rows: {min(row_counts):,}")
        lines.append(f"Maximum rows: {max(row_counts):,}")
        lines.append(
            f"Mean rows: {sum(row_counts) / len(row_counts):,.2f}"
        )

    if column_counts:
        lines.append("")
        lines.append("Column-count statistics")
        lines.append("-" * 80)
        lines.append(f"Minimum columns: {min(column_counts):,}")
        lines.append(f"Maximum columns: {max(column_counts):,}")

    if row_group_counts:
        lines.append("")
        lines.append("Row-group statistics")
        lines.append("-" * 80)
        lines.append(f"Minimum row groups: {min(row_group_counts):,}")
        lines.append(f"Maximum row groups: {max(row_group_counts):,}")

    lines.append("")
    lines.append("Schema frequencies")
    lines.append("-" * 80)

    for schema_id, count in schema_counts.most_common():
        lines.append(f"{schema_id}: {count:,} files")

    if failed:
        lines.append("")
        lines.append("Files that could not be read")
        lines.append("-" * 80)

        for record in failed:
            lines.append(
                f"{record.path}: {record.error}"
            )

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
            "Inspect a representative sample of SoccerMon objective "
            "Parquet files without loading the full dataset."
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
        help="Directory for schema-audit outputs.",
    )

    parser.add_argument(
        "--sample-per-team-year",
        type=int,
        default=25,
        help=(
            "Number of files to inspect from each team-year group. "
            "Default: 25, for about 100 files total."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample_per_team_year < 1:
        raise ValueError(
            "--sample-per-team-year must be at least 1"
        )

    print(f"Reading manifest: {args.manifest.resolve()}")

    rows = load_manifest(args.manifest)

    selected_rows = select_sample_rows(
        rows,
        args.sample_per_team_year,
    )

    print(
        f"Inspecting {len(selected_rows):,} representative "
        "Parquet files..."
    )

    records: list[SchemaRecord] = []
    schema_examples: dict[str, str] = {}

    for index, row in enumerate(selected_rows, start=1):
        record, schema_text = inspect_parquet_file(row)
        records.append(record)

        if schema_text is not None:
            schema_examples.setdefault(
                record.schema_id,
                schema_text,
            )

        if index % 25 == 0 or index == len(selected_rows):
            print(
                f"Inspected {index:,}/"
                f"{len(selected_rows):,} files"
            )

    file_audit_path = (
        args.output_dir
        / "objective_schema_file_audit.csv"
    )

    schemas_path = (
        args.output_dir
        / "objective_schema_definitions.txt"
    )

    summary_path = (
        args.output_dir
        / "objective_schema_summary.txt"
    )

    write_file_audit(records, file_audit_path)
    write_column_summary(schema_examples, schemas_path)
    write_summary(
        records,
        schema_examples,
        summary_path,
    )

    print(f"File audit written to: {file_audit_path}")
    print(f"Schema definitions written to: {schemas_path}")


if __name__ == "__main__":
    main()