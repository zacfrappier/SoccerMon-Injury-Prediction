from __future__ import annotations

# Lines 3-10: Standard-library imports
import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

# Lines 12-13: Third-party imports
import pandas as pd


# Line 17: Resolve repository root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SessionAuditRecord:
    json_path: str
    top_level_type: str
    session_records: int
    columns: str
    player_column: str
    date_column: str
    unique_players: int
    first_date: str
    last_date: str
    duplicate_rows: int
    missing_cells: int
    error: str


def find_session_json() -> Path:
    """Locate the SoccerMon subjective session.json file."""
    path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "subjective"
        / "subjective"
        / "training-load"
        / "session.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"session.json not found: {path}"
        )

    return path


def load_json(path: Path):
    """Load the raw JSON structure."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def describe_json_structure(data) -> str:
    """Return a compact description of the top-level JSON structure."""
    if isinstance(data, list):
        return "list"

    if isinstance(data, dict):
        return "dict"

    return type(data).__name__


def normalize_json(data) -> pd.DataFrame:
    """
    Normalize SoccerMon session.json into one row per player session.

    Expected structure:

    {
        "TeamA-player-id": [
            {...session 1...},
            {...session 2...},
        ],
        "TeamB-player-id": [
            {...session 1...},
            ...
        ]
    }

    The top-level dictionary key is treated as player_name.
    """

    # Case 1:
    # JSON is already a list of session dictionaries.
    if isinstance(data, list):
        return pd.json_normalize(data)

    if not isinstance(data, dict):
        raise ValueError(
            "Unsupported JSON top-level structure: "
            f"{type(data).__name__}"
        )

    # Case 2:
    # SoccerMon structure:
    # player_name -> list of session records
    if all(
        isinstance(value, list)
        for value in data.values()
    ):
        frames: list[pd.DataFrame] = []

        for player_name, sessions in data.items():

            # Player has no sessions.
            if not sessions:
                continue

            player_frame = pd.json_normalize(
                sessions
            )

            # Preserve the player ID from the JSON key.
            player_frame.insert(
                0,
                "player_name",
                player_name,
            )

            frames.append(
                player_frame
            )

        if not frames:
            return pd.DataFrame(
                columns=["player_name"]
            )

        return pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )

    # Case 3:
    # Dictionary of dictionaries.
    if all(
        isinstance(value, dict)
        for value in data.values()
    ):
        frame = pd.DataFrame.from_dict(
            data,
            orient="index",
        )

        frame.insert(
            0,
            "_json_key",
            frame.index,
        )

        frame.reset_index(
            drop=True,
            inplace=True,
        )

        return frame

    # Generic fallback.
    return pd.json_normalize(
        data
    )


def find_player_column(
    columns: list[str],
) -> str:
    """Find a likely player identifier field."""
    candidates = [
        "player_name",
        "player",
        "player_id",
        "athlete",
        "athlete_id",
        "user",
        "user_id",
    ]

    lower_map = {
        column.lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    # Fallback for nested names like athlete.name.
    for column in columns:
        lower = column.lower()

        if (
            "player" in lower
            or "athlete" in lower
        ):
            return column

    return ""


def find_date_column(
    columns: list[str],
) -> str:
    """Find a likely session date or timestamp field."""
    exact_candidates = [
        "timestamp",
        "date",
        "datetime",
        "start_time",
        "start",
        "created_at",
        "session_date",
    ]

    lower_map = {
        column.lower(): column
        for column in columns
    }

    for candidate in exact_candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    for column in columns:
        lower = column.lower()

        if (
            "date" in lower
            or "time" in lower
        ):
            return column

    return ""


def parse_dates(
    frame: pd.DataFrame,
    date_column: str,
) -> pd.Series:
    """Parse a detected session date field."""
    if not date_column:
        return pd.Series(
            dtype="datetime64[ns]"
        )

    return pd.to_datetime(
        frame[date_column],
        errors="coerce",
        dayfirst=True,
    )


def summarize_columns(
    frame: pd.DataFrame,
) -> list[dict[str, object]]:
    """Create one summary row per flattened JSON column."""
    rows: list[dict[str, object]] = []

    for column in frame.columns:
        series = frame[column]

        rows.append(
            {
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
                    series.dropna().astype(str).nunique()
                ),
                "example_values": "; ".join(
                    series.dropna()
                    .astype(str)
                    .drop_duplicates()
                    .head(5)
                    .tolist()
                ),
            }
        )

    return rows


def numeric_summary(
    frame: pd.DataFrame,
) -> list[dict[str, object]]:
    """Summarize columns that can be interpreted numerically."""
    rows: list[dict[str, object]] = []

    for column in frame.columns:
        numeric = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        valid = numeric.dropna()

        # Ignore columns that are mostly non-numeric.
        if len(frame) == 0:
            continue

        parse_fraction = (
            len(valid) / len(frame)
        )

        if parse_fraction < 0.5:
            continue

        rows.append(
            {
                "column": column,
                "numeric_parse_fraction": parse_fraction,
                "count": int(valid.count()),
                "minimum": (
                    float(valid.min())
                    if not valid.empty
                    else None
                ),
                "maximum": (
                    float(valid.max())
                    if not valid.empty
                    else None
                ),
                "mean": (
                    float(valid.mean())
                    if not valid.empty
                    else None
                ),
                "median": (
                    float(valid.median())
                    if not valid.empty
                    else None
                ),
                "zero_count": int(
                    (valid == 0).sum()
                ),
                "negative_count": int(
                    (valid < 0).sum()
                ),
            }
        )

    return rows


def write_summary(
    record: SessionAuditRecord,
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a human-readable session.json audit report."""
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("SoccerMon session.json Audit")
    lines.append("=" * 80)

    lines.append(
        f"JSON path: {record.json_path}"
    )
    lines.append(
        f"Top-level JSON type: "
        f"{record.top_level_type}"
    )
    lines.append(
        f"Normalized session records: "
        f"{record.session_records:,}"
    )
    lines.append(
        f"Columns after normalization: "
        f"{len(frame.columns):,}"
    )
    lines.append(
        f"Detected player column: "
        f"{record.player_column or 'NONE'}"
    )
    lines.append(
        f"Detected date column: "
        f"{record.date_column or 'NONE'}"
    )
    lines.append(
        f"Unique players: "
        f"{record.unique_players:,}"
    )
    lines.append(
        f"Date range: "
        f"{record.first_date or 'UNKNOWN'} "
        f"to "
        f"{record.last_date or 'UNKNOWN'}"
    )
    lines.append(
        f"Exact duplicate rows: "
        f"{record.duplicate_rows:,}"
    )
    lines.append(
        f"Missing cells: "
        f"{record.missing_cells:,}"
    )

    lines.append("")
    lines.append("Columns")
    lines.append("-" * 80)

    for column in frame.columns:
        lines.append(str(column))

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
            "Inspect SoccerMon training-load/session.json "
            "and normalize its structure for audit."
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
        help=(
            "Directory for session JSON audit outputs."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    session_path = find_session_json()

    print(
        f"Loading: {session_path}"
    )

    raw_data = load_json(
        session_path
    )

    top_level_type = (
        describe_json_structure(
            raw_data
        )
    )

    print(
        f"Top-level JSON type: "
        f"{top_level_type}"
    )

    frame = normalize_json(
        raw_data
    )

    print(
        f"Normalized to "
        f"{len(frame):,} rows and "
        f"{len(frame.columns):,} columns"
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
        unique_players = 0

    dates = parse_dates(
        frame,
        date_column,
    )

    valid_dates = dates.dropna()

    first_date = (
        str(valid_dates.min())
        if not valid_dates.empty
        else ""
    )

    last_date = (
        str(valid_dates.max())
        if not valid_dates.empty
        else ""
    )

    record = SessionAuditRecord(
        json_path=str(session_path),
        top_level_type=top_level_type,
        session_records=len(frame),
        columns=";".join(columns),
        player_column=player_column,
        date_column=date_column,
        unique_players=unique_players,
        first_date=first_date,
        last_date=last_date,
        duplicate_rows=int(
            frame.duplicated().sum()
        ),
        missing_cells=int(
            frame.isna().sum().sum()
        ),
        error="",
    )

    column_rows = summarize_columns(
        frame
    )

    numeric_rows = numeric_summary(
        frame
    )

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "session_json_summary.txt"
    )

    normalized_path = (
        output_dir
        / "session_json_normalized_preview.csv"
    )

    column_path = (
        output_dir
        / "session_json_columns.csv"
    )

    numeric_path = (
        output_dir
        / "session_json_numeric_summary.csv"
    )

    metadata_path = (
        output_dir
        / "session_json_metadata.csv"
    )

    write_summary(
        record=record,
        frame=frame,
        output_path=summary_path,
    )

    pd.DataFrame(
        [asdict(record)]
    ).to_csv(
        metadata_path,
        index=False,
    )

    pd.DataFrame(
        column_rows
    ).to_csv(
        column_path,
        index=False,
    )

    pd.DataFrame(
        numeric_rows
    ).to_csv(
        numeric_path,
        index=False,
    )

    # Only save the first 500 rows as a preview.
    frame.head(
        500
    ).to_csv(
        normalized_path,
        index=False,
    )

    print(
        f"Metadata written to: "
        f"{metadata_path}"
    )

    print(
        f"Column audit written to: "
        f"{column_path}"
    )

    print(
        f"Numeric audit written to: "
        f"{numeric_path}"
    )

    print(
        f"Normalized preview written to: "
        f"{normalized_path}"
    )


if __name__ == "__main__":
    main()