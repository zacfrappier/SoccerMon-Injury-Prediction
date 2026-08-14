from __future__ import annotations

# Lines 3-10: Standard-library imports
import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Lines 12-13: Third-party imports
import pandas as pd


# Line 17: Resolve repository root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PlayerOverlapRecord:
    player_name: str
    team: str
    player_id: str
    in_objective: bool
    in_subjective_wide: bool
    in_session_json: bool
    in_injury: bool
    in_all_core_sources: bool
    objective_only: bool
    subjective_only: bool


def load_objective_players(
    inventory_path: Path,
) -> set[str]:
    """
    Load objective player IDs from Script 2 output.

    Objective inventory stores:
    player_id + team

    Subjective data uses:
    TeamA-<player_id>
    """
    frame = pd.read_csv(
        inventory_path
    )

    required = {
        "player_id",
        "team",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Objective inventory is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    players = {
        f"{row.team}-{row.player_id}"
        for row in frame[
            [
                "team",
                "player_id",
            ]
        ].itertuples(
            index=False
        )
    }

    return players


def find_subjective_root() -> Path:
    """Locate SoccerMon subjective-data root."""
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


def load_subjective_wide_players(
    subjective_root: Path,
) -> set[str]:
    """
    Get player names from one of the 50-player wide tables.

    Script 8 established that the 15 wide tables share
    the same player set.
    """
    path = (
        subjective_root
        / "training-load"
        / "daily_load.csv"
    )

    frame = pd.read_csv(
        path,
        nrows=1,
    )

    players = {
        str(column)
        for column in frame.columns
        if str(column).startswith(
            ("TeamA-", "TeamB-")
        )
    }

    return players


def load_session_players(
    session_path: Path,
) -> set[str]:
    """Load player IDs from session.json top-level keys."""
    with session_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected session.json to contain a dictionary."
        )

    return {
        str(player_name)
        for player_name in data.keys()
    }


def load_injury_players(
    injury_path: Path,
) -> set[str]:
    """Load player IDs appearing in injury.csv."""
    frame = pd.read_csv(
        injury_path
    )

    if "player_name" not in frame.columns:
        raise ValueError(
            "injury.csv does not contain player_name."
        )

    return set(
        frame[
            "player_name"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )


def split_player_name(
    player_name: str,
) -> tuple[str, str]:
    """
    Split:
    TeamA-uuid

    into:
    TeamA
    uuid
    """
    if player_name.startswith(
        "TeamA-"
    ):
        return (
            "TeamA",
            player_name[
                len("TeamA-"):
            ],
        )

    if player_name.startswith(
        "TeamB-"
    ):
        return (
            "TeamB",
            player_name[
                len("TeamB-"):
            ],
        )

    return (
        "",
        player_name,
    )


def build_overlap_records(
    objective_players: set[str],
    subjective_players: set[str],
    session_players: set[str],
    injury_players: set[str],
) -> list[PlayerOverlapRecord]:
    """Create one overlap record per player appearing anywhere."""
    all_players = sorted(
        objective_players
        | subjective_players
        | session_players
        | injury_players
    )

    records: list[
        PlayerOverlapRecord
    ] = []

    for player_name in all_players:

        team, player_id = (
            split_player_name(
                player_name
            )
        )

        in_objective = (
            player_name
            in objective_players
        )

        in_subjective = (
            player_name
            in subjective_players
        )

        in_session = (
            player_name
            in session_players
        )

        in_injury = (
            player_name
            in injury_players
        )

        in_all_core = (
            in_objective
            and in_subjective
            and in_session
        )

        objective_only = (
            in_objective
            and not (
                in_subjective
                or in_session
                or in_injury
            )
        )

        subjective_only = (
            not in_objective
            and (
                in_subjective
                or in_session
                or in_injury
            )
        )

        records.append(
            PlayerOverlapRecord(
                player_name=player_name,
                team=team,
                player_id=player_id,
                in_objective=in_objective,
                in_subjective_wide=(
                    in_subjective
                ),
                in_session_json=(
                    in_session
                ),
                in_injury=in_injury,
                in_all_core_sources=(
                    in_all_core
                ),
                objective_only=(
                    objective_only
                ),
                subjective_only=(
                    subjective_only
                ),
            )
        )

    return records


def write_records(
    records: list[PlayerOverlapRecord],
    output_path: Path,
) -> None:
    """Write detailed overlap table."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        PlayerOverlapRecord
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


def write_summary(
    records: list[PlayerOverlapRecord],
    anomaly_players: set[str],
    output_path: Path,
) -> None:
    """Write a readable player-overlap report."""
    total_players = len(
        records
    )

    objective = [
        record
        for record in records
        if record.in_objective
    ]

    subjective = [
        record
        for record in records
        if record.in_subjective_wide
    ]

    sessions = [
        record
        for record in records
        if record.in_session_json
    ]

    injuries = [
        record
        for record in records
        if record.in_injury
    ]

    core_overlap = [
        record
        for record in records
        if record.in_all_core_sources
    ]

    objective_only = [
        record
        for record in records
        if record.objective_only
    ]

    subjective_only = [
        record
        for record in records
        if record.subjective_only
    ]

    injury_with_objective = [
        record
        for record in injuries
        if record.in_objective
    ]

    injury_with_subjective = [
        record
        for record in injuries
        if record.in_subjective_wide
    ]

    injury_with_core = [
        record
        for record in injuries
        if record.in_all_core_sources
    ]

    lines: list[str] = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Player Overlap Audit"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Unique players across all sources: "
        f"{total_players:,}"
    )

    lines.append("")
    lines.append(
        "Source populations"
    )
    lines.append(
        "-" * 80
    )

    lines.append(
        f"Objective players: "
        f"{len(objective):,}"
    )

    lines.append(
        f"Subjective wide-table players: "
        f"{len(subjective):,}"
    )

    lines.append(
        f"session.json players: "
        f"{len(sessions):,}"
    )

    lines.append(
        f"Injury-record players: "
        f"{len(injuries):,}"
    )

    lines.append("")
    lines.append(
        "Core overlap"
    )
    lines.append(
        "-" * 80
    )

    lines.append(
        "Players present in objective + "
        "subjective wide tables + session.json: "
        f"{len(core_overlap):,}"
    )

    lines.append(
        f"Objective-only players: "
        f"{len(objective_only):,}"
    )

    lines.append(
        f"Subjective-only players: "
        f"{len(subjective_only):,}"
    )

    lines.append("")
    lines.append(
        "Injury-label coverage"
    )
    lines.append(
        "-" * 80
    )

    lines.append(
        f"Injury players with objective data: "
        f"{len(injury_with_objective):,}/"
        f"{len(injuries):,}"
    )

    lines.append(
        f"Injury players with subjective data: "
        f"{len(injury_with_subjective):,}/"
        f"{len(injuries):,}"
    )

    lines.append(
        "Injury players present in all core sources: "
        f"{len(injury_with_core):,}/"
        f"{len(injuries):,}"
    )

    lines.append("")
    lines.append(
        "Daily-load anomaly players"
    )
    lines.append(
        "-" * 80
    )

    for player_name in sorted(
        anomaly_players
    ):
        record = next(
            (
                item
                for item in records
                if item.player_name
                == player_name
            ),
            None,
        )

        if record is None:
            lines.append(
                f"{player_name}: "
                "not found in overlap table"
            )
            continue

        lines.append(
            f"{player_name} | "
            f"objective: "
            f"{record.in_objective} | "
            f"subjective: "
            f"{record.in_subjective_wide} | "
            f"session: "
            f"{record.in_session_json} | "
            f"injury: "
            f"{record.in_injury}"
        )

    lines.append("")
    lines.append(
        "Objective-only players"
    )
    lines.append(
        "-" * 80
    )

    for record in objective_only:
        lines.append(
            record.player_name
        )

    lines.append("")
    lines.append(
        "Subjective-only players"
    )
    lines.append(
        "-" * 80
    )

    for record in subjective_only:
        lines.append(
            record.player_name
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
            "Compare SoccerMon player populations across "
            "objective data, subjective tables, session.json, "
            "and injury records."
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

    objective_inventory_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "audit"
        / "objective_player_inventory.csv"
    )

    session_path = (
        subjective_root
        / "training-load"
        / "session.json"
    )

    injury_path = (
        subjective_root
        / "injury"
        / "injury.csv"
    )

    print(
        "Loading player populations..."
    )

    objective_players = (
        load_objective_players(
            objective_inventory_path
        )
    )

    subjective_players = (
        load_subjective_wide_players(
            subjective_root
        )
    )

    session_players = (
        load_session_players(
            session_path
        )
    )

    injury_players = (
        load_injury_players(
            injury_path
        )
    )

    print(
        f"Objective players: "
        f"{len(objective_players):,}"
    )

    print(
        f"Subjective players: "
        f"{len(subjective_players):,}"
    )

    print(
        f"Session players: "
        f"{len(session_players):,}"
    )

    print(
        f"Injury players: "
        f"{len(injury_players):,}"
    )

    records = (
        build_overlap_records(
            objective_players=(
                objective_players
            ),
            subjective_players=(
                subjective_players
            ),
            session_players=(
                session_players
            ),
            injury_players=(
                injury_players
            ),
        )
    )

    anomaly_players = {
        "TeamA-32fed4b3-d7fc-482d-ba21-c46c58f015b5",
        "TeamA-4051bba7-1170-4c43-b912-8c38815a7625",
        "TeamA-5cd7a61b-88b2-46d2-94f8-5a0d4f682d93",
    }

    output_dir = (
        args.output_dir
    )

    detailed_path = (
        output_dir
        / "player_overlap_audit.csv"
    )

    summary_path = (
        output_dir
        / "player_overlap_summary.txt"
    )

    write_records(
        records,
        detailed_path,
    )

    write_summary(
        records=records,
        anomaly_players=(
            anomaly_players
        ),
        output_path=summary_path,
    )

    print(
        f"Detailed overlap table written to: "
        f"{detailed_path}"
    )


if __name__ == "__main__":
    main()