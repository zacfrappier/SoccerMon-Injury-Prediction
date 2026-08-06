from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


# Line 10: Resolve the repository root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PlayerRecord:
    player_id: str
    team: str
    session_files: int
    first_date: str
    last_date: str
    years: str
    months: int
    active_dates: int
    total_size_bytes: int


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Load the objective file manifest created by Script 1."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest does not exist: {manifest_path}\n"
            "Run src/audit/01_dataset_tree.py first."
        )

    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_player_inventory(
    rows: list[dict[str, str]],
) -> tuple[list[PlayerRecord], dict[str, set[str]]]:
    """Aggregate manifest rows into one record per player and team."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    player_teams: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        player_id = row["player_id"]
        team = row["team"]

        grouped[(player_id, team)].append(row)
        player_teams[player_id].add(team)

    inventory: list[PlayerRecord] = []

    for (player_id, team), player_rows in grouped.items():
        dates = sorted({row["date"] for row in player_rows})
        years = sorted({row["year"] for row in player_rows})
        months = {row["month"] for row in player_rows}

        total_size_bytes = sum(
            int(row["size_bytes"]) for row in player_rows
        )

        inventory.append(
            PlayerRecord(
                player_id=player_id,
                team=team,
                session_files=len(player_rows),
                first_date=dates[0],
                last_date=dates[-1],
                years=";".join(years),
                months=len(months),
                active_dates=len(dates),
                total_size_bytes=total_size_bytes,
            )
        )

    inventory.sort(
        key=lambda record: (
            record.team,
            -record.session_files,
            record.player_id,
        )
    )

    return inventory, player_teams


def write_inventory(
    inventory: list[PlayerRecord],
    output_path: Path,
) -> None:
    """Write one row per player-team combination."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "player_id",
        "team",
        "session_files",
        "first_date",
        "last_date",
        "years",
        "months",
        "active_dates",
        "total_size_bytes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in inventory:
            writer.writerow(asdict(record))


def write_summary(
    inventory: list[PlayerRecord],
    player_teams: dict[str, set[str]],
    output_path: Path,
) -> None:
    """Write a readable player-level summary."""
    unique_players = sorted(player_teams)

    team_players: dict[str, set[str]] = defaultdict(set)

    for record in inventory:
        team_players[record.team].add(record.player_id)

    multi_team_players = {
        player_id: teams
        for player_id, teams in player_teams.items()
        if len(teams) > 1
    }

    session_counts = [record.session_files for record in inventory]

    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("SoccerMon Objective Player Inventory")
    lines.append("=" * 70)
    lines.append(f"Unique player IDs: {len(unique_players):,}")
    lines.append(f"Player-team records: {len(inventory):,}")
    lines.append(
        f"Players appearing in multiple teams: "
        f"{len(multi_team_players):,}"
    )

    lines.append("")
    lines.append("Players by team")
    lines.append("-" * 70)

    for team in sorted(team_players):
        lines.append(
            f"{team}: {len(team_players[team]):,} unique players"
        )

    if session_counts:
        lines.append("")
        lines.append("Session-file counts per player-team record")
        lines.append("-" * 70)
        lines.append(f"Minimum: {min(session_counts):,}")
        lines.append(f"Maximum: {max(session_counts):,}")
        lines.append(
            f"Mean: {sum(session_counts) / len(session_counts):.2f}"
        )

    if multi_team_players:
        lines.append("")
        lines.append("Players appearing in multiple teams")
        lines.append("-" * 70)

        for player_id, teams in sorted(multi_team_players.items()):
            lines.append(
                f"{player_id}: {', '.join(sorted(teams))}"
            )

    lines.append("")
    lines.append("Player details")
    lines.append("-" * 70)

    for record in inventory:
        lines.append(
            f"{record.team} | {record.player_id} | "
            f"{record.session_files:,} files | "
            f"{record.first_date} to {record.last_date} | "
            f"years: {record.years}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines[:18]))
    print()
    print(f"Full summary written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a player inventory from the SoccerMon objective "
            "file manifest."
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
        default=PROJECT_ROOT / "data" / "processed" / "audit",
        help="Directory for player inventory outputs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading manifest: {args.manifest.resolve()}")

    rows = load_manifest(args.manifest)

    inventory, player_teams = build_player_inventory(rows)

    inventory_path = args.output_dir / "objective_player_inventory.csv"
    summary_path = args.output_dir / "objective_player_summary.txt"

    write_inventory(inventory, inventory_path)
    write_summary(
        inventory=inventory,
        player_teams=player_teams,
        output_path=summary_path,
    )

    print(f"Inventory written to: {inventory_path}")


if __name__ == "__main__":
    main()