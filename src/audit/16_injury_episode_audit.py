from __future__ import annotations

# Lines 3-10: Standard-library imports
import argparse
import json
from pathlib import Path

# Lines 12-13: Third-party imports
import pandas as pd


# Line 17: Resolve repository root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_injury_file() -> Path:
    """Locate SoccerMon injury.csv."""
    path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "subjective"
        / "subjective"
        / "injury"
        / "injury.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"injury.csv not found: {path}"
        )

    return path


def load_injuries(
    path: Path,
) -> pd.DataFrame:
    """Load and clean raw injury records."""
    frame = pd.read_csv(path)

    required = {
        "player_name",
        "type",
        "timestamp",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Missing injury columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    frame = frame.copy()

    frame["player_name"] = (
        frame["player_name"]
        .astype(str)
        .str.strip()
    )

    frame["injury_date"] = (
        pd.to_datetime(
            frame["timestamp"],
            format="%d.%m.%Y",
            errors="coerce",
        )
    )

    if frame[
        "injury_date"
    ].isna().any():

        count = int(
            frame[
                "injury_date"
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            f"{count} injury rows "
            "contain invalid dates."
        )

    return frame


def parse_injury_json(
    value: str,
) -> dict[str, str]:
    """
    Parse the JSON stored in injury.csv's type column.

    Example:
    {"left_knee":"major","right_knee":"major"}
    """
    try:
        parsed = json.loads(
            value
        )
    except Exception as exc:
        raise ValueError(
            f"Could not parse injury type: {value}"
        ) from exc

    if not isinstance(
        parsed,
        dict,
    ):
        raise ValueError(
            "Expected injury type "
            f"to be a JSON object: {value}"
        )

    return {
        str(region): str(severity)
        for region, severity
        in parsed.items()
    }


def explode_injury_components(
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert multi-region injury records into one row
    per player/date/body-region/severity.

    Example:

    {"left_knee":"major",
     "right_knee":"major"}

    becomes two rows.
    """
    rows: list[
        dict[str, object]
    ] = []

    for raw_index, row in (
        injuries.iterrows()
    ):
        components = (
            parse_injury_json(
                row["type"]
            )
        )

        for (
            body_region,
            severity,
        ) in components.items():

            rows.append(
                {
                    "raw_row_index": (
                        raw_index
                    ),
                    "player_name": (
                        row[
                            "player_name"
                        ]
                    ),
                    "injury_date": (
                        row[
                            "injury_date"
                        ]
                    ),
                    "body_region": (
                        body_region
                    ),
                    "severity": (
                        severity
                    ),
                    "original_type": (
                        row["type"]
                    ),
                }
            )

    frame = pd.DataFrame(
        rows
    )

    return frame.sort_values(
        [
            "player_name",
            "body_region",
            "severity",
            "injury_date",
        ]
    ).reset_index(
        drop=True
    )


def remove_exact_component_duplicates(
    components: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove duplicate observations of the same
    player/date/region/severity.

    This is only for episode construction.
    Raw data remains untouched.
    """
    return (
        components
        .drop_duplicates(
            subset=[
                "player_name",
                "injury_date",
                "body_region",
                "severity",
            ]
        )
        .copy()
    )


def construct_episodes(
    components: pd.DataFrame,
    max_gap_days: int,
) -> pd.DataFrame:
    """
    Construct candidate episodes.

    Same player + same body region + same severity
    continues the same episode when the gap between
    consecutive observations is <= max_gap_days.
    """
    frame = components.sort_values(
        [
            "player_name",
            "body_region",
            "severity",
            "injury_date",
        ]
    ).copy()

    frame[
        "previous_observation_date"
    ] = (
        frame.groupby(
            [
                "player_name",
                "body_region",
                "severity",
            ]
        )[
            "injury_date"
        ]
        .shift(1)
    )

    frame[
        "gap_days"
    ] = (
        frame[
            "injury_date"
        ]
        - frame[
            "previous_observation_date"
        ]
    ).dt.days

    frame[
        "starts_new_episode"
    ] = (
        frame[
            "previous_observation_date"
        ].isna()
        |
        (
            frame[
                "gap_days"
            ]
            > max_gap_days
        )
    )

    frame[
        "episode_number"
    ] = (
        frame.groupby(
            [
                "player_name",
                "body_region",
                "severity",
            ]
        )[
            "starts_new_episode"
        ]
        .cumsum()
        .astype(int)
    )

    frame[
        "episode_id"
    ] = (
        frame[
            "player_name"
        ]
        + "|"
        + frame[
            "body_region"
        ]
        + "|"
        + frame[
            "severity"
        ]
        + "|"
        + frame[
            "episode_number"
        ].astype(str)
    )

    return frame


def summarize_episodes(
    episode_rows: pd.DataFrame,
    max_gap_days: int,
) -> pd.DataFrame:
    """Create one row per candidate injury episode."""
    summary = (
        episode_rows
        .groupby(
            [
                "episode_id",
                "player_name",
                "body_region",
                "severity",
            ],
            as_index=False,
        )
        .agg(
            episode_start=(
                "injury_date",
                "min",
            ),
            episode_last_observed=(
                "injury_date",
                "max",
            ),
            injury_observations=(
                "injury_date",
                "size",
            ),
            unique_observation_dates=(
                "injury_date",
                "nunique",
            ),
        )
    )

    summary[
        "observed_span_days"
    ] = (
        summary[
            "episode_last_observed"
        ]
        - summary[
            "episode_start"
        ]
    ).dt.days

    summary[
        "episode_gap_rule_days"
    ] = (
        max_gap_days
    )

    return summary.sort_values(
        [
            "player_name",
            "episode_start",
            "body_region",
        ]
    )


def build_sensitivity_summary(
    components: pd.DataFrame,
    gap_values: list[int],
) -> tuple[
    pd.DataFrame,
    dict[int, pd.DataFrame],
    dict[int, pd.DataFrame],
]:
    """
    Construct episodes under several gap definitions.
    """
    rows: list[
        dict[str, object]
    ] = []

    episode_rows_by_gap: dict[
        int,
        pd.DataFrame
    ] = {}

    episodes_by_gap: dict[
        int,
        pd.DataFrame
    ] = {}

    for gap in gap_values:

        episode_rows = (
            construct_episodes(
                components,
                gap,
            )
        )

        episodes = (
            summarize_episodes(
                episode_rows,
                gap,
            )
        )

        episode_rows_by_gap[
            gap
        ] = episode_rows

        episodes_by_gap[
            gap
        ] = episodes

        rows.append(
            {
                "gap_rule_days": gap,
                "candidate_episodes": (
                    len(episodes)
                ),
                "players": (
                    episodes[
                        "player_name"
                    ].nunique()
                ),
                "body_regions": (
                    episodes[
                        "body_region"
                    ].nunique()
                ),
                "minor_episodes": int(
                    (
                        episodes[
                            "severity"
                        ]
                        == "minor"
                    ).sum()
                ),
                "major_episodes": int(
                    (
                        episodes[
                            "severity"
                        ]
                        == "major"
                    ).sum()
                ),
                "single_observation_episodes": int(
                    (
                        episodes[
                            "injury_observations"
                        ]
                        == 1
                    ).sum()
                ),
                "multi_observation_episodes": int(
                    (
                        episodes[
                            "injury_observations"
                        ]
                        > 1
                    ).sum()
                ),
                "median_observed_span_days": (
                    float(
                        episodes[
                            "observed_span_days"
                        ].median()
                    )
                    if not episodes.empty
                    else None
                ),
                "maximum_observed_span_days": (
                    int(
                        episodes[
                            "observed_span_days"
                        ].max()
                    )
                    if not episodes.empty
                    else None
                ),
            }
        )

    return (
        pd.DataFrame(rows),
        episode_rows_by_gap,
        episodes_by_gap,
    )


def build_region_summary(
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize candidate episodes by body region."""
    return (
        episodes
        .groupby(
            [
                "body_region",
                "severity",
            ],
            as_index=False,
        )
        .agg(
            candidate_episodes=(
                "episode_id",
                "size",
            ),
            affected_players=(
                "player_name",
                "nunique",
            ),
            median_span_days=(
                "observed_span_days",
                "median",
            ),
        )
        .sort_values(
            "candidate_episodes",
            ascending=False,
        )
    )


def build_player_history_summary(
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize candidate injury history by player.

    This previews features we may eventually construct
    without actually creating modeling features yet.
    """
    rows: list[
        dict[str, object]
    ] = []

    for (
        player_name,
        group,
    ) in episodes.groupby(
        "player_name"
    ):

        rows.append(
            {
                "player_name": (
                    player_name
                ),
                "candidate_episodes": (
                    len(group)
                ),
                "minor_episodes": int(
                    (
                        group[
                            "severity"
                        ]
                        == "minor"
                    ).sum()
                ),
                "major_episodes": int(
                    (
                        group[
                            "severity"
                        ]
                        == "major"
                    ).sum()
                ),
                "unique_body_regions": (
                    group[
                        "body_region"
                    ].nunique()
                ),
                "first_episode_start": (
                    group[
                        "episode_start"
                    ].min()
                ),
                "last_episode_start": (
                    group[
                        "episode_start"
                    ].max()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "candidate_episodes",
            ascending=False,
        )
    )


def write_summary(
    raw_injuries: pd.DataFrame,
    exploded: pd.DataFrame,
    deduplicated_components: pd.DataFrame,
    sensitivity: pd.DataFrame,
    selected_gap: int,
    selected_episodes: pd.DataFrame,
    region_summary: pd.DataFrame,
    player_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write human-readable episode audit."""
    lines: list[str] = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Injury Episode Audit"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Raw injury rows: "
        f"{len(raw_injuries):,}"
    )

    lines.append(
        f"Exploded body-region observations: "
        f"{len(exploded):,}"
    )

    lines.append(
        "Unique player/date/region/severity observations: "
        f"{len(deduplicated_components):,}"
    )

    lines.append("")
    lines.append(
        "Episode-gap sensitivity"
    )
    lines.append(
        "-" * 80
    )

    for _, row in (
        sensitivity.iterrows()
    ):

        lines.append(
            f"Gap <= "
            f"{int(row['gap_rule_days'])} days | "
            f"episodes: "
            f"{int(row['candidate_episodes'])} | "
            f"single-observation: "
            f"{int(row['single_observation_episodes'])} | "
            f"multi-observation: "
            f"{int(row['multi_observation_episodes'])} | "
            f"median span: "
            f"{row['median_observed_span_days']} days | "
            f"max span: "
            f"{row['maximum_observed_span_days']} days"
        )

    lines.append("")
    lines.append(
        "Selected audit view"
    )
    lines.append(
        "-" * 80
    )

    lines.append(
        f"Selected gap rule: "
        f"<= {selected_gap} days"
    )

    lines.append(
        f"Candidate episodes: "
        f"{len(selected_episodes):,}"
    )

    lines.append(
        f"Players with candidate episodes: "
        f"{selected_episodes['player_name'].nunique():,}"
    )

    lines.append(
        f"Minor candidate episodes: "
        f"{int((selected_episodes['severity'] == 'minor').sum()):,}"
    )

    lines.append(
        f"Major candidate episodes: "
        f"{int((selected_episodes['severity'] == 'major').sum()):,}"
    )

    lines.append("")
    lines.append(
        "Most common region/severity combinations"
    )
    lines.append(
        "-" * 80
    )

    for _, row in (
        region_summary
        .head(15)
        .iterrows()
    ):

        lines.append(
            f"{row['body_region']} | "
            f"{row['severity']} | "
            f"episodes: "
            f"{int(row['candidate_episodes'])} | "
            f"players: "
            f"{int(row['affected_players'])} | "
            f"median observed span: "
            f"{row['median_span_days']}"
        )

    lines.append("")
    lines.append(
        "Player-level candidate episode counts"
    )
    lines.append(
        "-" * 80
    )

    for _, row in (
        player_summary.iterrows()
    ):

        lines.append(
            f"{row['player_name']} | "
            f"episodes: "
            f"{int(row['candidate_episodes'])} | "
            f"minor: "
            f"{int(row['minor_episodes'])} | "
            f"major: "
            f"{int(row['major_episodes'])} | "
            f"regions: "
            f"{int(row['unique_body_regions'])}"
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
            "Construct and compare candidate SoccerMon "
            "injury episodes under multiple gap rules."
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

    parser.add_argument(
        "--selected-gap",
        type=int,
        default=7,
        help=(
            "Gap rule used for detailed audit outputs. "
            "Default: 7 days. This is NOT yet a final "
            "modeling decision."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    injury_path = (
        find_injury_file()
    )

    print(
        f"Loading: {injury_path}"
    )

    raw_injuries = (
        load_injuries(
            injury_path
        )
    )

    exploded = (
        explode_injury_components(
            raw_injuries
        )
    )

    deduplicated_components = (
        remove_exact_component_duplicates(
            exploded
        )
    )

    print(
        f"Raw rows: "
        f"{len(raw_injuries):,}"
    )

    print(
        f"Exploded component observations: "
        f"{len(exploded):,}"
    )

    print(
        "Unique player/date/region/severity "
        f"observations: "
        f"{len(deduplicated_components):,}"
    )

    gap_values = [
        3,
        7,
        14,
        28,
    ]

    (
        sensitivity,
        episode_rows_by_gap,
        episodes_by_gap,
    ) = build_sensitivity_summary(
        deduplicated_components,
        gap_values,
    )

    if (
        args.selected_gap
        not in episodes_by_gap
    ):
        raise ValueError(
            "--selected-gap must be one of: "
            + ", ".join(
                str(value)
                for value
                in gap_values
            )
        )

    selected_episode_rows = (
        episode_rows_by_gap[
            args.selected_gap
        ]
    )

    selected_episodes = (
        episodes_by_gap[
            args.selected_gap
        ]
    )

    region_summary = (
        build_region_summary(
            selected_episodes
        )
    )

    player_summary = (
        build_player_history_summary(
            selected_episodes
        )
    )

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exploded.to_csv(
        output_dir
        / "injury_components_exploded.csv",
        index=False,
    )

    deduplicated_components.to_csv(
        output_dir
        / "injury_components_deduplicated.csv",
        index=False,
    )

    sensitivity.to_csv(
        output_dir
        / "injury_episode_gap_sensitivity.csv",
        index=False,
    )

    selected_episode_rows.to_csv(
        output_dir
        / (
            "injury_episode_assignments_"
            f"{args.selected_gap}d.csv"
        ),
        index=False,
    )

    selected_episodes.to_csv(
        output_dir
        / (
            "injury_candidate_episodes_"
            f"{args.selected_gap}d.csv"
        ),
        index=False,
    )

    region_summary.to_csv(
        output_dir
        / (
            "injury_episode_region_summary_"
            f"{args.selected_gap}d.csv"
        ),
        index=False,
    )

    player_summary.to_csv(
        output_dir
        / (
            "injury_episode_player_summary_"
            f"{args.selected_gap}d.csv"
        ),
        index=False,
    )

    summary_path = (
        output_dir
        / "injury_episode_summary.txt"
    )

    write_summary(
        raw_injuries=(
            raw_injuries
        ),
        exploded=exploded,
        deduplicated_components=(
            deduplicated_components
        ),
        sensitivity=sensitivity,
        selected_gap=(
            args.selected_gap
        ),
        selected_episodes=(
            selected_episodes
        ),
        region_summary=(
            region_summary
        ),
        player_summary=(
            player_summary
        ),
        output_path=(
            summary_path
        ),
    )


if __name__ == "__main__":
    main()