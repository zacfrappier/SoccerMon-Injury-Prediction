from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

COHORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "paper_inspired"
    / "cohort"
    / "player_cohort.csv"
)

EPISODE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audit"
    / "injury_candidate_episodes_7d.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "paper_inspired"
    / "injury"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DAILY_FILE = (
    OUTPUT_DIR
    / "injury_daily.csv"
)

PLAYER_SUMMARY_FILE = (
    OUTPUT_DIR
    / "injury_player_summary.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "injury_daily_summary.txt"
)


# ============================================================
# Helpers
# ============================================================

def require_file(path: Path) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def parse_bool(series: pd.Series) -> pd.Series:

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


# ============================================================
# Load selected cohort
# ============================================================

def load_selected_cohort() -> pd.DataFrame:

    require_file(
        COHORT_FILE
    )

    cohort = pd.read_csv(
        COHORT_FILE
    )

    required = {
        "player_name",
        "team",
        "include_in_deephit",
    }

    missing = required - set(
        cohort.columns
    )

    if missing:

        raise ValueError(
            "Cohort file missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    include = parse_bool(
        cohort[
            "include_in_deephit"
        ]
    )

    selected = (
        cohort.loc[
            include,
            [
                "player_name",
                "team",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            "player_name"
        )
        .reset_index(
            drop=True
        )
    )

    return selected


# ============================================================
# Load injury episodes
# ============================================================

def load_injury_episodes(
    cohort: pd.DataFrame,
) -> pd.DataFrame:

    require_file(
        EPISODE_FILE
    )

    episodes = pd.read_csv(
        EPISODE_FILE
    )

    required = {
        "episode_id",
        "player_name",
        "body_region",
        "severity",
        "episode_start",
        "episode_last_observed",
    }

    missing = required - set(
        episodes.columns
    )

    if missing:

        raise ValueError(
            "Episode file missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    selected_players = set(
        cohort[
            "player_name"
        ]
    )

    episodes = episodes[
        episodes[
            "player_name"
        ].isin(
            selected_players
        )
    ].copy()

    episodes[
        "episode_start"
    ] = pd.to_datetime(
        episodes[
            "episode_start"
        ],
        errors="coerce",
    ).dt.normalize()

    episodes[
        "episode_last_observed"
    ] = pd.to_datetime(
        episodes[
            "episode_last_observed"
        ],
        errors="coerce",
    ).dt.normalize()

    episodes = episodes.dropna(
        subset=[
            "episode_start"
        ]
    )

    episodes = episodes.merge(
        cohort,
        on="player_name",
        how="left",
        validate="many_to_one",
    )

    return (
        episodes
        .sort_values(
            [
                "player_name",
                "episode_start",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Build injury-event daily table
# ============================================================

def build_injury_daily(
    episodes: pd.DataFrame,
) -> pd.DataFrame:

    if episodes.empty:

        return pd.DataFrame(
            columns=[
                "player_name",
                "team",
                "date",
                "injury_event",
                "injury_episode_count",
                "minor_episode_count",
                "major_episode_count",
                "unique_body_regions",
                "body_regions",
            ]
        )

    episodes = episodes.copy()

    episodes[
        "date"
    ] = episodes[
        "episode_start"
    ]

    episodes[
        "is_minor"
    ] = (
        episodes[
            "severity"
        ]
        .astype(str)
        .str.lower()
        == "minor"
    )

    episodes[
        "is_major"
    ] = (
        episodes[
            "severity"
        ]
        .astype(str)
        .str.lower()
        == "major"
    )

    grouped = (
        episodes
        .groupby(
            [
                "player_name",
                "team",
                "date",
            ],
            as_index=False,
        )
        .agg(
            injury_episode_count=(
                "episode_id",
                "nunique",
            ),

            minor_episode_count=(
                "is_minor",
                "sum",
            ),

            major_episode_count=(
                "is_major",
                "sum",
            ),

            unique_body_regions=(
                "body_region",
                "nunique",
            ),

            body_regions=(
                "body_region",
                lambda x: ";".join(
                    sorted(
                        set(
                            x.astype(str)
                        )
                    )
                ),
            ),
        )
    )

    grouped[
        "injury_event"
    ] = 1

    grouped = grouped[
        [
            "player_name",
            "team",
            "date",
            "injury_event",
            "injury_episode_count",
            "minor_episode_count",
            "major_episode_count",
            "unique_body_regions",
            "body_regions",
        ]
    ]

    return (
        grouped
        .sort_values(
            [
                "player_name",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Player summary
# ============================================================

def build_player_summary(
    episodes: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, player in cohort.iterrows():

        player_name = player[
            "player_name"
        ]

        team = player[
            "team"
        ]

        subset = episodes[
            episodes[
                "player_name"
            ]
            == player_name
        ]

        if subset.empty:

            rows.append(
                {
                    "player_name":
                    player_name,

                    "team":
                    team,

                    "candidate_injury_episodes":
                    0,

                    "minor_episodes":
                    0,

                    "major_episodes":
                    0,

                    "unique_body_regions":
                    0,

                    "first_episode_date":
                    pd.NaT,

                    "last_episode_date":
                    pd.NaT,
                }
            )

            continue

        rows.append(
            {
                "player_name":
                player_name,

                "team":
                team,

                "candidate_injury_episodes":
                len(
                    subset
                ),

                "minor_episodes":
                int(
                    (
                        subset[
                            "severity"
                        ]
                        .astype(str)
                        .str.lower()
                        == "minor"
                    ).sum()
                ),

                "major_episodes":
                int(
                    (
                        subset[
                            "severity"
                        ]
                        .astype(str)
                        .str.lower()
                        == "major"
                    ).sum()
                ),

                "unique_body_regions":
                subset[
                    "body_region"
                ].nunique(),

                "first_episode_date":
                subset[
                    "episode_start"
                ].min(),

                "last_episode_date":
                subset[
                    "episode_start"
                ].max(),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "player_name"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 80
    )

    print(
        "SoccerMon Paper-Inspired Injury Daily Builder"
    )

    print(
        "=" * 80
    )

    print()

    cohort = (
        load_selected_cohort()
    )

    episodes = (
        load_injury_episodes(
            cohort
        )
    )

    injury_daily = (
        build_injury_daily(
            episodes
        )
    )

    player_summary = (
        build_player_summary(
            episodes,
            cohort,
        )
    )

    injury_daily.to_csv(
        DAILY_FILE,
        index=False,
    )

    player_summary.to_csv(
        PLAYER_SUMMARY_FILE,
        index=False,
    )

    injured_players = (
        episodes[
            "player_name"
        ]
        .nunique()
    )

    total_episodes = len(
        episodes
    )

    event_days = len(
        injury_daily
    )

    minor_episodes = int(
        (
            episodes[
                "severity"
            ]
            .astype(str)
            .str.lower()
            == "minor"
        ).sum()
    )

    major_episodes = int(
        (
            episodes[
                "severity"
            ]
            .astype(str)
            .str.lower()
            == "major"
        ).sum()
    )

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Paper-Inspired Injury Daily Summary"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "Cohort"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Selected cohort players: "
        f"{len(cohort):,}"
    )

    lines.append(
        f"Players with candidate injury episodes: "
        f"{injured_players:,}"
    )

    lines.append("")

    lines.append(
        "Injury episodes"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Candidate episodes: "
        f"{total_episodes:,}"
    )

    lines.append(
        f"Minor episodes: "
        f"{minor_episodes:,}"
    )

    lines.append(
        f"Major episodes: "
        f"{major_episodes:,}"
    )

    lines.append(
        f"Unique injury-event days: "
        f"{event_days:,}"
    )

    lines.append("")

    lines.append(
        "Interpretation"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "Injury events are defined using the start date "
        "of candidate episodes created with the previously "
        "audited 7-day episode-gap rule."
    )

    lines.append(
        "Repeated observations belonging to the same candidate "
        "episode are therefore not treated as separate new "
        "injuries."
    )

    lines.append(
        "Players without injury episodes remain in the cohort "
        "and will later contribute censored observations to the "
        "survival model."
    )

    SUMMARY_FILE.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(
            lines
        )
    )

    print()

    print(
        f"Injury daily table written to:\n"
        f"{DAILY_FILE}"
    )

    print()

    print(
        f"Player injury summary written to:\n"
        f"{PLAYER_SUMMARY_FILE}"
    )

    print()

    print(
        f"Summary written to:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":

    main()