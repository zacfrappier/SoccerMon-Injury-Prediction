from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audit"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "injury"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "injury"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PLAYER_FILE = (
    AUDIT_DIR
    / "injury_player_summary.csv"
)

EPISODE_PLAYER_FILE = (
    AUDIT_DIR
    / "injury_episode_player_summary_7d.csv"
)

EPISODE_REGION_FILE = (
    AUDIT_DIR
    / "injury_episode_region_summary_7d.csv"
)

GAP_FILE = (
    AUDIT_DIR
    / "injury_episode_gap_sensitivity.csv"
)

ALIGNMENT_FILE = (
    AUDIT_DIR
    / "injury_objective_alignment.csv"
)


# ============================================================
# Helpers
# ============================================================

def short_player_name(
    player_name: str,
) -> str:

    player_name = str(
        player_name
    )

    if "-" not in player_name:

        return player_name[:12]

    team, player_id = (
        player_name.split(
            "-",
            1,
        )
    )

    return (
        f"{team}-"
        f"{player_id[:8]}"
    )


def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required audit file not found: {path}"
        )


# ============================================================
# Load files
# ============================================================

for path in [
    PLAYER_FILE,
    EPISODE_PLAYER_FILE,
    EPISODE_REGION_FILE,
    GAP_FILE,
    ALIGNMENT_FILE,
]:

    require_file(
        path
    )


players = pd.read_csv(
    PLAYER_FILE
)

episode_players = pd.read_csv(
    EPISODE_PLAYER_FILE
)

regions = pd.read_csv(
    EPISODE_REGION_FILE
)

gap_sensitivity = pd.read_csv(
    GAP_FILE
)

alignment = pd.read_csv(
    ALIGNMENT_FILE
)


# ============================================================
# FIGURE 1
# Injury records by player
# ============================================================

player_plot = (
    players
    .sort_values(
        "injury_records",
        ascending=True,
    )
    .copy()
)

player_plot[
    "plot_name"
] = (
    player_plot[
        "player_name"
    ]
    .apply(
        short_player_name
    )
)


fig, ax = plt.subplots(
    figsize=(
        10,
        max(
            6,
            len(player_plot)
            * 0.45,
        ),
    )
)

ax.barh(
    player_plot[
        "plot_name"
    ],
    player_plot[
        "injury_records"
    ],
)

ax.set_title(
    "Injury Records by Player"
)

ax.set_xlabel(
    "Number of Injury Records"
)

ax.set_ylabel(
    "Player"
)

ax.grid(
    axis="x",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "injury_records_by_player.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


player_plot.to_csv(
    TABLE_DIR
    / "injury_records_by_player.csv",
    index=False,
)


# ============================================================
# FIGURE 2
# Candidate injury episodes by player
# ============================================================

episode_plot = (
    episode_players
    .sort_values(
        "candidate_episodes",
        ascending=True,
    )
    .copy()
)

episode_plot[
    "plot_name"
] = (
    episode_plot[
        "player_name"
    ]
    .apply(
        short_player_name
    )
)


fig, ax = plt.subplots(
    figsize=(
        10,
        max(
            6,
            len(
                episode_plot
            )
            * 0.45,
        ),
    )
)

ax.barh(
    episode_plot[
        "plot_name"
    ],
    episode_plot[
        "candidate_episodes"
    ],
)

ax.set_title(
    "Candidate Injury Episodes by Player (7-Day Gap Rule)"
)

ax.set_xlabel(
    "Number of Candidate Injury Episodes"
)

ax.set_ylabel(
    "Player"
)

ax.grid(
    axis="x",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "injury_episode_counts_by_player.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


episode_plot.to_csv(
    TABLE_DIR
    / "injury_episode_counts_by_player.csv",
    index=False,
)


# ============================================================
# FIGURE 3
# Injury episode regions
# ============================================================

region_totals = (
    regions
    .groupby(
        "body_region",
        as_index=False,
    )[
        "candidate_episodes"
    ]
    .sum()
    .sort_values(
        "candidate_episodes",
        ascending=True,
    )
)


fig, ax = plt.subplots(
    figsize=(
        10,
        max(
            6,
            len(
                region_totals
            )
            * 0.4,
        ),
    )
)

ax.barh(
    region_totals[
        "body_region"
    ],
    region_totals[
        "candidate_episodes"
    ],
)

ax.set_title(
    "Candidate Injury Episodes by Body Region"
)

ax.set_xlabel(
    "Number of Candidate Episodes"
)

ax.set_ylabel(
    "Body Region"
)

ax.grid(
    axis="x",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "injury_episode_regions.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


region_totals.to_csv(
    TABLE_DIR
    / "injury_episode_regions.csv",
    index=False,
)


# ============================================================
# FIGURE 4
# Gap-rule sensitivity
# ============================================================

gap_plot = (
    gap_sensitivity
    .sort_values(
        "gap_rule_days"
    )
    .copy()
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.plot(
    gap_plot[
        "gap_rule_days"
    ],
    gap_plot[
        "candidate_episodes"
    ],
    marker="o",
    label="Candidate episodes",
)

ax.plot(
    gap_plot[
        "gap_rule_days"
    ],
    gap_plot[
        "single_observation_episodes"
    ],
    marker="o",
    label="Single-observation episodes",
)

ax.plot(
    gap_plot[
        "gap_rule_days"
    ],
    gap_plot[
        "multi_observation_episodes"
    ],
    marker="o",
    label="Multi-observation episodes",
)

ax.set_title(
    "Sensitivity of Injury Episode Counts to Gap Rule"
)

ax.set_xlabel(
    "Gap Rule (Days)"
)

ax.set_ylabel(
    "Number of Episodes"
)

ax.legend()

ax.grid(
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "injury_episode_gap_sensitivity.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


gap_plot.to_csv(
    TABLE_DIR
    / "injury_episode_gap_sensitivity.csv",
    index=False,
)


# ============================================================
# FIGURE 5
# Injury-objective alignment
# ============================================================

for column in [
    "in_core_population",
    "injury_within_objective_range",
    "objective_on_injury_date",
]:

    alignment[column] = (
        alignment[column]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


alignment_summary = pd.DataFrame(
    {
        "category": [
            "Injury records",
            "Within objective range",
            "Objective on injury date",
            "Objective in prior 7 days",
            "Objective in prior 28 days",
        ],
        "count": [
            len(alignment),
            int(
                alignment[
                    "injury_within_objective_range"
                ].sum()
            ),
            int(
                alignment[
                    "objective_on_injury_date"
                ].sum()
            ),
            int(
                (
                    alignment[
                        "objective_days_prior_7d"
                    ]
                    > 0
                ).sum()
            ),
            int(
                (
                    alignment[
                        "objective_days_prior_28d"
                    ]
                    > 0
                ).sum()
            ),
        ],
    }
)


fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.bar(
    alignment_summary[
        "category"
    ],
    alignment_summary[
        "count"
    ],
)

ax.set_title(
    "Objective-Data Availability Around Injury Records"
)

ax.set_ylabel(
    "Number of Injury Records"
)

ax.set_xlabel(
    "Coverage Condition"
)

ax.tick_params(
    axis="x",
    rotation=20,
)

ax.grid(
    axis="y",
    alpha=0.3,
)

for index, value in enumerate(
    alignment_summary[
        "count"
    ]
):

    ax.text(
        index,
        value,
        str(
            int(value)
        ),
        ha="center",
        va="bottom",
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "injury_objective_coverage.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


alignment_summary.to_csv(
    TABLE_DIR
    / "injury_objective_coverage.csv",
    index=False,
)


# ============================================================
# Terminal summary
# ============================================================

print()
print(
    "Visual 08 complete."
)

print()

print(
    "Injured players:",
    players[
        "player_name"
    ].nunique(),
)

print(
    "Injury records:",
    int(
        players[
            "injury_records"
        ].sum()
    ),
)

print(
    "Candidate episodes under 7-day rule:",
    int(
        episode_players[
            "candidate_episodes"
        ].sum()
    ),
)

print(
    "Body regions represented:",
    regions[
        "body_region"
    ].nunique(),
)

print(
    "Injury records within objective range:",
    int(
        alignment[
            "injury_within_objective_range"
        ].sum()
    ),
)

print(
    "Injury records with objective data on same day:",
    int(
        alignment[
            "objective_on_injury_date"
        ].sum()
    ),
)

print(
    "Injury records with objective data in prior 7 days:",
    int(
        (
            alignment[
                "objective_days_prior_7d"
            ]
            > 0
        ).sum()
    ),
)

print(
    "Injury records with objective data in prior 28 days:",
    int(
        (
            alignment[
                "objective_days_prior_28d"
            ]
            > 0
        ).sum()
    ),
)

print(
    f"\nFigures written to:\n"
    f"{FIGURE_DIR}"
)

print(
    f"\nTables written to:\n"
    f"{TABLE_DIR}"
)