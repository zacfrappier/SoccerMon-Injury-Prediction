from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audit"
    / "objective_player_inventory.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "player_coverage"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "player_coverage"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Validate input
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Audit 2 player inventory not found: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

required_columns = {
    "player_id",
    "team",
    "session_files",
    "first_date",
    "last_date",
    "active_dates",
    "total_size_bytes",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        "Player inventory is missing required columns: "
        + ", ".join(
            sorted(missing_columns)
        )
    )


# ============================================================
# Clean / prepare
# ============================================================

df["first_date"] = pd.to_datetime(
    df["first_date"],
    format="%Y-%m-%d",
    errors="coerce",
)

df["last_date"] = pd.to_datetime(
    df["last_date"],
    format="%Y-%m-%d",
    errors="coerce",
)

df["session_files"] = pd.to_numeric(
    df["session_files"],
    errors="coerce",
)

df["active_dates"] = pd.to_numeric(
    df["active_dates"],
    errors="coerce",
)

df["total_size_gb"] = (
    df["total_size_bytes"]
    / (1024 ** 3)
)

df["observation_span_days"] = (
    df["last_date"]
    - df["first_date"]
).dt.days + 1


# ============================================================
# Short player labels
# ============================================================

def short_player_name(
    team: str,
    player_id: str,
) -> str:
    """
    Shorten UUIDs for readable plot labels.

    Full IDs remain unchanged in output CSV files.
    """
    return (
        f"{team}-"
        f"{str(player_id)[:8]}"
    )


df["plot_name"] = [
    short_player_name(
        team,
        player_id,
    )
    for team, player_id
    in zip(
        df["team"],
        df["player_id"],
    )
]


# ============================================================
# Basic checks
# ============================================================

print(
    f"Players loaded: "
    f"{len(df):,}"
)

print(
    f"Team A players: "
    f"{(df['team'] == 'TeamA').sum():,}"
)

print(
    f"Team B players: "
    f"{(df['team'] == 'TeamB').sum():,}"
)

print(
    f"Session files represented: "
    f"{df['session_files'].sum():,.0f}"
)

print(
    f"Median session files per player: "
    f"{df['session_files'].median():.1f}"
)


# ============================================================
# Save general player coverage table
# ============================================================

coverage_columns = [
    "player_id",
    "team",
    "session_files",
    "active_dates",
    "first_date",
    "last_date",
    "observation_span_days",
    "total_size_gb",
]

df[
    coverage_columns
].to_csv(
    TABLE_DIR
    / "player_coverage_summary.csv",
    index=False,
)


# ============================================================
# Figure 1:
# Session files per player
# ============================================================

session_order = (
    df
    .sort_values(
        "session_files",
        ascending=True,
    )
    .copy()
)

fig, ax = plt.subplots(
    figsize=(
        12,
        max(
            12,
            len(session_order) * 0.28,
        ),
    )
)

ax.barh(
    session_order[
        "plot_name"
    ],
    session_order[
        "session_files"
    ],
)

ax.set_title(
    "Objective Session Files per Player"
)

ax.set_xlabel(
    "Number of Parquet Session Files"
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
    / "session_files_per_player.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 2:
# Active dates per player
# ============================================================

active_order = (
    df
    .sort_values(
        "active_dates",
        ascending=True,
    )
    .copy()
)

fig, ax = plt.subplots(
    figsize=(
        12,
        max(
            12,
            len(active_order) * 0.28,
        ),
    )
)

ax.barh(
    active_order[
        "plot_name"
    ],
    active_order[
        "active_dates"
    ],
)

ax.set_title(
    "Objective Recording Days per Player"
)

ax.set_xlabel(
    "Number of Unique Active Dates"
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
    / "active_dates_per_player.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 3:
# Observation span timeline
# ============================================================

timeline = (
    df
    .dropna(
        subset=[
            "first_date",
            "last_date",
        ]
    )
    .sort_values(
        [
            "team",
            "first_date",
        ]
    )
    .reset_index(
        drop=True
    )
)

fig, ax = plt.subplots(
    figsize=(
        14,
        max(
            12,
            len(timeline) * 0.28,
        ),
    )
)

for index, row in (
    timeline.iterrows()
):

    ax.hlines(
        y=index,
        xmin=row[
            "first_date"
        ],
        xmax=row[
            "last_date"
        ],
        linewidth=3,
    )

    ax.plot(
        row[
            "first_date"
        ],
        index,
        marker="o",
        markersize=3,
    )

    ax.plot(
        row[
            "last_date"
        ],
        index,
        marker="o",
        markersize=3,
    )


ax.set_yticks(
    range(
        len(timeline)
    )
)

ax.set_yticklabels(
    timeline[
        "plot_name"
    ]
)

ax.set_title(
    "Objective Observation Span by Player"
)

ax.set_xlabel(
    "Calendar Date"
)

ax.set_ylabel(
    "Player"
)

ax.grid(
    axis="x",
    alpha=0.3,
)

fig.autofmt_xdate()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "observation_span_per_player.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 4:
# Distribution of player coverage
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    df[
        "active_dates"
    ].dropna(),
    bins=15,
)

ax.axvline(
    df[
        "active_dates"
    ].median(),
    linestyle="--",
    label=(
        "Median = "
        f"{df['active_dates'].median():.1f}"
    ),
)

ax.set_title(
    "Distribution of Objective Recording Days per Player"
)

ax.set_xlabel(
    "Unique Active Dates"
)

ax.set_ylabel(
    "Number of Players"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "player_coverage_distribution.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Team-level summary table
# ============================================================

team_summary = (
    df
    .groupby(
        "team"
    )
    .agg(
        players=(
            "player_id",
            "nunique",
        ),
        total_session_files=(
            "session_files",
            "sum",
        ),
        mean_session_files=(
            "session_files",
            "mean",
        ),
        median_session_files=(
            "session_files",
            "median",
        ),
        mean_active_dates=(
            "active_dates",
            "mean",
        ),
        median_active_dates=(
            "active_dates",
            "median",
        ),
        total_storage_gb=(
            "total_size_gb",
            "sum",
        ),
    )
)

team_summary.to_csv(
    TABLE_DIR
    / "player_coverage_by_team.csv"
)


# ============================================================
# Sparse-player table
# ============================================================

sparse_players = (
    df[
        [
            "player_id",
            "team",
            "session_files",
            "active_dates",
            "first_date",
            "last_date",
            "observation_span_days",
        ]
    ]
    .sort_values(
        "active_dates",
        ascending=True,
    )
)

sparse_players.to_csv(
    TABLE_DIR
    / "players_by_objective_coverage.csv",
    index=False,
)


# ============================================================
# Terminal summary
# ============================================================

print()
print(
    "Visual 02 complete."
)

print(
    f"\nFigures written to:\n"
    f"{FIGURE_DIR}"
)

print(
    f"\nTables written to:\n"
    f"{TABLE_DIR}"
)

print(
    "\nLowest objective coverage:"
)

print(
    sparse_players
    .head(10)
    .to_string(
        index=False
    )
)