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
    / "objective_file_manifest.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "dataset_structure"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "dataset_structure"
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
        f"Audit 1 manifest not found: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

required_columns = {
    "team",
    "year",
    "month",
    "date",
    "player_id",
    "size_bytes",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        "Manifest is missing required columns: "
        + ", ".join(
            sorted(missing_columns)
        )
    )


# ============================================================
# Clean / prepare
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    format="%Y-%m-%d",
    errors="coerce",
)

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce",
)

df["size_gb"] = (
    df["size_bytes"]
    / (1024 ** 3)
)

df = df.dropna(
    subset=[
        "date",
        "year",
        "team",
        "month",
        "player_id",
    ]
)

df["year"] = (
    df["year"]
    .astype(int)
)


# ============================================================
# Basic checks
# ============================================================

print(
    f"Manifest rows: "
    f"{len(df):,}"
)

print(
    f"Teams: "
    f"{df['team'].nunique():,}"
)

print(
    f"Years: "
    f"{df['year'].nunique():,}"
)

print(
    f"Months: "
    f"{df['month'].nunique():,}"
)

print(
    f"Players: "
    f"{df['player_id'].nunique():,}"
)

print(
    f"Total size: "
    f"{df['size_gb'].sum():.2f} GB"
)


# ============================================================
# Table 1:
# File counts by team and year
# ============================================================

files_team_year = (
    df
    .groupby(
        [
            "team",
            "year",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

files_team_year.to_csv(
    TABLE_DIR
    / "files_by_team_year.csv"
)


# ============================================================
# Figure 1:
# File counts by team and year
# ============================================================

ax = files_team_year.plot(
    kind="bar",
    figsize=(9, 6),
)

ax.set_title(
    "Objective Parquet Files by Team and Year"
)

ax.set_xlabel(
    "Team"
)

ax.set_ylabel(
    "Number of Parquet Files"
)

ax.tick_params(
    axis="x",
    rotation=0,
)

ax.legend(
    title="Year"
)

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "files_by_team_year.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Table 2:
# Monthly file counts
# ============================================================

files_by_month = (
    df
    .groupby(
        "month"
    )
    .size()
    .rename(
        "file_count"
    )
    .sort_index()
)

files_by_month.to_csv(
    TABLE_DIR
    / "files_by_month.csv"
)


# ============================================================
# Figure 2:
# Monthly file timeline
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)

ax.plot(
    files_by_month.index,
    files_by_month.values,
    marker="o",
)

ax.set_title(
    "Objective Data Availability by Month"
)

ax.set_xlabel(
    "Month"
)

ax.set_ylabel(
    "Number of Parquet Files"
)

ax.tick_params(
    axis="x",
    rotation=45,
)

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "files_by_month.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Table 3:
# Storage size by team and year
# ============================================================

storage_team_year = (
    df
    .groupby(
        [
            "team",
            "year",
        ]
    )[
        "size_gb"
    ]
    .sum()
    .unstack(
        fill_value=0
    )
)

storage_team_year.to_csv(
    TABLE_DIR
    / "storage_by_team_year.csv"
)


# ============================================================
# Figure 3:
# Storage by team and year
# ============================================================

ax = storage_team_year.plot(
    kind="bar",
    figsize=(9, 6),
)

ax.set_title(
    "Objective Dataset Storage by Team and Year"
)

ax.set_xlabel(
    "Team"
)

ax.set_ylabel(
    "Storage Size (GB)"
)

ax.tick_params(
    axis="x",
    rotation=0,
)

ax.legend(
    title="Year"
)

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "storage_by_team_year.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Table 4:
# Active players by month
# ============================================================

active_players_month = (
    df
    .groupby(
        [
            "month",
            "team",
        ]
    )[
        "player_id"
    ]
    .nunique()
    .unstack(
        fill_value=0
    )
    .sort_index()
)

active_players_month.to_csv(
    TABLE_DIR
    / "active_players_by_month.csv"
)


# ============================================================
# Figure 4:
# Active players by month
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)

for team in (
    active_players_month.columns
):

    ax.plot(
        active_players_month.index,
        active_players_month[
            team
        ],
        marker="o",
        label=team,
    )

ax.set_title(
    "Players With Objective Data by Month"
)

ax.set_xlabel(
    "Month"
)

ax.set_ylabel(
    "Unique Players"
)

ax.tick_params(
    axis="x",
    rotation=45,
)

ax.legend(
    title="Team"
)

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "active_players_by_month.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Terminal summary
# ============================================================

print()
print(
    "Audit 1 visualization complete."
)

print(
    f"\nFigures written to:\n"
    f"{FIGURE_DIR}"
)

print(
    f"\nTables written to:\n"
    f"{TABLE_DIR}"
)