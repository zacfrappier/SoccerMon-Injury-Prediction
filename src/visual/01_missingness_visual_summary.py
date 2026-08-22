import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "missingness"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "missingness"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# --------------------------------------------------
# Load Script 17 missing-date output
# --------------------------------------------------

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audit"
    / "calendar_missing_dates.csv"
)

print(
    f"Loading missing dates from:\n"
    f"{INPUT_FILE}"
)

df = pd.read_csv(
    INPUT_FILE
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

df = df.dropna(
    subset=["date"]
)


# --------------------------------------------------
# Keep wellness features only
# --------------------------------------------------

WELLNESS_FEATURES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_duration",
    "sleep_quality",
    "soreness",
    "stress",
]

wellness_missing = df[
    df["feature"].isin(
        WELLNESS_FEATURES
    )
].copy()


# --------------------------------------------------
# Create month identifier
# --------------------------------------------------

wellness_missing["month"] = (
    wellness_missing[
        "date"
    ]
    .dt.to_period("M")
    .astype(str)
)


# --------------------------------------------------
# Count unique missing DAYS
#
# Important:
# A day with all 7 wellness features missing
# should count as ONE missing day, not seven.
# --------------------------------------------------

missing_days = (
    wellness_missing[
        [
            "player_name",
            "date",
            "month",
        ]
    ]
    .drop_duplicates()
)


monthly = (
    missing_days
    .groupby(
        [
            "player_name",
            "month",
        ]
    )[
        "date"
    ]
    .nunique()
    .unstack(
        fill_value=0
    )
)


# Make sure all months appear in chronological order
monthly = monthly.reindex(
    sorted(
        monthly.columns
    ),
    axis=1,
)

print(
    f"\nPlayers found: "
    f"{len(monthly):,}"
)

print(
    f"Months found: "
    f"{len(monthly.columns):,}"
)
# --------------------------------------------------
# Convert missing-day counts to monthly percentages
# --------------------------------------------------

monthly_fraction = monthly.copy().astype(float)

for month_column in monthly_fraction.columns:

    year, month = map(
        int,
        month_column.split("-"),
    )

    days_in_month = calendar.monthrange(
        year,
        month,
    )[1]

    monthly_fraction[
        month_column
    ] = (
        monthly_fraction[
            month_column
        ]
        / days_in_month
    )


# Save both versions
monthly.to_csv(
    TABLE_DIR
    / "missing_days_by_player_month.csv"
)

monthly_fraction.to_csv(
    TABLE_DIR
    / "missing_fraction_by_player_month.csv"
)


# --------------------------------------------------
# Shorter labels for plotting
# --------------------------------------------------

def short_player_name(
    player_name: str,
) -> str:

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


monthly_fraction.index = [
    short_player_name(
        player
    )
    for player
    in monthly_fraction.index
]


# --------------------------------------------------
# Split teams
# --------------------------------------------------

team_a = monthly_fraction[
    monthly_fraction.index.str.startswith(
        "TeamA-"
    )
]

team_b = monthly_fraction[
    monthly_fraction.index.str.startswith(
        "TeamB-"
    )
]


def create_heatmap(
    frame: pd.DataFrame,
    team_name: str,
    output_name: str,
) -> None:

    fig, ax = plt.subplots(
        figsize=(
            16,
            max(
                7,
                len(frame) * 0.42,
            ),
        )
    )

    image = ax.imshow(
        frame.values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        range(
            len(frame.columns)
        )
    )

    ax.set_xticklabels(
        frame.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(
            len(frame.index)
        )
    )

    ax.set_yticklabels(
        frame.index
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Player"
    )

    ax.set_title(
        f"{team_name} Wellness Data Absence by Month"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Fraction of calendar days "
        "with wellness data absent"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / output_name,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


create_heatmap(
    team_a,
    "Team A",
    "team_a_wellness_missingness_heatmap.png",
)

create_heatmap(
    team_b,
    "Team B",
    "team_b_wellness_missingness_heatmap.png",
)


# --------------------------------------------------
# Terminal summary only
# --------------------------------------------------

print(
    "Missingness visualization complete."
)

print(
    f"\nTables written to:\n"
    f"{TABLE_DIR}"
)

print(
    f"\nFigures written to:\n"
    f"{RESULTS_DIR}"
)

print(
    "\nTeam A players:",
    len(team_a),
)

print(
    "Team B players:",
    len(team_b),
)