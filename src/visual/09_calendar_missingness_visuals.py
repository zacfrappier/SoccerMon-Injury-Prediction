from __future__ import annotations

import calendar
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

INPUT_FILE = (
    AUDIT_DIR
    / "calendar_missing_dates.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "calendar_missingness"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "calendar_missingness"
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
# Wellness features
#
# A day with all seven wellness features missing should count
# as ONE missing calendar day, not seven missing days.
# ============================================================

WELLNESS_FEATURES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_duration",
    "sleep_quality",
    "soreness",
    "stress",
]


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


def create_monthly_heatmap(
    frame: pd.DataFrame,
    team_name: str,
    output_name: str,
) -> None:

    if frame.empty:

        print(
            f"No rows available for {team_name}. "
            "Skipping heatmap."
        )

        return


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
        frame.index,
        fontsize=8,
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
        FIGURE_DIR
        / output_name,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Validate input
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Audit 17 output not found: {INPUT_FILE}"
    )


# ============================================================
# Load Audit 17 output
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print(
    "Calendar-missingness columns:"
)

print(
    list(
        df.columns
    )
)

print()

print(
    f"Rows loaded: {len(df):,}"
)


required_columns = {
    "player_name",
    "date",
    "feature",
}


missing_columns = (
    required_columns
    - set(
        df.columns
    )
)


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(
            sorted(
                missing_columns
            )
        )
    )


# ============================================================
# Clean dates
# ============================================================

df[
    "date"
] = pd.to_datetime(
    df[
        "date"
    ],
    errors="coerce",
)


df = df.dropna(
    subset=[
        "player_name",
        "date",
    ]
)


# ============================================================
# Keep wellness features
# ============================================================

wellness_missing = df[
    df[
        "feature"
    ].isin(
        WELLNESS_FEATURES
    )
].copy()


print(
    f"Wellness missing-value rows: "
    f"{len(wellness_missing):,}"
)


# ============================================================
# Reduce feature-level missingness to player-day missingness
#
# Important:
#
# If fatigue, mood, readiness, etc. are all missing on the
# same day, that date counts as ONE missing day.
# ============================================================

missing_days = (
    wellness_missing[
        [
            "player_name",
            "date",
        ]
    ]
    .drop_duplicates()
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


print(
    f"Unique missing player-days: "
    f"{len(missing_days):,}"
)


# ============================================================
# Add team
# ============================================================

missing_days[
    "team"
] = (
    missing_days[
        "player_name"
    ]
    .astype(str)
    .str.split(
        "-",
        n=1,
    )
    .str[0]
)


# ============================================================
# Add month
# ============================================================

missing_days[
    "month"
] = (
    missing_days[
        "date"
    ]
    .dt.to_period(
        "M"
    )
    .astype(str)
)


# ============================================================
# FIGURES 1 & 2
# Monthly missingness heatmaps
# ============================================================

monthly_counts = (
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


monthly_counts = (
    monthly_counts
    .reindex(
        sorted(
            monthly_counts.columns
        ),
        axis=1,
    )
)


monthly_fraction = (
    monthly_counts
    .copy()
    .astype(float)
)


for month_column in (
    monthly_fraction.columns
):

    year, month = map(
        int,
        month_column.split(
            "-"
        ),
    )

    days_in_month = (
        calendar.monthrange(
            year,
            month,
        )[1]
    )

    monthly_fraction[
        month_column
    ] = (
        monthly_fraction[
            month_column
        ]
        / days_in_month
    )


# Save full player IDs before shortening labels
monthly_counts.to_csv(
    TABLE_DIR
    / "missing_days_by_player_month.csv"
)

monthly_fraction.to_csv(
    TABLE_DIR
    / "missing_fraction_by_player_month.csv"
)


monthly_fraction_plot = (
    monthly_fraction.copy()
)


monthly_fraction_plot.index = [
    short_player_name(
        player
    )
    for player
    in monthly_fraction_plot.index
]


team_a = monthly_fraction_plot[
    monthly_fraction_plot.index
    .str.startswith(
        "TeamA-"
    )
]


team_b = monthly_fraction_plot[
    monthly_fraction_plot.index
    .str.startswith(
        "TeamB-"
    )
]


create_monthly_heatmap(
    team_a,
    "Team A",
    "team_a_monthly_missingness_heatmap.png",
)


create_monthly_heatmap(
    team_b,
    "Team B",
    "team_b_monthly_missingness_heatmap.png",
)


# ============================================================
# FIGURE 3
# Total missing calendar days by player
# ============================================================

player_missing = (
    missing_days
    .groupby(
        [
            "player_name",
            "team",
        ],
        as_index=False,
    )[
        "date"
    ]
    .nunique()
    .rename(
        columns={
            "date":
            "missing_days"
        }
    )
    .sort_values(
        "missing_days",
        ascending=True,
    )
)


player_missing[
    "plot_name"
] = (
    player_missing[
        "player_name"
    ]
    .apply(
        short_player_name
    )
)


player_missing.to_csv(
    TABLE_DIR
    / "missing_days_by_player.csv",
    index=False,
)


fig, ax = plt.subplots(
    figsize=(
        11,
        max(
            10,
            len(
                player_missing
            )
            * 0.28,
        ),
    )
)


ax.barh(
    player_missing[
        "plot_name"
    ],
    player_missing[
        "missing_days"
    ],
)


ax.set_title(
    "Missing Wellness Calendar Days by Player"
)

ax.set_xlabel(
    "Number of Missing Calendar Days"
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
    / "missing_days_by_player.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Derive consecutive missing-day streaks
# ============================================================

streak_rows = []


for player_name, player_frame in (
    missing_days
    .groupby(
        "player_name"
    )
):

    dates = (
        player_frame[
            "date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


    if not dates:

        continue


    streak_start = dates[0]
    previous_date = dates[0]
    streak_length = 1


    for current_date in dates[1:]:

        difference = (
            current_date
            - previous_date
        ).days


        if difference == 1:

            streak_length += 1

        else:

            streak_rows.append(
                {
                    "player_name":
                    player_name,

                    "team":
                    str(
                        player_name
                    ).split(
                        "-",
                        1,
                    )[0],

                    "streak_start":
                    streak_start,

                    "streak_end":
                    previous_date,

                    "streak_length_days":
                    streak_length,
                }
            )

            streak_start = (
                current_date
            )

            streak_length = 1


        previous_date = (
            current_date
        )


    streak_rows.append(
        {
            "player_name":
            player_name,

            "team":
            str(
                player_name
            ).split(
                "-",
                1,
            )[0],

            "streak_start":
            streak_start,

            "streak_end":
            previous_date,

            "streak_length_days":
            streak_length,
        }
    )


streaks = pd.DataFrame(
    streak_rows
)


streaks.to_csv(
    TABLE_DIR
    / "missing_day_streaks.csv",
    index=False,
)


# ============================================================
# FIGURE 4
# Distribution of missing gap lengths
# ============================================================

if not streaks.empty:

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )


    max_streak = int(
        streaks[
            "streak_length_days"
        ].max()
    )


    bins = range(
        1,
        max_streak + 2,
    )


    ax.hist(
        streaks[
            "streak_length_days"
        ],
        bins=bins,
        align="left",
    )


    median_streak = (
        streaks[
            "streak_length_days"
        ]
        .median()
    )


    ax.axvline(
        median_streak,
        linestyle="--",
        label=(
            f"Median = "
            f"{median_streak:.1f} days"
        ),
    )


    ax.set_title(
        "Distribution of Consecutive Missing-Day Streaks"
    )

    ax.set_xlabel(
        "Consecutive Missing Days"
    )

    ax.set_ylabel(
        "Number of Missingness Streaks"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend()


    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "missing_gap_length_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FIGURE 5
# Number of players missing by calendar date
# ============================================================

daily_missing = (
    missing_days
    .groupby(
        "date"
    )[
        "player_name"
    ]
    .nunique()
    .reset_index(
        name="players_missing"
    )
    .sort_values(
        "date"
    )
)


daily_missing.to_csv(
    TABLE_DIR
    / "players_missing_by_date.csv",
    index=False,
)


fig, ax = plt.subplots(
    figsize=(16, 6)
)


ax.plot(
    daily_missing[
        "date"
    ],
    daily_missing[
        "players_missing"
    ],
)


ax.set_title(
    "Players With Missing Wellness Data Over Time"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Number of Players Missing"
)

ax.grid(
    alpha=0.3,
)


plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "missing_days_over_time.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Additional summary tables
# ============================================================

player_streak_summary = (
    streaks
    .groupby(
        "player_name",
        as_index=False,
    )
    .agg(
        missing_streaks=(
            "streak_length_days",
            "size",
        ),
        median_streak_days=(
            "streak_length_days",
            "median",
        ),
        mean_streak_days=(
            "streak_length_days",
            "mean",
        ),
        longest_streak_days=(
            "streak_length_days",
            "max",
        ),
    )
    if not streaks.empty
    else pd.DataFrame()
)


if not player_streak_summary.empty:

    player_streak_summary.to_csv(
        TABLE_DIR
        / "missing_streak_summary_by_player.csv",
        index=False,
    )


# ============================================================
# Terminal summary
# ============================================================

print()

print(
    "Visual 09 complete."
)

print()

print(
    "Players represented:",
    missing_days[
        "player_name"
    ].nunique(),
)

print(
    "Unique missing player-days:",
    len(
        missing_days
    ),
)

print(
    "Total missing streaks:",
    len(
        streaks
    ),
)


if not streaks.empty:

    print(
        "Median missing streak length:",
        round(
            streaks[
                "streak_length_days"
            ].median(),
            2,
        ),
    )

    print(
        "Longest missing streak:",
        int(
            streaks[
                "streak_length_days"
            ].max()
        ),
    )


if not player_missing.empty:

    worst_player = (
        player_missing
        .sort_values(
            "missing_days",
            ascending=False,
        )
        .iloc[0]
    )

    print(
        "Most missing days for one player:",
        int(
            worst_player[
                "missing_days"
            ]
        ),
    )

    print(
        "Player with most missing days:",
        worst_player[
            "player_name"
        ],
    )


print(
    f"\nFigures written to:\n"
    f"{FIGURE_DIR}"
)

print(
    f"\nTables written to:\n"
    f"{TABLE_DIR}"
)