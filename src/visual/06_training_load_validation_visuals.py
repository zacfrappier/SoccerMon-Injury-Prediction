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

VALIDATION_FILE = (
    AUDIT_DIR
    / "daily_load_validation.csv"
)

MISMATCH_FILE = (
    AUDIT_DIR
    / "daily_load_mismatch_classification.csv"
)

ALIGNMENT_FILE = (
    AUDIT_DIR
    / "daily_load_alignment_summary.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "training_load_validation"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "training_load_validation"
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
# Validate required files
# ============================================================

for path in [
    VALIDATION_FILE,
    MISMATCH_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required audit output not found: {path}"
        )


validation = pd.read_csv(
    VALIDATION_FILE
)

mismatches = pd.read_csv(
    MISMATCH_FILE
)


# ============================================================
# Clean validation data
# ============================================================

numeric_columns = [
    "raw_srpe_sum",
    "deduplicated_srpe_sum",
    "provided_daily_load",
    "raw_difference",
    "deduplicated_difference",
]

for column in numeric_columns:

    if column in validation.columns:

        validation[column] = pd.to_numeric(
            validation[column],
            errors="coerce",
        )


if "date" in validation.columns:

    validation["date"] = pd.to_datetime(
        validation["date"],
        errors="coerce",
    )


# ============================================================
# Clean mismatch data
# ============================================================

for column in [
    "raw_srpe_sum",
    "deduplicated_srpe_sum",
    "provided_daily_load",
    "difference",
    "absolute_difference",
]:

    if column in mismatches.columns:

        mismatches[column] = pd.to_numeric(
            mismatches[column],
            errors="coerce",
        )


if "date" in mismatches.columns:

    mismatches["date"] = pd.to_datetime(
        mismatches["date"],
        errors="coerce",
    )


# ============================================================
# FIGURE 1
# Provided vs reconstructed daily load
# ============================================================

comparison = validation.dropna(
    subset=[
        "raw_srpe_sum",
        "provided_daily_load",
    ]
).copy()


fig, ax = plt.subplots(
    figsize=(8, 8)
)

ax.scatter(
    comparison[
        "raw_srpe_sum"
    ],
    comparison[
        "provided_daily_load"
    ],
    alpha=0.35,
    s=15,
)

minimum = min(
    comparison[
        "raw_srpe_sum"
    ].min(),
    comparison[
        "provided_daily_load"
    ].min(),
)

maximum = max(
    comparison[
        "raw_srpe_sum"
    ].max(),
    comparison[
        "provided_daily_load"
    ].max(),
)

ax.plot(
    [
        minimum,
        maximum,
    ],
    [
        minimum,
        maximum,
    ],
    linestyle="--",
    label="Perfect agreement",
)

ax.set_title(
    "Provided vs Reconstructed Daily Training Load"
)

ax.set_xlabel(
    "Reconstructed Daily Load "
    "(sum of session sRPE)"
)

ax.set_ylabel(
    "Provided Daily Load"
)

ax.legend()

ax.grid(
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "provided_vs_reconstructed_daily_load.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 2
# Distribution of mismatch magnitude
# ============================================================

mismatch_values = mismatches[
    "absolute_difference"
].dropna()


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    mismatch_values,
    bins=15,
)

median_difference = (
    mismatch_values.median()
)

ax.axvline(
    median_difference,
    linestyle="--",
    label=(
        f"Median mismatch = "
        f"{median_difference:.0f}"
    ),
)

ax.set_title(
    "Magnitude of Daily-Load Mismatches"
)

ax.set_xlabel(
    "Absolute Difference"
)

ax.set_ylabel(
    "Number of Mismatched Player-Days"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "daily_load_difference_distribution.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 3
# Mismatches by player
# ============================================================

mismatches_by_player = (
    mismatches
    .groupby(
        "player_name"
    )
    .size()
    .sort_values(
        ascending=True
    )
)

mismatches_by_player.to_csv(
    TABLE_DIR
    / "mismatches_by_player.csv"
)


short_labels = []

for player in (
    mismatches_by_player.index
):

    if "-" in player:

        team, player_id = (
            player.split(
                "-",
                1,
            )
        )

        short_labels.append(
            f"{team}-{player_id[:8]}"
        )

    else:

        short_labels.append(
            player[:12]
        )


fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.barh(
    short_labels,
    mismatches_by_player.values,
)

ax.set_title(
    "Daily-Load Mismatches by Player"
)

ax.set_xlabel(
    "Number of Mismatched Player-Days"
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
    / "mismatches_by_player.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 4
# Mismatch timeline
# ============================================================

timeline = (
    mismatches
    .dropna(
        subset=[
            "date",
            "player_name",
        ]
    )
    .copy()
)

players = sorted(
    timeline[
        "player_name"
    ].unique()
)

player_to_y = {
    player: index
    for index, player
    in enumerate(players)
}


fig, ax = plt.subplots(
    figsize=(14, 6)
)

for player in players:

    player_data = timeline[
        timeline[
            "player_name"
        ]
        == player
    ]

    y_value = (
        player_to_y[
            player
        ]
    )

    ax.scatter(
        player_data[
            "date"
        ],
        [
            y_value
        ]
        * len(
            player_data
        ),
        s=35,
    )


short_player_labels = []

for player in players:

    if "-" in player:

        team, player_id = (
            player.split(
                "-",
                1,
            )
        )

        short_player_labels.append(
            f"{team}-{player_id[:8]}"
        )

    else:

        short_player_labels.append(
            player[:12]
        )


ax.set_yticks(
    range(
        len(players)
    )
)

ax.set_yticklabels(
    short_player_labels
)

ax.set_title(
    "Timeline of Daily-Load Mismatches"
)

ax.set_xlabel(
    "Date"
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
    / "mismatch_timeline.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Save mismatch detail table
# ============================================================

mismatches.sort_values(
    [
        "player_name",
        "date",
    ]
).to_csv(
    TABLE_DIR
    / "daily_load_mismatch_details.csv",
    index=False,
)


# ============================================================
# Optional alignment table
# ============================================================

if ALIGNMENT_FILE.exists():

    alignment = pd.read_csv(
        ALIGNMENT_FILE
    )

    alignment.to_csv(
        TABLE_DIR
        / "alignment_summary.csv",
        index=False,
    )


# ============================================================
# Terminal summary
# ============================================================

total_compared = len(
    comparison
)

exact_matches = (
    comparison[
        "raw_difference"
    ]
    .fillna(float("inf"))
    .eq(0)
    .sum()
)

match_percentage = (
    exact_matches
    / total_compared
    * 100
    if total_compared
    else 0
)


print()
print(
    "Visual 06 complete."
)

print()
print(
    f"Player-days compared: "
    f"{total_compared:,}"
)

print(
    f"Exact matches: "
    f"{exact_matches:,}"
)

print(
    f"Exact match percentage: "
    f"{match_percentage:.2f}%"
)

print(
    f"Mismatched player-days: "
    f"{len(mismatches):,}"
)

print(
    f"Players with mismatches: "
    f"{mismatches['player_name'].nunique():,}"
)

print(
    f"\nFigures written to:\n"
    f"{FIGURE_DIR}"
)

print(
    f"\nTables written to:\n"
    f"{TABLE_DIR}"
)