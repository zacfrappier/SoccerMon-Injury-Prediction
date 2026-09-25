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

OVERLAP_FILE = (
    AUDIT_DIR
    / "player_overlap_audit.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "player_overlap"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "player_overlap"
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

if not OVERLAP_FILE.exists():

    raise FileNotFoundError(
        f"Audit 14 output not found: {OVERLAP_FILE}"
    )


df = pd.read_csv(
    OVERLAP_FILE
)


print(
    "Player-overlap columns:"
)

print(
    list(df.columns)
)

print()

print(
    f"Rows loaded: {len(df):,}"
)


# ============================================================
# Helpers
# ============================================================

def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for candidate in candidates:

        if candidate in frame.columns:

            return candidate

    return None


def to_bool(
    series: pd.Series,
) -> pd.Series:

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
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


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


# ============================================================
# Detect columns
# ============================================================

player_col = find_column(
    df,
    [
        "player_name",
        "player",
        "player_id",
    ],
)

objective_col = find_column(
    df,
    [
        "in_objective",
        "objective",
        "has_objective",
        "objective_present",
    ],
)

subjective_col = find_column(
    df,
    [
        "in_subjective",
        "subjective",
        "has_subjective",
        "subjective_present",
    ],
)

session_col = find_column(
    df,
    [
        "in_session_json",
        "in_session",
        "session_json",
        "has_session",
    ],
)

injury_col = find_column(
    df,
    [
        "in_injury",
        "injury",
        "has_injury",
        "injury_present",
    ],
)


if player_col is None:

    raise ValueError(
        "Could not identify player column. "
        f"Available columns: {list(df.columns)}"
    )


print(
    f"Detected player column: "
    f"{player_col}"
)

print(
    f"Detected objective column: "
    f"{objective_col}"
)

print(
    f"Detected subjective column: "
    f"{subjective_col}"
)

print(
    f"Detected session column: "
    f"{session_col}"
)

print(
    f"Detected injury column: "
    f"{injury_col}"
)


# ============================================================
# Convert presence columns to Boolean
# ============================================================

presence_columns = [
    objective_col,
    subjective_col,
    session_col,
    injury_col,
]

for column in presence_columns:

    if column is not None:

        df[column] = to_bool(
            df[column]
        )


# ============================================================
# FIGURE 1
# Population overlap categories
# ============================================================

if (
    objective_col is not None
    and subjective_col is not None
):

    objective_only = (
        df[
            objective_col
        ]
        & ~df[
            subjective_col
        ]
    ).sum()

    subjective_only = (
        ~df[
            objective_col
        ]
        & df[
            subjective_col
        ]
    ).sum()

    both = (
        df[
            objective_col
        ]
        & df[
            subjective_col
        ]
    ).sum()


    overlap_summary = pd.DataFrame(
        {
            "population": [
                "Objective only",
                "Objective + subjective",
                "Subjective only",
            ],
            "players": [
                objective_only,
                both,
                subjective_only,
            ],
        }
    )


    overlap_summary.to_csv(
        TABLE_DIR
        / "objective_subjective_overlap.csv",
        index=False,
    )


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        overlap_summary[
            "population"
        ],
        overlap_summary[
            "players"
        ],
    )

    ax.set_title(
        "Objective and Subjective Player Overlap"
    )

    ax.set_xlabel(
        "Player Population"
    )

    ax.set_ylabel(
        "Number of Players"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for index, value in enumerate(
        overlap_summary[
            "players"
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
        / "objective_subjective_overlap.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FIGURE 2
# Players available by source
# ============================================================

source_counts = []

source_labels = []


for label, column in [
    (
        "Objective",
        objective_col,
    ),
    (
        "Subjective",
        subjective_col,
    ),
    (
        "Session JSON",
        session_col,
    ),
    (
        "Injury",
        injury_col,
    ),
]:

    if column is None:

        continue

    source_labels.append(
        label
    )

    source_counts.append(
        int(
            df[column].sum()
        )
    )


if source_counts:

    source_summary = pd.DataFrame(
        {
            "source": source_labels,
            "players": source_counts,
        }
    )

    source_summary.to_csv(
        TABLE_DIR
        / "players_by_data_source.csv",
        index=False,
    )


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        source_summary[
            "source"
        ],
        source_summary[
            "players"
        ],
    )

    ax.set_title(
        "Players Represented by Data Source"
    )

    ax.set_xlabel(
        "Data Source"
    )

    ax.set_ylabel(
        "Number of Unique Players"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for index, value in enumerate(
        source_summary[
            "players"
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
        / "players_by_data_source.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FIGURE 3
# Player × data-source availability matrix
# ============================================================

matrix_columns = []

matrix_labels = []


for label, column in [
    (
        "Objective",
        objective_col,
    ),
    (
        "Subjective",
        subjective_col,
    ),
    (
        "Session",
        session_col,
    ),
    (
        "Injury",
        injury_col,
    ),
]:

    if column is None:

        continue

    matrix_columns.append(
        column
    )

    matrix_labels.append(
        label
    )


if matrix_columns:

    matrix = (
        df[
            [
                player_col
            ]
            + matrix_columns
        ]
        .copy()
    )


    # Sort players by number of sources available
    matrix[
        "source_count"
    ] = (
        matrix[
            matrix_columns
        ]
        .sum(
            axis=1
        )
    )


    matrix = matrix.sort_values(
        [
            "source_count",
            player_col,
        ],
        ascending=[
            False,
            True,
        ],
    )


    matrix[
        "plot_name"
    ] = (
        matrix[
            player_col
        ]
        .apply(
            short_player_name
        )
    )


    matrix.to_csv(
        TABLE_DIR
        / "player_source_availability_matrix.csv",
        index=False,
    )


    fig, ax = plt.subplots(
        figsize=(
            8,
            max(
                12,
                len(matrix)
                * 0.27,
            ),
        )
    )


    image = ax.imshow(
        matrix[
            matrix_columns
        ]
        .astype(int)
        .values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )


    ax.set_xticks(
        range(
            len(
                matrix_columns
            )
        )
    )

    ax.set_xticklabels(
        matrix_labels
    )


    ax.set_yticks(
        range(
            len(matrix)
        )
    )

    ax.set_yticklabels(
        matrix[
            "plot_name"
        ],
        fontsize=7,
    )


    ax.set_title(
        "Player Availability Across SoccerMon Data Sources"
    )

    ax.set_xlabel(
        "Data Source"
    )

    ax.set_ylabel(
        "Player"
    )


    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_ticks(
        [
            0,
            1,
        ]
    )

    colorbar.set_ticklabels(
        [
            "Absent",
            "Present",
        ]
    )


    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "player_source_availability_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FIGURE 4
# Modeling population sizes
# ============================================================

if (
    objective_col is not None
    and subjective_col is not None
):

    objective_population = int(
        df[
            objective_col
        ].sum()
    )

    subjective_population = int(
        df[
            subjective_col
        ].sum()
    )

    multimodal_population = int(
        (
            df[
                objective_col
            ]
            & df[
                subjective_col
            ]
        ).sum()
    )


    modeling_labels = [
        "Objective\navailable",
        "Subjective\navailable",
        "Objective +\nSubjective",
    ]

    modeling_values = [
        objective_population,
        subjective_population,
        multimodal_population,
    ]


    if injury_col is not None:

        injured_multimodal = int(
            (
                df[
                    objective_col
                ]
                & df[
                    subjective_col
                ]
                & df[
                    injury_col
                ]
            ).sum()
        )

        modeling_labels.append(
            "Injured +\nMultimodal"
        )

        modeling_values.append(
            injured_multimodal
        )


    modeling_summary = pd.DataFrame(
        {
            "population": modeling_labels,
            "players": modeling_values,
        }
    )


    modeling_summary.to_csv(
        TABLE_DIR
        / "modeling_population_summary.csv",
        index=False,
    )


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        modeling_labels,
        modeling_values,
    )

    ax.set_title(
        "Potential Modeling Populations"
    )

    ax.set_ylabel(
        "Number of Players"
    )

    ax.set_xlabel(
        "Available Data"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )


    for index, value in enumerate(
        modeling_values
    ):

        ax.text(
            index,
            value,
            str(
                value
            ),
            ha="center",
            va="bottom",
        )


    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "modeling_population_sizes.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Terminal summary
# ============================================================

print()
print(
    "Visual 07 complete."
)

print(
    f"\nFigures written to:\n"
    f"{FIGURE_DIR}"
)

print(
    f"\nTables written to:\n"
    f"{TABLE_DIR}"
)


if (
    objective_col is not None
    and subjective_col is not None
):

    print()

    print(
        "Objective players:",
        int(
            df[
                objective_col
            ].sum()
        ),
    )

    print(
        "Subjective players:",
        int(
            df[
                subjective_col
            ].sum()
        ),
    )

    print(
        "Objective + subjective players:",
        int(
            (
                df[
                    objective_col
                ]
                & df[
                    subjective_col
                ]
            ).sum()
        ),
    )

    print(
        "Objective-only players:",
        int(
            (
                df[
                    objective_col
                ]
                & ~df[
                    subjective_col
                ]
            ).sum()
        ),
    )

    print(
        "Subjective-only players:",
        int(
            (
                ~df[
                    objective_col
                ]
                & df[
                    subjective_col
                ]
            ).sum()
        ),
    )


if injury_col is not None:

    print(
        "Players with injury records:",
        int(
            df[
                injury_col
            ].sum()
        ),
    )