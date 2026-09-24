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
    / "subjective_data"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "subjective_data"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FILE_INVENTORY = (
    AUDIT_DIR
    / "subjective_file_inventory.csv"
)

STRUCTURE_TABLES = (
    AUDIT_DIR
    / "subjective_structure_tables.csv"
)

PLAYER_MISSINGNESS = (
    AUDIT_DIR
    / "subjective_player_missingness.csv"
)

SESSION_NUMERIC = (
    AUDIT_DIR
    / "session_json_numeric_summary.csv"
)


# ============================================================
# Helper functions
# ============================================================

def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for candidate in candidates:

        if candidate in frame.columns:
            return candidate

    return None


def load_optional(
    path: Path,
) -> pd.DataFrame | None:

    if not path.exists():

        print(
            f"Skipping missing file: {path.name}"
        )

        return None

    frame = pd.read_csv(path)

    print()
    print(
        f"{path.name} columns:"
    )

    print(
        list(frame.columns)
    )

    return frame


# ============================================================
# Load Audit outputs
# ============================================================

file_inventory = load_optional(
    FILE_INVENTORY
)

structure_tables = load_optional(
    STRUCTURE_TABLES
)

player_missingness = load_optional(
    PLAYER_MISSINGNESS
)

session_numeric = load_optional(
    SESSION_NUMERIC
)


# ============================================================
# FIGURE 1
# Subjective rows by source/category
# ============================================================

if file_inventory is not None:

    category_col = find_column(
        file_inventory,
        [
            "category",
            "group",
            "folder",
            "data_type",
            "dataset_category",
        ],
    )

    row_col = find_column(
        file_inventory,
        [
            "rows",
            "row_count",
            "number_rows",
            "records",
        ],
    )

    if (
        category_col is not None
        and row_col is not None
    ):

        file_inventory[
            row_col
        ] = pd.to_numeric(
            file_inventory[
                row_col
            ],
            errors="coerce",
        )

        category_rows = (
            file_inventory
            .groupby(
                category_col
            )[
                row_col
            ]
            .sum()
            .sort_values(
                ascending=True
            )
        )

        category_rows.to_csv(
            TABLE_DIR
            / "subjective_rows_by_category.csv"
        )

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        ax.barh(
            category_rows.index,
            category_rows.values,
        )

        ax.set_title(
            "Subjective Dataset Rows by Category"
        )

        ax.set_xlabel(
            "Number of Rows"
        )

        ax.set_ylabel(
            "Subjective Data Category"
        )

        ax.grid(
            axis="x",
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR
            / "subjective_rows_by_category.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# FIGURE 2
# Missingness by subjective feature/table
# ============================================================

if structure_tables is not None:

    file_col = find_column(
        structure_tables,
        [
            "file",
            "filename",
            "table",
            "name",
            "dataset",
        ],
    )

    missing_col = find_column(
        structure_tables,
        [
            "missing_fraction",
            "missing_player_fraction",
            "missing_value_fraction",
            "missingness_fraction",
            "missing_percentage",
        ],
    )

    if (
        file_col is not None
        and missing_col is not None
    ):

        missing_plot = (
            structure_tables[
                [
                    file_col,
                    missing_col,
                ]
            ]
            .copy()
        )

        missing_plot[
            missing_col
        ] = pd.to_numeric(
            missing_plot[
                missing_col
            ],
            errors="coerce",
        )

        missing_plot = (
            missing_plot
            .dropna()
            .sort_values(
                missing_col,
                ascending=True,
            )
        )

        # If values are percentages rather than 0–1,
        # convert them to fractions.
        if (
            missing_plot[
                missing_col
            ].max()
            > 1
        ):

            missing_plot[
                missing_col
            ] = (
                missing_plot[
                    missing_col
                ]
                / 100
            )

        missing_plot.to_csv(
            TABLE_DIR
            / "subjective_missingness_by_table.csv",
            index=False,
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        ax.barh(
            missing_plot[
                file_col
            ],
            missing_plot[
                missing_col
            ],
        )

        ax.set_title(
            "Subjective Data Missingness by Table"
        )

        ax.set_xlabel(
            "Fraction of Values Missing"
        )

        ax.set_ylabel(
            "Subjective Table"
        )

        ax.set_xlim(
            0,
            1,
        )

        ax.grid(
            axis="x",
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR
            / "subjective_missingness_by_table.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# FIGURE 3
# Player-level subjective missingness
# ============================================================

if player_missingness is not None:

    player_col = find_column(
        player_missingness,
        [
            "player_name",
            "player",
            "player_id",
        ],
    )

    missing_col = find_column(
        player_missingness,
        [
            "missing_fraction",
            "missing_player_fraction",
            "missingness_fraction",
            "fraction_missing",
        ],
    )

    if (
        player_col is not None
        and missing_col is not None
    ):

        player_plot = (
            player_missingness[
                [
                    player_col,
                    missing_col,
                ]
            ]
            .copy()
        )

        player_plot[
            missing_col
        ] = pd.to_numeric(
            player_plot[
                missing_col
            ],
            errors="coerce",
        )

        player_plot = (
            player_plot
            .dropna()
        )

        if (
            player_plot[
                missing_col
            ].max()
            > 1
        ):

            player_plot[
                missing_col
            ] = (
                player_plot[
                    missing_col
                ]
                / 100
            )

        # If several rows exist per player
        # because of several wellness files,
        # average their missingness.
        player_plot = (
            player_plot
            .groupby(
                player_col,
                as_index=False,
            )[
                missing_col
            ]
            .mean()
            .sort_values(
                missing_col,
                ascending=True,
            )
        )

        # Short plotting labels
        player_plot[
            "plot_name"
        ] = (
            player_plot[
                player_col
            ]
            .astype(str)
            .apply(
                lambda x:
                (
                    x.split("-")[0]
                    + "-"
                    + x.split("-", 1)[1][:8]
                )
                if "-"
                in x
                else x[:12]
            )
        )

        player_plot.to_csv(
            TABLE_DIR
            / "subjective_missingness_by_player.csv",
            index=False,
        )

        fig, ax = plt.subplots(
            figsize=(
                11,
                max(
                    10,
                    len(
                        player_plot
                    )
                    * 0.30,
                ),
            )
        )

        ax.barh(
            player_plot[
                "plot_name"
            ],
            player_plot[
                missing_col
            ],
        )

        ax.set_title(
            "Average Subjective Missingness by Player"
        )

        ax.set_xlabel(
            "Fraction of Subjective Values Missing"
        )

        ax.set_ylabel(
            "Player"
        )

        ax.set_xlim(
            0,
            1,
        )

        ax.grid(
            axis="x",
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR
            / "subjective_missingness_by_player.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# FIGURE 4
# Session variable ranges
# ============================================================

if session_numeric is not None:

    required = {
        "column",
        "minimum",
        "maximum",
        "median",
    }

    if required.issubset(
        session_numeric.columns
    ):

        session_plot = (
            session_numeric[
                session_numeric[
                    "column"
                ].isin(
                    [
                        "rpe",
                        "duration",
                        "srpe",
                    ]
                )
            ]
            .copy()
        )

        for column in [
            "minimum",
            "maximum",
            "median",
        ]:

            session_plot[
                column
            ] = pd.to_numeric(
                session_plot[
                    column
                ],
                errors="coerce",
            )

        session_plot.to_csv(
            TABLE_DIR
            / "session_variable_ranges.csv",
            index=False,
        )

        x_positions = range(
            len(
                session_plot
            )
        )

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        for index, row in (
            session_plot
            .reset_index(
                drop=True
            )
            .iterrows()
        ):

            ax.vlines(
                x=index,
                ymin=row[
                    "minimum"
                ],
                ymax=row[
                    "maximum"
                ],
                linewidth=3,
            )

            ax.plot(
                index,
                row[
                    "median"
                ],
                marker="o",
                markersize=8,
            )

        ax.set_xticks(
            list(
                x_positions
            )
        )

        ax.set_xticklabels(
            session_plot[
                "column"
            ]
            .str.upper()
        )

        ax.set_title(
            "Observed Session Variable Ranges"
        )

        ax.set_xlabel(
            "Session Variable"
        )

        ax.set_ylabel(
            "Observed Value"
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR
            / "session_variable_ranges.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# Terminal summary
# ============================================================

print()
print(
    "Visual 05 complete."
)

print(
    f"\nFigures written to:\n"
    f"{FIGURE_DIR}"
)

print(
    f"\nTables written to:\n"
    f"{TABLE_DIR}"
)