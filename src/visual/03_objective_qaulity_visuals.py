from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audit"
    / "objective_schema_file_audit.csv"
)

MANIFEST_FILE = (
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
    / "objective_quality"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective_quality"
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
# Validate inputs
# ============================================================

if not SCHEMA_FILE.exists():
    raise FileNotFoundError(
        f"Audit 3 output not found: {SCHEMA_FILE}"
    )

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(
        f"Audit 1 manifest not found: {MANIFEST_FILE}"
    )


schema = pd.read_csv(
    SCHEMA_FILE
)

manifest = pd.read_csv(
    MANIFEST_FILE
)


required_schema_columns = {
    "path",
    "filename",
    "team",
    "year",
    "player_id",
    "row_count",
    "column_count",
    "row_groups",
    "schema_id",
    "error",
}

missing_columns = (
    required_schema_columns
    - set(schema.columns)
)

if missing_columns:
    raise ValueError(
        "Schema audit is missing required columns: "
        + ", ".join(
            sorted(missing_columns)
        )
    )


if "size_bytes" not in manifest.columns:
    raise ValueError(
        "Manifest does not contain size_bytes."
    )


# ============================================================
# Clean values
# ============================================================

schema["row_count"] = pd.to_numeric(
    schema["row_count"],
    errors="coerce",
)

schema["column_count"] = pd.to_numeric(
    schema["column_count"],
    errors="coerce",
)

schema["row_groups"] = pd.to_numeric(
    schema["row_groups"],
    errors="coerce",
)

schema["year"] = pd.to_numeric(
    schema["year"],
    errors="coerce",
)


# ============================================================
# Merge file size from Audit 1
# ============================================================

manifest_small = (
    manifest[
        [
            "path",
            "size_bytes",
        ]
    ]
    .drop_duplicates(
        subset=["path"]
    )
)

data = schema.merge(
    manifest_small,
    on="path",
    how="left",
)

data["size_mb"] = (
    data["size_bytes"]
    / (1024 ** 2)
)


# ============================================================
# Basic summary
# ============================================================

successful = data[
    data["error"].isna()
    | (
        data["error"]
        .astype(str)
        .str.strip()
        == ""
    )
].copy()


print(
    f"Files inspected: {len(data):,}"
)

print(
    f"Successful reads: {len(successful):,}"
)

print(
    f"Unique schemas: "
    f"{successful['schema_id'].nunique():,}"
)

print(
    f"Minimum rows: "
    f"{successful['row_count'].min():,.0f}"
)

print(
    f"Maximum rows: "
    f"{successful['row_count'].max():,.0f}"
)

print(
    f"Median rows: "
    f"{successful['row_count'].median():,.0f}"
)


# ============================================================
# Table 1:
# Structural summary
# ============================================================

structure_summary = pd.DataFrame(
    {
        "metric": [
            "files_inspected",
            "successful_reads",
            "unique_schemas",
            "minimum_rows",
            "maximum_rows",
            "mean_rows",
            "median_rows",
            "minimum_columns",
            "maximum_columns",
            "minimum_row_groups",
            "maximum_row_groups",
        ],
        "value": [
            len(data),
            len(successful),
            successful[
                "schema_id"
            ].nunique(),
            successful[
                "row_count"
            ].min(),
            successful[
                "row_count"
            ].max(),
            successful[
                "row_count"
            ].mean(),
            successful[
                "row_count"
            ].median(),
            successful[
                "column_count"
            ].min(),
            successful[
                "column_count"
            ].max(),
            successful[
                "row_groups"
            ].min(),
            successful[
                "row_groups"
            ].max(),
        ],
    }
)

structure_summary.to_csv(
    TABLE_DIR
    / "objective_structure_summary.csv",
    index=False,
)


# ============================================================
# Figure 1:
# Row-count distribution
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    successful[
        "row_count"
    ].dropna(),
    bins=15,
)

median_rows = (
    successful[
        "row_count"
    ].median()
)

ax.axvline(
    median_rows,
    linestyle="--",
    label=(
        f"Median = "
        f"{median_rows:,.0f}"
    ),
)

ax.set_title(
    "Distribution of Rows per Objective Parquet File"
)

ax.set_xlabel(
    "Sensor Rows per File"
)

ax.set_ylabel(
    "Number of Files"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "row_count_distribution.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 2:
# Row count by team
# ============================================================

team_a_rows = successful.loc[
    successful["team"] == "TeamA",
    "row_count",
].dropna()

team_b_rows = successful.loc[
    successful["team"] == "TeamB",
    "row_count",
].dropna()


fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.boxplot(
    [
        team_a_rows,
        team_b_rows,
    ],
    tick_labels=[
        "Team A",
        "Team B",
    ],
)

ax.set_title(
    "Objective Parquet Row Counts by Team"
)

ax.set_xlabel(
    "Team"
)

ax.set_ylabel(
    "Sensor Rows per File"
)

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "row_count_by_team.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 3:
# File size vs row count
# ============================================================

plot_data = successful.dropna(
    subset=[
        "row_count",
        "size_mb",
    ]
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.scatter(
    plot_data[
        "row_count"
    ],
    plot_data[
        "size_mb"
    ],
    alpha=0.7,
)

ax.set_title(
    "Objective Parquet File Size vs Row Count"
)

ax.set_xlabel(
    "Sensor Rows per File"
)

ax.set_ylabel(
    "Compressed Parquet Size (MB)"
)

ax.grid(
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "file_size_vs_row_count.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 4:
# Row counts by year
# ============================================================

year_groups = []

year_labels = []

for year in sorted(
    successful[
        "year"
    ].dropna().unique()
):

    values = successful.loc[
        successful[
            "year"
        ]
        == year,
        "row_count",
    ].dropna()

    year_groups.append(
        values
    )

    year_labels.append(
        str(int(year))
    )


fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.boxplot(
    year_groups,
    tick_labels=year_labels,
)

ax.set_title(
    "Objective Parquet Row Counts by Year"
)

ax.set_xlabel(
    "Year"
)

ax.set_ylabel(
    "Sensor Rows per File"
)

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "row_count_by_year.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Table 2:
# Team summary
# ============================================================

team_summary = (
    successful
    .groupby(
        "team"
    )
    .agg(
        inspected_files=(
            "filename",
            "count",
        ),
        mean_rows=(
            "row_count",
            "mean",
        ),
        median_rows=(
            "row_count",
            "median",
        ),
        minimum_rows=(
            "row_count",
            "min",
        ),
        maximum_rows=(
            "row_count",
            "max",
        ),
        mean_file_size_mb=(
            "size_mb",
            "mean",
        ),
    )
)

team_summary.to_csv(
    TABLE_DIR
    / "objective_quality_by_team.csv"
)


# ============================================================
# Table 3:
# Largest / smallest sampled files
# ============================================================

file_extremes = successful[
    [
        "filename",
        "team",
        "year",
        "player_id",
        "row_count",
        "size_mb",
    ]
].sort_values(
    "row_count",
    ascending=False,
)

file_extremes.to_csv(
    TABLE_DIR
    / "sampled_files_by_row_count.csv",
    index=False,
)


# ============================================================
# Terminal summary
# ============================================================

print()
print(
    "Visual 03 complete."
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
    "\nLargest sampled objective files:"
)

print(
    file_extremes
    .head(5)
    .to_string(
        index=False
    )
)

print(
    "\nSmallest sampled objective files:"
)

print(
    file_extremes
    .tail(5)
    .to_string(
        index=False
    )
)