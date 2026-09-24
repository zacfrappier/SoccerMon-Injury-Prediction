from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audit"
    / "objective_timestamp_multiplicity_file_audit.csv"
)

COLUMN_VARIATION = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audit"
    / "objective_timestamp_column_variation.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "sensor_sampling"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "sensor_sampling"
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

for path in [
    FILE_AUDIT,
    COLUMN_VARIATION,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required audit output not found: {path}"
        )


file_audit = pd.read_csv(
    FILE_AUDIT
)

column_variation = pd.read_csv(
    COLUMN_VARIATION
)


print(
    "File-audit columns:"
)

print(
    list(file_audit.columns)
)

print()

print(
    "Column-variation columns:"
)

print(
    list(column_variation.columns)
)


# ============================================================
# Helper:
# Find a column from several possible names
# ============================================================

def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for candidate in candidates:

        if candidate in frame.columns:
            return candidate

    return None


# ============================================================
# Detect important Audit 6 columns
# ============================================================

rows_per_timestamp_col = find_column(
    file_audit,
    [
        "median_rows_per_timestamp",
        "rows_per_timestamp",
        "mean_rows_per_timestamp",
        "median_timestamp_multiplicity",
    ],
)

timestamp_frequency_col = find_column(
    file_audit,
    [
        "unique_timestamps_per_second",
        "timestamp_frequency_hz",
        "unique_timestamp_frequency_hz",
        "timestamp_rate_hz",
        "timestamps_per_second",
    ],
)

row_frequency_col = find_column(
    file_audit,
    [
        "estimated_rows_per_second",
        "effective_row_frequency_hz",
        "row_frequency_hz",
        "rows_per_second",
        "effective_rows_per_second",
    ],
)


# ============================================================
# Detect column-variation fields
# ============================================================

sensor_name_col = find_column(
    column_variation,
    [
        "column",
        "sensor_column",
        "column_name",
        "feature",
    ],
)

variation_fraction_col = find_column(
    column_variation,
    [
        "varying_group_fraction",
        "variation_fraction",
        "fraction_groups_with_variation",
        "varying_fraction",
        "within_timestamp_variation_fraction",
    ],
)


# ============================================================
# Numeric conversion
# ============================================================

for column in [
    rows_per_timestamp_col,
    timestamp_frequency_col,
    row_frequency_col,
]:

    if column is not None:

        file_audit[column] = (
            pd.to_numeric(
                file_audit[column],
                errors="coerce",
            )
        )


if variation_fraction_col is not None:

    column_variation[
        variation_fraction_col
    ] = pd.to_numeric(
        column_variation[
            variation_fraction_col
        ],
        errors="coerce",
    )


# ============================================================
# Table 1:
# Sampling summary
# ============================================================

sampling_summary_rows = []

for label, column in [
    (
        "rows_per_timestamp",
        rows_per_timestamp_col,
    ),
    (
        "timestamp_frequency_hz",
        timestamp_frequency_col,
    ),
    (
        "effective_row_frequency_hz",
        row_frequency_col,
    ),
]:

    if column is None:
        continue

    values = (
        file_audit[column]
        .dropna()
    )

    sampling_summary_rows.append(
        {
            "metric": label,
            "files": len(values),
            "minimum": values.min(),
            "median": values.median(),
            "mean": values.mean(),
            "maximum": values.max(),
        }
    )


sampling_summary = pd.DataFrame(
    sampling_summary_rows
)

sampling_summary.to_csv(
    TABLE_DIR
    / "sensor_sampling_summary.csv",
    index=False,
)


# ============================================================
# Figure 1:
# Rows per timestamp
# ============================================================

if rows_per_timestamp_col is not None:

    values = (
        file_audit[
            rows_per_timestamp_col
        ]
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        values,
        bins=15,
    )

    median_value = (
        values.median()
    )

    ax.axvline(
        median_value,
        linestyle="--",
        label=(
            f"Median = "
            f"{median_value:.2f}"
        ),
    )

    ax.set_title(
        "Rows per Repeated Timestamp"
    )

    ax.set_xlabel(
        "Rows per Timestamp"
    )

    ax.set_ylabel(
        "Number of Sampled Files"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "rows_per_timestamp_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Figure 2:
# Timestamp frequency
# ============================================================

if timestamp_frequency_col is not None:

    values = (
        file_audit[
            timestamp_frequency_col
        ]
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        values,
        bins=15,
    )

    median_value = (
        values.median()
    )

    ax.axvline(
        median_value,
        linestyle="--",
        label=(
            f"Median = "
            f"{median_value:.2f} Hz"
        ),
    )

    ax.set_title(
        "Unique Timestamp Sampling Frequency"
    )

    ax.set_xlabel(
        "Unique Timestamps per Second (Hz)"
    )

    ax.set_ylabel(
        "Number of Sampled Files"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "timestamp_frequency_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Figure 3:
# Effective row frequency
# ============================================================

if row_frequency_col is not None:

    values = (
        file_audit[
            row_frequency_col
        ]
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        values,
        bins=15,
    )

    median_value = (
        values.median()
    )

    ax.axvline(
        median_value,
        linestyle="--",
        label=(
            f"Median = "
            f"{median_value:.2f} rows/s"
        ),
    )

    ax.set_title(
        "Effective Objective Sensor Row Frequency"
    )

    ax.set_xlabel(
        "Rows per Second"
    )

    ax.set_ylabel(
        "Number of Sampled Files"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "effective_row_frequency_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Figure 4:
# Within-timestamp sensor variation
# ============================================================

if (
    sensor_name_col is not None
    and variation_fraction_col is not None
):

    variation_plot = (
        column_variation[
            [
                sensor_name_col,
                variation_fraction_col,
            ]
        ]
        .dropna()
        .groupby(
            sensor_name_col,
            as_index=False,
        )[
            variation_fraction_col
        ]
        .mean()
        .sort_values(
            variation_fraction_col,
            ascending=True,
        )
    )

    variation_plot.to_csv(
        TABLE_DIR
        / "within_timestamp_sensor_variation.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        variation_plot[
            sensor_name_col
        ],
        variation_plot[
            variation_fraction_col
        ],
    )

    ax.set_title(
        "Sensor Variation Within Repeated Timestamps"
    )

    ax.set_xlabel(
        "Fraction of Timestamp Groups With Variation"
    )

    ax.set_ylabel(
        "Sensor Variable"
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
        / "within_timestamp_sensor_variation.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Terminal interpretation summary
# ============================================================

print()
print(
    "Visual 04 complete."
)

print(
    f"\nFigures written to:\n"
    f"{FIGURE_DIR}"
)

print(
    f"\nTables written to:\n"
    f"{TABLE_DIR}"
)


if rows_per_timestamp_col is not None:

    print(
        "\nMedian rows per timestamp:",
        round(
            file_audit[
                rows_per_timestamp_col
            ].median(),
            3,
        ),
    )


if timestamp_frequency_col is not None:

    print(
        "Median timestamp frequency:",
        round(
            file_audit[
                timestamp_frequency_col
            ].median(),
            3,
        ),
        "Hz",
    )


if row_frequency_col is not None:

    print(
        "Median effective row frequency:",
        round(
            file_audit[
                row_frequency_col
            ].median(),
            3,
        ),
        "rows/sec",
    )


if (
    sensor_name_col is not None
    and variation_fraction_col is not None
):

    print(
        "\nSensor variation summary:"
    )

    print(
        variation_plot.to_string(
            index=False
        )
    )