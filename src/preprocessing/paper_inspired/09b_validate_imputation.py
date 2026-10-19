
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 01. Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data/modeling/paper_inspired"

ORIGINAL_FILE = (
    BASE / "features/paper_features_unimputed.csv"
)

IMPUTATION_DIR = BASE / "imputation"
OUTPUT_DIR = IMPUTATION_DIR / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["median", "linear", "bespoke"]

KEYS = ["player_name", "team", "date"]

# Only these features should be imputed by Script 09.
FEATURES = [
    "fatigue", "mood", "readiness",
    "sleep_duration", "sleep_quality",
    "soreness", "stress",
    "rpe_mean", "rpe_max",
    "duration_total", "duration_mean",
    "srpe_total", "srpe_mean", "srpe_max",
    "duration_obj",
    "Speed_km_h_mean", "Speed_km_h_max",
    "Speed_km_h_std",
    "sp_lir_p", "sp_lir_t", "sp_lir_d",
    "sp_mir_p", "sp_mir_t", "sp_mir_d",
    "sp_hir_p", "sp_hir_t", "sp_hir_d",
    "sp_spr_p", "sp_spr_t", "sp_spr_d",
    "Distance", "distance_per_min",
    "daily_load", "atl_7d", "weekly_load_7d",
    "monotony_7d", "strain_7d",
    "ctl28", "ctl42", "acwr_7_28",
]

PROTECTED = [
    "injury_event", "injury_episode_count",
    "minor_episode_count", "major_episode_count",
    "unique_body_regions", "body_regions",
    "has_injury", "has_subjective",
    "has_training", "has_objective",
    "subjective_missingness_7d",
    "past_injury_count",
]


# ============================================================
# 02. Load and align
# ============================================================

def load_table(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    if df.duplicated(KEYS).any():
        raise ValueError(f"Duplicate keys in {path}")

    return df.sort_values(KEYS).reset_index(drop=True)


# ============================================================
# 03. Validate one method
# ============================================================

def validate_method(original, result, method, features):
    errors = []
    rows = []

    if len(original) != len(result):
        errors.append("Row count changed")

    if not original[KEYS].equals(result[KEYS]):
        errors.append("Player/date alignment changed")
        return rows, errors

    for column in PROTECTED:
        if column in original.columns:
            if column not in result.columns:
                errors.append(f"Protected column missing: {column}")
            elif not original[column].equals(result[column]):
                errors.append(f"Protected column changed: {column}")

    for feature in features:
        before = pd.to_numeric(
            original[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        after = pd.to_numeric(
            result[feature], errors="coerce"
        )

        observed = before.notna()
        originally_missing = ~observed

        # Existing observed values must not change.
        changed = int((
            ~np.isclose(
                before[observed].to_numpy(),
                after[observed].to_numpy(),
                equal_nan=True,
                rtol=1e-9,
                atol=1e-9,
            )
        ).sum())

        newly_filled = int(
            (originally_missing & after.notna()).sum()
        )

        remaining = int(after.isna().sum())
        infinite = int(np.isinf(after.to_numpy()).sum())

        negative = int((after.dropna() < 0).sum())

        out_of_range = 0
        if feature.endswith("_p"):
            out_of_range = int(
                ((after.dropna() < 0) |
                 (after.dropna() > 1)).sum()
            )

        if changed:
            errors.append(
                f"{feature}: {changed} observed values changed"
            )

        if infinite:
            errors.append(
                f"{feature}: {infinite} infinite values"
            )

        if out_of_range:
            errors.append(
                f"{feature}: {out_of_range} invalid proportions"
            )

        rows.append({
            "method": method,
            "feature": feature,
            "original_observed": int(observed.sum()),
            "original_missing": int(originally_missing.sum()),
            "newly_filled": newly_filled,
            "remaining_missing": remaining,
            "remaining_missing_fraction":
                remaining / len(original),
            "observed_values_changed": changed,
            "infinite_values": infinite,
            "negative_values": negative,
            "out_of_range_proportions": out_of_range,
            "original_mean": before.mean(),
            "imputed_mean": after.mean(),
            "original_median": before.median(),
            "imputed_median": after.median(),
            "mean_change": after.mean() - before.mean(),
            "median_change": after.median() - before.median(),
        })

    return rows, errors


# ============================================================
# 04. Main validation
# ============================================================

def main():
    print("=" * 80)
    print("SoccerMon Script 09 Imputation Validation")
    print("=" * 80)

    original = load_table(ORIGINAL_FILE)

    features = [f for f in FEATURES if f in original.columns]

    if not features:
        raise ValueError("No candidate features found")

    print(f"Original rows: {len(original):,}")
    print(f"Players: {original['player_name'].nunique()}")
    print(f"Features: {len(features)}")

    all_rows = []
    all_errors = []

    for method in METHODS:
        print(f"\nValidating {method}...")

        path = (
            IMPUTATION_DIR /
            f"paper_features_{method}.csv"
        )

        result = load_table(path)

        rows, errors = validate_method(
            original, result, method, features
        )

        all_rows.extend(rows)

        for error in errors:
            all_errors.append({
                "method": method,
                "issue": error,
            })

        print(f"Validation issues: {len(errors)}")

    details = pd.DataFrame(all_rows)

    summary = (
        details.groupby("method", as_index=False)
        .agg(
            original_missing=("original_missing", "sum"),
            newly_filled=("newly_filled", "sum"),
            remaining_missing=("remaining_missing", "sum"),
            observed_values_changed=(
                "observed_values_changed", "sum"
            ),
            infinite_values=("infinite_values", "sum"),
            invalid_proportions=(
                "out_of_range_proportions", "sum"
            ),
        )
    )

    summary["fraction_filled"] = (
        summary["newly_filled"] /
        summary["original_missing"].replace(0, np.nan)
    )

    issues = pd.DataFrame(
        all_errors, columns=["method", "issue"]
    )

    details.to_csv(
        OUTPUT_DIR / "feature_validation_details.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "imputation_validation_summary.csv",
        index=False,
    )

    issues.to_csv(
        OUTPUT_DIR / "validation_issues.csv",
        index=False,
    )

    text = [
        "=" * 80,
        "SoccerMon Imputation Validation Summary",
        "=" * 80,
        "",
        f"Players: {original['player_name'].nunique()}",
        f"Rows: {len(original):,}",
        f"Features checked: {len(features)}",
        "",
        summary.to_string(index=False),
        "",
        f"Validation issues: {len(issues)}",
        "",
        "Interpretation:",
        "Observed values must remain unchanged.",
        "Remaining missing values are reported, not filled.",
        "This audit does not prove imputation accuracy.",
        "Leakage-safe fold-specific imputation is still required.",
    ]

    (OUTPUT_DIR / "validation_summary.txt").write_text(
        "\n".join(text) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(text))
    print()
    print(f"Outputs written to: {OUTPUT_DIR}")

    if not issues.empty:
        print("\nValidation issues:")
        print(issues.to_string(index=False))


if __name__ == "__main__":
    main()
