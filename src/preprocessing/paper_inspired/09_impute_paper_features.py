
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 01. Paths and settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_DIR = (
    PROJECT_ROOT / "data" / "modeling" / "paper_inspired"
)

INPUT_FILE = (
    BASE_DIR / "features" / "paper_features_unimputed.csv"
)

OUTPUT_DIR = BASE_DIR / "imputation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "imputation_summary.txt"
COVERAGE_FILE = OUTPUT_DIR / "imputation_coverage.csv"

LOOKBACK_DAYS = 14
MIN_OVERLAP_DAYS = 3

# Features intended as model inputs.
# Do not impute injury outcomes, presence indicators,
# identifiers, or injury-event metadata.

CANDIDATE_FEATURES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_duration",
    "sleep_quality",
    "soreness",
    "stress",
    "rpe_mean",
    "rpe_max",
    "duration_total",
    "duration_mean",
    "srpe_total",
    "srpe_mean",
    "srpe_max",
    "duration_obj",
    "Speed_km_h_mean",
    "Speed_km_h_max",
    "Speed_km_h_std",
    "sp_lir_p",
    "sp_lir_t",
    "sp_lir_d",
    "sp_mir_p",
    "sp_mir_t",
    "sp_mir_d",
    "sp_hir_p",
    "sp_hir_t",
    "sp_hir_d",
    "sp_spr_p",
    "sp_spr_t",
    "sp_spr_d",
    "Distance",
    "distance_per_min",
    "daily_load",
    "atl_7d",
    "weekly_load_7d",
    "monotony_7d",
    "strain_7d",
    "ctl28",
    "ctl42",
    "acwr_7_28",
]

# Already-derived, fully observed history features.
# Preserve them without replacement.
PRESERVED_FEATURES = [
    "subjective_missingness_7d",
    "past_injury_count",
]


# ============================================================
# 02. Load data
# ============================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = {"player_name", "team", "date"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.normalize()

    if df.duplicated(["player_name", "date"]).any():
        raise ValueError("Duplicate player-date records found.")

    if not df["team"].eq("TeamB").all():
        raise ValueError("Expected TeamB-only data.")

    features = [
        f for f in CANDIDATE_FEATURES
        if f in df.columns
    ]

    absent = sorted(set(CANDIDATE_FEATURES) - set(features))
    if absent:
        print("Unavailable candidate features:", absent)

    for feature in features:
        df[feature] = pd.to_numeric(
            df[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    df = df.sort_values(
        ["player_name", "date"]
    ).reset_index(drop=True)

    return df, features


# ============================================================
# 03. Player median imputation
#
# Exploratory full-cohort version.
# For final evaluation, fit medians on training data only.
# ============================================================

def impute_player_median(df, features):
    result = df.copy()

    for feature in features:
        player_medians = result.groupby(
            "player_name"
        )[feature].transform("median")

        result[feature] = result[feature].fillna(
            player_medians
        )

    return result


# ============================================================
# 04. Linear interpolation
#
# Time-based interpolation, independently per player.
# Only fills internal gaps between real observations.
#
# IMPORTANT:
# This uses later observations and is therefore
# retrospective. Final evaluation must prevent
# train/test leakage.
# ============================================================

def impute_linear(df, features):
    groups = []

    for player, group in df.groupby(
        "player_name", sort=True
    ):
        group = group.sort_values("date").copy()

        for feature in features:
            series = pd.Series(
                group[feature].to_numpy(dtype=float),
                index=pd.DatetimeIndex(group["date"]),
            )

            interpolated = series.interpolate(
                method="time",
                limit_area="inside",
            )

            group[feature] = interpolated.to_numpy()

        groups.append(group)

    return pd.concat(
        groups, ignore_index=True
    ).sort_values(
        ["player_name", "date"]
    ).reset_index(drop=True)


# ============================================================
# 05. Bespoke teammate-relative imputation
#
# Documented reconstruction, NOT the authors'
# unpublished exact formula.
#
# For a missing feature on date t:
#
# 1. Find the player's previous 14 calendar days.
# 2. Find teammate medians on corresponding dates.
# 3. Estimate the player's relative standing using
#    paired historical observations.
# 4. Apply that ratio to the teammate median on t.
#
# Requires >=3 paired historical observations.
# Does not use future dates.
# Does not fill unsupported gaps.
# ============================================================

def impute_bespoke(df, features):
    result = df.copy()

    # All reference statistics come from observed data,
    # never from newly imputed values.
    original = df.copy()

    for feature in features:
        print(f"  Bespoke feature: {feature}")

        # Each date/player gets the median of OTHER
        # teammates' observed feature values.
        peer_reference = {}

        for (team, date), day in original.groupby(
            ["team", "date"], sort=False
        ):
            valid = day[
                ["player_name", feature]
            ].dropna()

            for player in day["player_name"]:
                others = valid.loc[
                    valid["player_name"] != player,
                    feature,
                ]

                peer_reference[(player, date)] = (
                    float(others.median())
                    if not others.empty
                    else np.nan
                )

        for player, group in original.groupby(
            "player_name", sort=False
        ):
            group = group.sort_values("date")
            observed = group.set_index("date")[feature]

            for idx in group.index:
                if pd.notna(original.at[idx, feature]):
                    continue

                date = original.at[idx, "date"]

                current_peer = peer_reference.get(
                    (player, date), np.nan
                )

                if not np.isfinite(current_peer):
                    continue

                history_start = date - pd.Timedelta(
                    days=LOOKBACK_DAYS
                )

                history = observed.loc[
                    (observed.index >= history_start)
                    & (observed.index < date)
                ].dropna()

                ratios = []

                for prior_date, value in history.items():
                    prior_peer = peer_reference.get(
                        (player, prior_date), np.nan
                    )

                    if (
                        np.isfinite(prior_peer)
                        and abs(prior_peer) > 1e-9
                    ):
                        ratios.append(value / prior_peer)

                if len(ratios) < MIN_OVERLAP_DAYS:
                    continue

                relative_standing = float(
                    np.median(ratios)
                )

                estimate = relative_standing * current_peer

                if not np.isfinite(estimate):
                    continue

                # Most candidate features are nonnegative.
                # Do not manufacture negative workload,
                # duration, or sensor exposure values.
                estimate = max(0.0, estimate)

                # Preserve valid proportions.
                if feature.endswith("_p"):
                    estimate = np.clip(estimate, 0, 1)

                result.at[idx, feature] = estimate

    return result


# ============================================================
# 06. Coverage comparison
# ============================================================

def build_coverage(original, outputs, features):
    rows = []

    for feature in features:
        original_missing = int(
            original[feature].isna().sum()
        )

        for method, frame in outputs.items():
            remaining = int(
                frame[feature].isna().sum()
            )

            filled = original_missing - remaining

            rows.append({
                "method": method,
                "feature": feature,
                "rows": len(frame),
                "original_missing": original_missing,
                "filled": filled,
                "remaining_missing": remaining,
                "remaining_missing_fraction":
                    remaining / len(frame),
                "original_median":
                    original[feature].median(),
                "imputed_median":
                    frame[feature].median(),
                "original_mean":
                    original[feature].mean(),
                "imputed_mean":
                    frame[feature].mean(),
            })

    return pd.DataFrame(rows)


# ============================================================
# 07. Integrity checks
# ============================================================

def validate_output(original, output, features, method):
    if len(original) != len(output):
        raise ValueError(f"{method}: row count changed")

    keys = ["player_name", "team", "date"]

    if not original[keys].equals(output[keys]):
        raise ValueError(f"{method}: player-date alignment changed")

    # Existing observed measurements must be unchanged.
    for feature in features:
        observed = original[feature].notna()

        if not np.allclose(
            original.loc[observed, feature],
            output.loc[observed, feature],
            equal_nan=True,
        ):
            raise ValueError(
                f"{method}: observed values changed: {feature}"
            )

        if np.isinf(output[feature].to_numpy()).any():
            raise ValueError(
                f"{method}: infinite values: {feature}"
            )

    # Do not modify outcomes or event metadata.
    protected = [
        "injury_event",
        "injury_episode_count",
        "minor_episode_count",
        "major_episode_count",
        "has_injury",
        "has_subjective",
        "has_training",
        "has_objective",
        *PRESERVED_FEATURES,
    ]

    for column in protected:
        if column in original.columns:
            if not original[column].equals(output[column]):
                raise ValueError(
                    f"{method}: protected column changed: {column}"
                )


# ============================================================
# 08. Main
# ============================================================

def main():
    print("=" * 80)
    print("SoccerMon Paper-Inspired Imputation Comparison")
    print("=" * 80)

    original, features = load_data()

    print(f"Players: {original['player_name'].nunique()}")
    print(f"Player-days: {len(original):,}")
    print(f"Candidate features: {len(features)}")
    print()

    print("Running player median...")
    median = impute_player_median(original, features)

    print("Running linear interpolation...")
    linear = impute_linear(original, features)

    print("Running bespoke teammate-relative...")
    bespoke = impute_bespoke(original, features)

    outputs = {
        "median": median,
        "linear": linear,
        "bespoke": bespoke,
    }

    for method, frame in outputs.items():
        validate_output(
            original, frame, features, method
        )

        path = OUTPUT_DIR / f"paper_features_{method}.csv"

        frame.to_csv(path, index=False)
        print(f"Saved: {path}")

    coverage = build_coverage(
        original, outputs, features
    )

    coverage.to_csv(
        COVERAGE_FILE, index=False
    )

    lines = [
        "=" * 80,
        "SoccerMon Imputation Summary",
        "=" * 80,
        "",
        f"Players: {original['player_name'].nunique()}",
        f"Player-days: {len(original):,}",
        f"Candidate features: {len(features)}",
        "",
        "Method comparison",
        "-" * 80,
    ]

    for method, frame in outputs.items():
        before = int(
            original[features].isna().sum().sum()
        )
        after = int(
            frame[features].isna().sum().sum()
        )

        lines.extend([
            f"{method}:",
            f"  Missing cells before: {before:,}",
            f"  Missing cells after:  {after:,}",
            f"  Cells imputed:        {before - after:,}",
            "",
        ])

    lines.extend([
        "Interpretation",
        "-" * 80,
        "These are exploratory full-cohort imputation outputs.",
        "The bespoke method is a documented reconstruction.",
        "Injury outcomes and presence indicators were preserved.",
        "No remaining NaNs were automatically filled with zero.",
        "Final evaluation requires fold-specific preprocessing.",
    ])

    SUMMARY_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))
    print()
    print(f"Coverage: {COVERAGE_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
