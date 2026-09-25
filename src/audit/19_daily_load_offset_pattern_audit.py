from pathlib import Path

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
    / "daily_load_unaccounted_load_context.csv"
)

OUTPUT_OFFSET_SUMMARY = (
    AUDIT_DIR
    / "daily_load_offset_match_summary.csv"
)

OUTPUT_PLAYER_SUMMARY = (
    AUDIT_DIR
    / "daily_load_offset_matches_by_player.csv"
)

OUTPUT_DETAILS = (
    AUDIT_DIR
    / "daily_load_offset_match_details.csv"
)

OUTPUT_TEXT = (
    AUDIT_DIR
    / "daily_load_offset_pattern_summary.txt"
)


# ============================================================
# Offset definitions
# ============================================================

OFFSET_COLUMNS = {
    -7: "unaccounted_equals_-7d_load",
    -2: "unaccounted_equals_-2d_load",
    -1: "unaccounted_equals_-1d_load",
    1: "unaccounted_equals_+1d_load",
    2: "unaccounted_equals_+2d_load",
    7: "unaccounted_equals_+7d_load",
}


# ============================================================
# Boolean conversion helper
# ============================================================

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
            }
        )
        .fillna(False)
        .astype(bool)
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Audit 18 output not found: {INPUT_FILE}"
        )

    frame = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows loaded: {len(frame):,}"
    )


    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = {
        "player_name",
        "date",
        "unaccounted_load",
    }

    required_columns.update(
        OFFSET_COLUMNS.values()
    )

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:

        raise ValueError(
            "Audit 18 context file is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )


    # --------------------------------------------------------
    # Clean values
    # --------------------------------------------------------

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["unaccounted_load"] = (
        pd.to_numeric(
            frame["unaccounted_load"],
            errors="coerce",
        )
    )

    for column in OFFSET_COLUMNS.values():

        frame[column] = to_bool(
            frame[column]
        )


    # --------------------------------------------------------
    # Count matches per offset
    # --------------------------------------------------------

    offset_rows = []

    for offset, column in OFFSET_COLUMNS.items():

        match_count = int(
            frame[column].sum()
        )

        match_fraction = (
            match_count
            / len(frame)
            if len(frame)
            else 0.0
        )

        offset_rows.append(
            {
                "offset_days": offset,
                "match_count": match_count,
                "match_fraction": match_fraction,
            }
        )


    offset_summary = pd.DataFrame(
        offset_rows
    ).sort_values(
        "offset_days"
    )


    # --------------------------------------------------------
    # Determine how many offsets each mismatch matches
    # --------------------------------------------------------

    flag_columns = list(
        OFFSET_COLUMNS.values()
    )

    frame[
        "number_of_matching_offsets"
    ] = (
        frame[
            flag_columns
        ]
        .sum(
            axis=1
        )
    )


    frame[
        "any_nearby_match"
    ] = (
        frame[
            "number_of_matching_offsets"
        ]
        > 0
    )


    # --------------------------------------------------------
    # Store which offsets matched
    # --------------------------------------------------------

    def matched_offsets(
        row: pd.Series,
    ) -> str:

        matches = []

        for offset, column in OFFSET_COLUMNS.items():

            if bool(
                row[column]
            ):

                matches.append(
                    f"{offset:+d}"
                )

        return ";".join(
            matches
        )


    frame[
        "matching_offsets"
    ] = frame.apply(
        matched_offsets,
        axis=1,
    )


    # --------------------------------------------------------
    # Keep only rows with at least one nearby match
    # --------------------------------------------------------

    matched = frame[
        frame[
            "any_nearby_match"
        ]
    ].copy()


    # --------------------------------------------------------
    # Player-level offset summary
    # --------------------------------------------------------

    player_rows = []

    for player_name, player_frame in (
        frame.groupby(
            "player_name"
        )
    ):

        row = {
            "player_name": player_name,
            "mismatch_days": len(
                player_frame
            ),
            "days_with_any_nearby_match": int(
                player_frame[
                    "any_nearby_match"
                ].sum()
            ),
        }

        for offset, column in OFFSET_COLUMNS.items():

            row[
                f"matches_{offset:+d}d"
            ] = int(
                player_frame[
                    column
                ].sum()
            )

        player_rows.append(
            row
        )


    player_summary = pd.DataFrame(
        player_rows
    )


    # --------------------------------------------------------
    # Count rows matching multiple offsets
    # --------------------------------------------------------

    single_offset_days = int(
        (
            frame[
                "number_of_matching_offsets"
            ]
            == 1
        ).sum()
    )

    multiple_offset_days = int(
        (
            frame[
                "number_of_matching_offsets"
            ]
            > 1
        ).sum()
    )

    no_offset_days = int(
        (
            frame[
                "number_of_matching_offsets"
            ]
            == 0
        ).sum()
    )


    # --------------------------------------------------------
    # Directional totals
    # --------------------------------------------------------

    negative_columns = [
        OFFSET_COLUMNS[-7],
        OFFSET_COLUMNS[-2],
        OFFSET_COLUMNS[-1],
    ]

    positive_columns = [
        OFFSET_COLUMNS[1],
        OFFSET_COLUMNS[2],
        OFFSET_COLUMNS[7],
    ]


    frame[
        "matches_any_negative_offset"
    ] = (
        frame[
            negative_columns
        ]
        .any(
            axis=1
        )
    )

    frame[
        "matches_any_positive_offset"
    ] = (
        frame[
            positive_columns
        ]
        .any(
            axis=1
        )
    )


    negative_days = int(
        frame[
            "matches_any_negative_offset"
        ].sum()
    )

    positive_days = int(
        frame[
            "matches_any_positive_offset"
        ].sum()
    )


    # --------------------------------------------------------
    # Best offset
    # --------------------------------------------------------

    ranked_offsets = (
        offset_summary
        .sort_values(
            [
                "match_count",
                "offset_days",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    best_offset = (
        int(
            ranked_offsets.iloc[0][
                "offset_days"
            ]
        )
        if not ranked_offsets.empty
        else None
    )

    best_count = (
        int(
            ranked_offsets.iloc[0][
                "match_count"
            ]
        )
        if not ranked_offsets.empty
        else 0
    )


    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    offset_summary.to_csv(
        OUTPUT_OFFSET_SUMMARY,
        index=False,
    )

    player_summary.to_csv(
        OUTPUT_PLAYER_SUMMARY,
        index=False,
    )

    matched[
        [
            "player_name",
            "date",
            "unaccounted_load",
            "matching_offsets",
            "number_of_matching_offsets",
        ]
        + flag_columns
    ].sort_values(
        [
            "player_name",
            "date",
        ]
    ).to_csv(
        OUTPUT_DETAILS,
        index=False,
    )


    # --------------------------------------------------------
    # Text summary
    # --------------------------------------------------------

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Daily-Load Offset Pattern Audit"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        f"Mismatch player-days analyzed: "
        f"{len(frame):,}"
    )

    lines.append(
        f"Days with at least one nearby-load match: "
        f"{int(frame['any_nearby_match'].sum()):,}"
    )

    lines.append(
        f"Days with no nearby-load match: "
        f"{no_offset_days:,}"
    )

    lines.append(
        f"Days matching exactly one tested offset: "
        f"{single_offset_days:,}"
    )

    lines.append(
        f"Days matching multiple tested offsets: "
        f"{multiple_offset_days:,}"
    )

    lines.append("")

    lines.append(
        "Matches by offset"
    )

    lines.append(
        "-" * 80
    )

    for _, row in (
        offset_summary
        .sort_values(
            "offset_days"
        )
        .iterrows()
    ):

        lines.append(
            f"{int(row['offset_days']):+d} days: "
            f"{int(row['match_count']):,} "
            f"({row['match_fraction'] * 100:.2f}%)"
        )

    lines.append("")

    lines.append(
        "Directional summary"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Days matching any negative offset: "
        f"{negative_days:,}"
    )

    lines.append(
        f"Days matching any positive offset: "
        f"{positive_days:,}"
    )

    lines.append("")

    lines.append(
        "Most common tested offset"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"{best_offset:+d} days: "
        f"{best_count:,} matches"
        if best_offset is not None
        else "No matching offset found."
    )

    lines.append("")

    lines.append(
        "Interpretation"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "A dominant single offset would support further investigation "
        "of localized date-placement or reporting behavior. "
        "A diffuse pattern across several offsets would be more consistent "
        "with repeated common sRPE values or multiple possible nearby matches."
    )

    lines.append(
        "These matches are associative only and do not demonstrate that "
        "a session was shifted from one calendar date to another."
    )


    OUTPUT_TEXT.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print()

    print(
        "\n".join(lines)
    )

    print()

    print(
        f"Offset summary written to:\n"
        f"{OUTPUT_OFFSET_SUMMARY}"
    )

    print()

    print(
        f"Player summary written to:\n"
        f"{OUTPUT_PLAYER_SUMMARY}"
    )

    print()

    print(
        f"Matched details written to:\n"
        f"{OUTPUT_DETAILS}"
    )

    print()

    print(
        f"Text summary written to:\n"
        f"{OUTPUT_TEXT}"
    )


if __name__ == "__main__":

    main()