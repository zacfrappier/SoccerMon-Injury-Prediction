from __future__ import annotations

import json
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

SUBJECTIVE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "subjective"
    / "subjective"
)

SESSION_FILE = (
    SUBJECTIVE_ROOT
    / "training-load"
    / "session.json"
)

GAME_FILE = (
    SUBJECTIVE_ROOT
    / "game-performance"
    / "game-performance.csv"
)

MISMATCH_FILE = (
    AUDIT_DIR
    / "daily_load_alignment_same_date.csv"
)

OBJECTIVE_MANIFEST = (
    AUDIT_DIR
    / "objective_file_manifest.csv"
)


OUTPUT_DETAIL = (
    AUDIT_DIR
    / "daily_load_unaccounted_load_context.csv"
)

OUTPUT_FREQUENCY = (
    AUDIT_DIR
    / "daily_load_unaccounted_load_frequency.csv"
)

OUTPUT_PLAYER = (
    AUDIT_DIR
    / "daily_load_unaccounted_load_by_player.csv"
)

OUTPUT_SUMMARY = (
    AUDIT_DIR
    / "daily_load_unaccounted_load_summary.txt"
)


# ============================================================
# Load normalized session.json
# ============================================================

def load_sessions() -> pd.DataFrame:

    with SESSION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    frames = []

    for player_name, sessions in data.items():

        if not isinstance(sessions, list):
            continue

        if not sessions:
            continue

        frame = pd.json_normalize(
            sessions
        )

        frame.insert(
            0,
            "player_name",
            player_name,
        )

        frames.append(
            frame
        )

    if not frames:

        raise RuntimeError(
            "No session records found."
        )

    sessions = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    sessions["date"] = pd.to_datetime(
        sessions["date"],
        format="%d.%m.%Y",
        errors="coerce",
    )

    for column in [
        "srpe",
        "rpe",
        "duration",
    ]:

        if column in sessions.columns:

            sessions[column] = pd.to_numeric(
                sessions[column],
                errors="coerce",
            )

    return sessions


# ============================================================
# Aggregate session information by player-day
# ============================================================

def aggregate_sessions(
    sessions: pd.DataFrame,
) -> pd.DataFrame:

    return (
        sessions
        .groupby(
            [
                "player_name",
                "date",
            ],
            as_index=False,
        )
        .agg(
            session_count=(
                "srpe",
                "size",
            ),
            reconstructed_daily_load=(
                "srpe",
                "sum",
            ),
            minimum_session_srpe=(
                "srpe",
                "min",
            ),
            maximum_session_srpe=(
                "srpe",
                "max",
            ),
            mean_session_srpe=(
                "srpe",
                "mean",
            ),
        )
    )


# ============================================================
# Load the 110 corrected mismatches
# ============================================================

def load_mismatches() -> pd.DataFrame:

    frame = pd.read_csv(
        MISMATCH_FILE
    )

    # --------------------------------------------------------
    # Detect the reconstructed/session date column.
    #
    # Audit 11 merged reconstructed and provided date fields,
    # so pandas may name the reconstructed date
    # "date_reconstructed" rather than simply "date".
    # --------------------------------------------------------

    date_candidates = [
        "date_reconstructed",
        "date",
        "comparison_date",
    ]

    date_column = next(
        (
            column
            for column in date_candidates
            if column in frame.columns
        ),
        None,
    )

    if date_column is None:

        raise ValueError(
            "Could not identify reconstructed date column. "
            f"Available columns: {list(frame.columns)}"
        )

    frame[date_column] = pd.to_datetime(
        frame[date_column],
        errors="coerce",
    )

    # Use one standardized name throughout Audit 18.
    if date_column != "date":

        frame = frame.rename(
            columns={
                date_column: "date"
            }
        )

    # --------------------------------------------------------
    # Convert match flag
    # --------------------------------------------------------

    frame["matches"] = (
        frame["matches"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    for column in [
        "reconstructed_daily_load",
        "provided_daily_load",
        "difference",
        "absolute_difference",
    ]:

        if column in frame.columns:

            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Keep only the 110 mismatches
    # --------------------------------------------------------

    frame = frame[
        frame["matches"] == False
    ].copy()

    # Provided - reconstructed.
    #
    # Positive means the provided daily-load table contains
    # more load than can be reconstructed from session.json.
    frame[
        "unaccounted_load"
    ] = (
        frame[
            "provided_daily_load"
        ]
        - frame[
            "reconstructed_daily_load"
        ]
    )

    return frame


# ============================================================
# Add nearby session context
# ============================================================

def add_neighbor_context(
    mismatches: pd.DataFrame,
    daily_sessions: pd.DataFrame,
) -> pd.DataFrame:

    result = mismatches.copy()

    lookup = (
        daily_sessions
        .set_index(
            [
                "player_name",
                "date",
            ]
        )[
            "reconstructed_daily_load"
        ]
        .to_dict()
    )

    for offset in [
        -7,
        -2,
        -1,
        1,
        2,
        7,
    ]:

        column = (
            f"session_load_"
            f"{offset:+d}d"
        )

        result[column] = [
            lookup.get(
                (
                    player,
                    date
                    + pd.Timedelta(
                        days=offset
                    ),
                ),
                0.0,
            )
            for player, date
            in zip(
                result[
                    "player_name"
                ],
                result[
                    "date"
                ],
            )
        ]

    return result


# ============================================================
# Test whether unaccounted load equals nearby daily load
# ============================================================

def add_neighbor_match_flags(
    frame: pd.DataFrame,
    tolerance: float = 1e-6,
) -> pd.DataFrame:

    result = frame.copy()

    for offset in [
        -7,
        -2,
        -1,
        1,
        2,
        7,
    ]:

        load_column = (
            f"session_load_"
            f"{offset:+d}d"
        )

        flag_column = (
            f"unaccounted_equals_"
            f"{offset:+d}d_load"
        )

        result[
            flag_column
        ] = (
            (
                result[
                    "unaccounted_load"
                ]
                - result[
                    load_column
                ]
            )
            .abs()
            <= tolerance
        )

    return result


# ============================================================
# Add objective-data presence
# ============================================================

def add_objective_presence(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    result = frame.copy()

    result[
        "objective_data_on_date"
    ] = False

    result[
        "objective_files_on_date"
    ] = 0

    if not OBJECTIVE_MANIFEST.exists():

        return result

    objective = pd.read_csv(
        OBJECTIVE_MANIFEST
    )

    required = {
        "team",
        "player_id",
        "date",
    }

    if not required.issubset(
        objective.columns
    ):

        return result

    objective["date"] = pd.to_datetime(
        objective["date"],
        errors="coerce",
    )

    objective[
        "player_name"
    ] = (
        objective["team"]
        .astype(str)
        + "-"
        + objective["player_id"]
        .astype(str)
    )

    counts = (
        objective
        .groupby(
            [
                "player_name",
                "date",
            ]
        )
        .size()
        .rename(
            "objective_files_on_date"
        )
        .reset_index()
    )

    result = result.drop(
        columns=[
            "objective_data_on_date",
            "objective_files_on_date",
        ]
    )

    result = result.merge(
        counts,
        on=[
            "player_name",
            "date",
        ],
        how="left",
    )

    result[
        "objective_files_on_date"
    ] = (
        result[
            "objective_files_on_date"
        ]
        .fillna(0)
        .astype(int)
    )

    result[
        "objective_data_on_date"
    ] = (
        result[
            "objective_files_on_date"
        ]
        > 0
    )

    return result


# ============================================================
# Attempt game-performance date overlap
# ============================================================

def add_game_context(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    result = frame.copy()

    # Normalize mismatch dates to calendar dates
    result["date"] = (
        pd.to_datetime(
            result["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    result[
        "game_record_on_date"
    ] = False

    if not GAME_FILE.exists():

        return result

    game = pd.read_csv(
        GAME_FILE
    )

    print()
    print(
        "Game-performance columns:"
    )

    print(
        list(
            game.columns
        )
    )

    player_candidates = [
        "player_name",
        "player",
        "player_id",
    ]

    date_candidates = [
        "date",
        "game_date",
        "match_date",
        "timestamp",
    ]

    player_column = next(
        (
            column
            for column
            in player_candidates
            if column
            in game.columns
        ),
        None,
    )

    date_column = next(
        (
            column
            for column
            in date_candidates
            if column
            in game.columns
        ),
        None,
    )

    if (
        player_column is None
        or date_column is None
    ):

        print(
            "Could not automatically "
            "match game-performance "
            "player/date columns."
        )

        return result

    game[date_column] = (
        pd.to_datetime(
            game[date_column],
            dayfirst=True,
            errors="coerce",
        )
        .dt.normalize()
    )

    game_keys = set(
        zip(
            game[
                player_column
            ].astype(str),
            game[
                date_column
            ],
        )
    )


    result[
        "game_record_on_date"
    ] = [
        (
            str(player),
            date,
        )
        in game_keys
        for player, date
        in zip(
            result[
                "player_name"
            ],
            result[
                "date"
            ],
        )
    ]

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "Loading session records..."
    )

    sessions = load_sessions()

    print(
        f"Session records: "
        f"{len(sessions):,}"
    )

    daily_sessions = (
        aggregate_sessions(
            sessions
        )
    )

    mismatches = (
        load_mismatches()
    )

    print(
        f"Corrected mismatches: "
        f"{len(mismatches):,}"
    )

    context = (
        add_neighbor_context(
            mismatches,
            daily_sessions,
        )
    )

    context = (
        add_neighbor_match_flags(
            context
        )
    )

    context = (
        add_objective_presence(
            context
        )
    )

    context = (
        add_game_context(
            context
        )
    )


    # --------------------------------------------------------
    # Repeated unaccounted-load values
    # --------------------------------------------------------

    frequency = (
        context[
            "unaccounted_load"
        ]
        .value_counts()
        .rename_axis(
            "unaccounted_load"
        )
        .reset_index(
            name="count"
        )
        .sort_values(
            [
                "count",
                "unaccounted_load",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )


    # --------------------------------------------------------
    # Player summary
    # --------------------------------------------------------

    player_summary = (
        context
        .groupby(
            "player_name"
        )
        .agg(
            mismatch_days=(
                "date",
                "size",
            ),
            minimum_unaccounted_load=(
                "unaccounted_load",
                "min",
            ),
            median_unaccounted_load=(
                "unaccounted_load",
                "median",
            ),
            mean_unaccounted_load=(
                "unaccounted_load",
                "mean",
            ),
            maximum_unaccounted_load=(
                "unaccounted_load",
                "max",
            ),
            objective_overlap_days=(
                "objective_data_on_date",
                "sum",
            ),
            game_overlap_days=(
                "game_record_on_date",
                "sum",
            ),
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    neighbor_flag_columns = [
        column
        for column in context.columns
        if column.startswith(
            "unaccounted_equals_"
        )
    ]

    any_neighbor_match = (
        context[
            neighbor_flag_columns
        ]
        .any(
            axis=1
        )
        if neighbor_flag_columns
        else pd.Series(
            False,
            index=context.index,
        )
    )

    objective_overlap = int(
        context[
            "objective_data_on_date"
        ].sum()
    )

    game_overlap = int(
        context[
            "game_record_on_date"
        ].sum()
    )

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Unaccounted Daily-Load Audit"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        f"Mismatched player-days: "
        f"{len(context):,}"
    )

    lines.append(
        "All provided values greater "
        "than reconstructed: "
        f"{bool((context['unaccounted_load'] > 0).all())}"
    )

    lines.append("")

    lines.append(
        "Unaccounted load"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Minimum: "
        f"{context['unaccounted_load'].min():.2f}"
    )

    lines.append(
        f"Median: "
        f"{context['unaccounted_load'].median():.2f}"
    )

    lines.append(
        f"Mean: "
        f"{context['unaccounted_load'].mean():.2f}"
    )

    lines.append(
        f"Maximum: "
        f"{context['unaccounted_load'].max():.2f}"
    )

    lines.append("")

    lines.append(
        "Context checks"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "Unaccounted load equals one "
        "tested nearby day's session load: "
        f"{int(any_neighbor_match.sum()):,}"
    )

    lines.append(
        "Mismatch days with objective data: "
        f"{objective_overlap:,}"
    )

    lines.append(
        "Mismatch days with game-performance record: "
        f"{game_overlap:,}"
    )

    lines.append("")

    lines.append(
        "Nearby offsets tested: "
        "-7, -2, -1, +1, +2, +7 days"
    )

    lines.append("")

    lines.append(
        "Interpretation"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "This audit provides contextual evidence only. "
        "A nearby-load match, objective-data overlap, "
        "or game-performance overlap does not by itself "
        "establish the cause of a daily-load discrepancy."
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    context.to_csv(
        OUTPUT_DETAIL,
        index=False,
    )

    frequency.to_csv(
        OUTPUT_FREQUENCY,
        index=False,
    )

    player_summary.to_csv(
        OUTPUT_PLAYER,
        index=False,
    )

    OUTPUT_SUMMARY.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


    print()
    print(
        "\n".join(lines)
    )

    print()
    print(
        f"Detailed context written to:\n"
        f"{OUTPUT_DETAIL}"
    )

    print()
    print(
        f"Frequency table written to:\n"
        f"{OUTPUT_FREQUENCY}"
    )

    print()
    print(
        f"Player summary written to:\n"
        f"{OUTPUT_PLAYER}"
    )

    print()
    print(
        f"Summary written to:\n"
        f"{OUTPUT_SUMMARY}"
    )


if __name__ == "__main__":

    main()