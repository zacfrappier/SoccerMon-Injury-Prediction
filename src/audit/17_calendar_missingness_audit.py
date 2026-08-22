from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


WELLNESS_FILES = [
    "fatigue.csv",
    "mood.csv",
    "readiness.csv",
    "sleep_duration.csv",
    "sleep_quality.csv",
    "soreness.csv",
    "stress.csv",
]

TRAINING_LOAD_FILES = [
    "daily_load.csv",
    "weekly_load.csv",
    "atl.csv",
    "ctl28.csv",
    "ctl42.csv",
    "acwr.csv",
    "monotony.csv",
    "strain.csv",
]


def find_subjective_root() -> Path:
    root = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "subjective"
        / "subjective"
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Subjective root not found: {root}"
        )

    return root


def load_overlap_table() -> pd.DataFrame:
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "audit"
        / "player_overlap_audit.csv"
    )

    frame = pd.read_csv(path)

    return frame


def load_objective_manifest() -> pd.DataFrame:
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "audit"
        / "objective_file_manifest.csv"
    )

    frame = pd.read_csv(path)

    frame["player_name"] = (
        frame["team"].astype(str)
        + "-"
        + frame["player_id"].astype(str)
    )

    frame["date"] = pd.to_datetime(
        frame["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    return frame[
        [
            "player_name",
            "date",
        ]
    ].drop_duplicates()


def load_wide_table(
    path: Path,
    value_name: str,
) -> pd.DataFrame:
    """
    Convert a SoccerMon wide table into:

    player_name | date | <value_name>
    """
    frame = pd.read_csv(path)

    date_column = str(
        frame.columns[0]
    )

    frame[date_column] = pd.to_datetime(
        frame[date_column],
        format="%d.%m.%Y",
        errors="coerce",
    )

    player_columns = [
        column
        for column in frame.columns
        if str(column).startswith(
            ("TeamA-", "TeamB-")
        )
    ]

    long_frame = frame.melt(
        id_vars=[date_column],
        value_vars=player_columns,
        var_name="player_name",
        value_name=value_name,
    )

    long_frame = long_frame.rename(
        columns={
            date_column: "date"
        }
    )

    return long_frame


def load_session_dates(
    session_path: Path,
) -> pd.DataFrame:
    """
    Return one row per player/date that has at least one session record.
    """
    with session_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    rows: list[dict[str, object]] = []

    for player_name, sessions in data.items():

        if not isinstance(sessions, list):
            continue

        for session in sessions:

            date_value = session.get(
                "date"
            )

            rows.append(
                {
                    "player_name": (
                        player_name
                    ),
                    "date": date_value,
                }
            )

    frame = pd.DataFrame(rows)

    frame["date"] = pd.to_datetime(
        frame["date"],
        format="%d.%m.%Y",
        errors="coerce",
    )

    return (
        frame
        .dropna(
            subset=["date"]
        )
        .drop_duplicates(
            subset=[
                "player_name",
                "date",
            ]
        )
    )


def load_injury_dates(
    injury_path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        injury_path
    )

    frame["date"] = pd.to_datetime(
        frame["timestamp"],
        format="%d.%m.%Y",
        errors="coerce",
    )

    result = (
        frame[
            [
                "player_name",
                "date",
            ]
        ]
        .dropna(
            subset=["date"]
        )
        .drop_duplicates()
        .copy()
    )

    result[
        "injury_report_present"
    ] = True

    return result


def get_subjective_players(
    subjective_root: Path,
) -> list[str]:
    """
    Read player columns from daily_load.csv.
    """
    path = (
        subjective_root
        / "training-load"
        / "daily_load.csv"
    )

    frame = pd.read_csv(
        path,
        nrows=1,
    )

    return [
        str(column)
        for column in frame.columns
        if str(column).startswith(
            ("TeamA-", "TeamB-")
        )
    ]


def determine_player_windows(
    players: list[str],
    objective: pd.DataFrame,
    session_dates: pd.DataFrame,
    subjective_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Determine each player's observed calendar window.

    We use the earliest/latest date seen in any non-event
    predictor source:
      - objective
      - session.json
      - wellness/training-load tables

    Injury reports are not used to define the observation window
    because they can occur before objective monitoring and are
    event-based.
    """
    rows: list[
        dict[str, object]
    ] = []

    for player_name in players:

        dates: list[
            pd.Timestamp
        ] = []

        objective_dates = objective.loc[
            objective[
                "player_name"
            ]
            == player_name,
            "date",
        ].dropna()

        dates.extend(
            objective_dates.tolist()
        )

        session_player_dates = (
            session_dates.loc[
                session_dates[
                    "player_name"
                ]
                == player_name,
                "date",
            ]
            .dropna()
        )

        dates.extend(
            session_player_dates.tolist()
        )

        for table in subjective_tables:

            table_dates = (
                table.loc[
                    table[
                        "player_name"
                    ]
                    == player_name,
                    "date",
                ]
                .dropna()
            )

            dates.extend(
                table_dates.tolist()
            )

        if dates:
            first_date = min(dates)
            last_date = max(dates)
        else:
            first_date = pd.NaT
            last_date = pd.NaT

        rows.append(
            {
                "player_name": (
                    player_name
                ),
                "observation_start": (
                    first_date
                ),
                "observation_end": (
                    last_date
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_master_calendar(
    players: list[str],
    player_windows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per player/calendar day inside that player's
    observed predictor window.
    """
    rows: list[
        pd.DataFrame
    ] = []

    for _, row in (
        player_windows.iterrows()
    ):

        player_name = (
            row["player_name"]
        )

        start = (
            row[
                "observation_start"
            ]
        )

        end = (
            row[
                "observation_end"
            ]
        )

        if (
            pd.isna(start)
            or pd.isna(end)
        ):
            continue

        dates = pd.date_range(
            start=start,
            end=end,
            freq="D",
        )

        player_calendar = pd.DataFrame(
            {
                "player_name": (
                    player_name
                ),
                "date": dates,
            }
        )

        rows.append(
            player_calendar
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "player_name",
                "date",
            ]
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def add_presence_flag(
    calendar: pd.DataFrame,
    source: pd.DataFrame,
    flag_name: str,
) -> pd.DataFrame:
    """
    Add True/False indicating whether a player/date exists
    in a source table.
    """
    source_keys = (
        source[
            [
                "player_name",
                "date",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    source_keys[
        flag_name
    ] = True

    merged = calendar.merge(
        source_keys,
        on=[
            "player_name",
            "date",
        ],
        how="left",
    )

    merged[
        flag_name
    ] = (
        merged[
            flag_name
        ]
        .fillna(False)
        .astype(bool)
    )

    return merged


def add_value_table(
    calendar: pd.DataFrame,
    table: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """
    Add both the actual value and a presence flag.
    """
    merged = calendar.merge(
        table,
        on=[
            "player_name",
            "date",
        ],
        how="left",
    )

    merged[
        f"{value_column}_present"
    ] = (
        merged[
            value_column
        ].notna()
    )

    return merged


def add_injury_reports(
    calendar: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    merged = calendar.merge(
        injuries,
        on=[
            "player_name",
            "date",
        ],
        how="left",
    )

    merged[
        "injury_report_present"
    ] = (
        merged[
            "injury_report_present"
        ]
        .fillna(False)
        .astype(bool)
    )

    return merged


def add_core_flag(
    calendar: pd.DataFrame,
    overlap: pd.DataFrame,
) -> pd.DataFrame:
    core_map = (
        overlap[
            [
                "player_name",
                "in_all_core_sources",
            ]
        ]
        .copy()
    )

    return calendar.merge(
        core_map,
        on="player_name",
        how="left",
    )


def build_missing_date_table(
    calendar: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Convert missing feature values into one row per
    player/date/feature absence.
    """
    rows: list[
        pd.DataFrame
    ] = []

    for feature in feature_names:

        present_column = (
            f"{feature}_present"
        )

        missing = calendar[
            ~calendar[
                present_column
            ]
        ][
            [
                "player_name",
                "date",
                "in_all_core_sources",
            ]
        ].copy()

        missing[
            "feature"
        ] = feature

        rows.append(
            missing
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def build_player_missingness_summary(
    calendar: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for player_name, group in (
        calendar.groupby(
            "player_name"
        )
    ):

        row: dict[
            str,
            object
        ] = {
            "player_name": (
                player_name
            ),
            "calendar_days": (
                len(group)
            ),
            "first_date": (
                group[
                    "date"
                ].min()
            ),
            "last_date": (
                group[
                    "date"
                ].max()
            ),
        }

        for feature in (
            feature_names
        ):

            present_column = (
                f"{feature}_present"
            )

            present_count = int(
                group[
                    present_column
                ].sum()
            )

            missing_count = (
                len(group)
                - present_count
            )

            row[
                f"{feature}_present_days"
            ] = (
                present_count
            )

            row[
                f"{feature}_missing_days"
            ] = (
                missing_count
            )

            row[
                f"{feature}_missing_fraction"
            ] = (
                missing_count
                / len(group)
                if len(group) > 0
                else 0.0
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_feature_missingness_summary(
    calendar: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for feature in feature_names:

        present_column = (
            f"{feature}_present"
        )

        total = len(
            calendar
        )

        present = int(
            calendar[
                present_column
            ].sum()
        )

        missing = (
            total
            - present
        )

        rows.append(
            {
                "feature": (
                    feature
                ),
                "calendar_cells": (
                    total
                ),
                "present_days": (
                    present
                ),
                "missing_days": (
                    missing
                ),
                "missing_fraction": (
                    missing
                    / total
                    if total > 0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_missing_streaks(
    calendar: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Identify consecutive missing-date streaks for each
    player and feature.
    """
    rows: list[
        dict[str, object]
    ] = []

    for feature in (
        feature_names
    ):

        present_column = (
            f"{feature}_present"
        )

        for player_name, group in (
            calendar.groupby(
                "player_name"
            )
        ):

            player = (
                group
                .sort_values(
                    "date"
                )
                .copy()
            )

            missing = (
                ~player[
                    present_column
                ]
            )

            streak_group = (
                missing
                != missing.shift()
            ).cumsum()

            player[
                "_missing"
            ] = missing

            player[
                "_streak_group"
            ] = streak_group

            for _, streak in (
                player[
                    player[
                        "_missing"
                    ]
                ]
                .groupby(
                    "_streak_group"
                )
            ):

                rows.append(
                    {
                        "player_name": (
                            player_name
                        ),
                        "feature": (
                            feature
                        ),
                        "streak_start": (
                            streak[
                                "date"
                            ].min()
                        ),
                        "streak_end": (
                            streak[
                                "date"
                            ].max()
                        ),
                        "missing_days": (
                            len(streak)
                        ),
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "missing_days",
            ascending=False,
        )
    )


def write_summary(
    calendar: pd.DataFrame,
    feature_summary: pd.DataFrame,
    player_summary: pd.DataFrame,
    missing_dates: pd.DataFrame,
    streaks: pd.DataFrame,
    output_path: Path,
) -> None:
    lines: list[str] = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "SoccerMon Calendar Missingness Audit"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Players in calendar: "
        f"{calendar['player_name'].nunique():,}"
    )

    lines.append(
        f"Player-day rows: "
        f"{len(calendar):,}"
    )

    lines.append(
        f"Calendar date range: "
        f"{calendar['date'].min()} "
        f"to "
        f"{calendar['date'].max()}"
    )

    lines.append("")
    lines.append(
        "Feature missingness"
    )
    lines.append(
        "-" * 80
    )

    for _, row in (
        feature_summary.iterrows()
    ):

        lines.append(
            f"{row['feature']} | "
            f"present: "
            f"{int(row['present_days']):,} | "
            f"missing: "
            f"{int(row['missing_days']):,} | "
            f"missing fraction: "
            f"{row['missing_fraction']:.4f}"
        )

    lines.append("")
    lines.append(
        "Largest missing streaks"
    )
    lines.append(
        "-" * 80
    )

    for _, row in (
        streaks
        .head(20)
        .iterrows()
    ):

        lines.append(
            f"{row['player_name']} | "
            f"{row['feature']} | "
            f"{row['streak_start']} "
            f"to "
            f"{row['streak_end']} | "
            f"{int(row['missing_days'])} days"
        )

    lines.append("")
    lines.append(
        "Interpretation notes"
    )
    lines.append(
        "-" * 80
    )

    lines.append(
        "Wellness absence may represent missing daily questionnaire data."
    )

    lines.append(
        "Training-load absence does not automatically mean missing data; "
        "the player may not have had an activity/session."
    )

    lines.append(
        "Objective absence does not automatically mean sensor-data failure; "
        "GPS monitoring did not occur for every possible player-day."
    )

    lines.append(
        "Injury-report absence is not treated as missingness because injury "
        "reporting is event/status based rather than required daily."
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(lines)
    )

    print()

    print(
        f"Summary written to: "
        f"{output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a SoccerMon player-by-calendar-date "
            "coverage and missingness audit."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "audit"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subjective_root = (
        find_subjective_root()
    )

    print(
        "Loading player/source data..."
    )

    overlap = (
        load_overlap_table()
    )

    objective = (
        load_objective_manifest()
    )

    session_dates = (
        load_session_dates(
            subjective_root
            / "training-load"
            / "session.json"
        )
    )

    injuries = (
        load_injury_dates(
            subjective_root
            / "injury"
            / "injury.csv"
        )
    )

    players = (
        get_subjective_players(
            subjective_root
        )
    )

    print(
        f"Subjective players: "
        f"{len(players):,}"
    )

    wellness_tables: dict[
        str,
        pd.DataFrame
    ] = {}

    training_tables: dict[
        str,
        pd.DataFrame
    ] = {}

    for filename in WELLNESS_FILES:

        feature = (
            filename
            .replace(
                ".csv",
                "",
            )
        )

        wellness_tables[
            feature
        ] = load_wide_table(
            subjective_root
            / "wellness"
            / filename,
            feature,
        )

    for filename in (
        TRAINING_LOAD_FILES
    ):

        feature = (
            filename
            .replace(
                ".csv",
                "",
            )
        )

        training_tables[
            feature
        ] = load_wide_table(
            subjective_root
            / "training-load"
            / filename,
            feature,
        )

    all_subjective_tables = (
        list(
            wellness_tables.values()
        )
        + list(
            training_tables.values()
        )
    )

    player_windows = (
        determine_player_windows(
            players=players,
            objective=objective,
            session_dates=(
                session_dates
            ),
            subjective_tables=(
                all_subjective_tables
            ),
        )
    )

    calendar = (
        build_master_calendar(
            players,
            player_windows,
        )
    )

    calendar = (
        add_core_flag(
            calendar,
            overlap,
        )
    )

    calendar = (
        add_presence_flag(
            calendar,
            objective,
            "objective_present",
        )
    )

    calendar = (
        add_presence_flag(
            calendar,
            session_dates,
            "session_present",
        )
    )

    for feature, table in (
        wellness_tables.items()
    ):

        calendar = (
            add_value_table(
                calendar,
                table,
                feature,
            )
        )

    for feature, table in (
        training_tables.items()
    ):

        calendar = (
            add_value_table(
                calendar,
                table,
                feature,
            )
        )

    calendar = (
        add_injury_reports(
            calendar,
            injuries,
        )
    )

    feature_names = (
        list(
            wellness_tables.keys()
        )
        + list(
            training_tables.keys()
        )
    )

    missing_dates = (
        build_missing_date_table(
            calendar,
            feature_names,
        )
    )

    player_summary = (
        build_player_missingness_summary(
            calendar,
            feature_names,
        )
    )

    feature_summary = (
        build_feature_missingness_summary(
            calendar,
            feature_names,
        )
    )

    streaks = (
        build_missing_streaks(
            calendar,
            feature_names,
        )
    )

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    calendar.to_csv(
        output_dir
        / "player_calendar_coverage.csv",
        index=False,
    )

    missing_dates.to_csv(
        output_dir
        / "calendar_missing_dates.csv",
        index=False,
    )

    player_summary.to_csv(
        output_dir
        / "calendar_missingness_by_player.csv",
        index=False,
    )

    feature_summary.to_csv(
        output_dir
        / "calendar_missingness_by_feature.csv",
        index=False,
    )

    streaks.to_csv(
        output_dir
        / "calendar_missing_streaks.csv",
        index=False,
    )

    player_windows.to_csv(
        output_dir
        / "player_observation_windows.csv",
        index=False,
    )

    summary_path = (
        output_dir
        / "calendar_missingness_summary.txt"
    )

    write_summary(
        calendar=calendar,
        feature_summary=(
            feature_summary
        ),
        player_summary=(
            player_summary
        ),
        missing_dates=(
            missing_dates
        ),
        streaks=streaks,
        output_path=(
            summary_path
        ),
    )


if __name__ == "__main__":
    main()