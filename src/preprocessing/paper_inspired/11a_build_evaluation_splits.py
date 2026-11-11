#!/usr/bin/env python3
"""SoccerMon paper-inspired Script 11A: leakage-aware evaluation assignments.

Input: Script 10 survival_targets.csv. Outputs are assignment metadata only.
No sensor features are loaded, imputed, scaled, or modified here.

Chronological protocol: select a globally shared date cutoff near the desired
train fraction of distinct prediction dates; training labels must end strictly
before the first test prediction date. Rows violating this are marked 'purged'.
LOPO protocol: one held-out player per fold, with all dates for that player in
that fold's test set. A held-out player's *past inputs* may be used at inference,
but fitted preprocessing and model parameters must not use that player's data.

These are distinct evaluation protocols, not a combined nested LOPO/time split.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path('data/modeling/paper_inspired/survival/survival_targets.csv')
DEFAULT_OUTPUT = Path('data/modeling/paper_inspired/splits')
DATE_COLS = ['prediction_date', 'history_start', 'history_end',
             'followup_start', 'planned_followup_end', 'observed_followup_end',
             'first_future_injury_date']
REQUIRED = ['player_name', 'prediction_date', 'history_start', 'history_end',
            'planned_followup_end', 'event', 'duration_days',
            'first_future_injury_date']


def load_targets(path: Path, horizon: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Survival targets not found: {path}\n'
                                'Run Script 10, or supply --input PATH.')
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(REQUIRED) - set(header))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    df = pd.read_csv(path, parse_dates=[c for c in DATE_COLS if c in header])
    if df.empty:
        raise ValueError('Survival target CSV is empty.')
    if df['player_name'].isna().any() or df['player_name'].astype(str).str.strip().eq('').any():
        raise ValueError('Missing or blank player_name.')
    if df[REQUIRED[:4] + ['planned_followup_end']].isna().any().any():
        raise ValueError('Missing player/date/history/follow-up identifiers.')
    if df.duplicated(['player_name', 'prediction_date']).any():
        examples = df.loc[df.duplicated(['player_name', 'prediction_date'], keep=False),
                          ['player_name', 'prediction_date']].head(5)
        raise ValueError('Duplicate (player, prediction_date) target keys:\n' + examples.to_string(index=False))
    df['event'] = pd.to_numeric(df['event'], errors='raise')
    df['duration_days'] = pd.to_numeric(df['duration_days'], errors='raise')
    if df['event'].isna().any() or (~df['event'].isin([0, 1])).any():
        raise ValueError('event must contain only 0 and 1.')
    if df['duration_days'].isna().any() or (~df['duration_days'].between(1, horizon)).any():
        raise ValueError(f'duration_days must be between 1 and {horizon}.')
    if (df['history_end'] >= df['prediction_date']).any():
        raise ValueError('History must end before prediction date.')
    if (df['history_start'] > df['history_end']).any():
        raise ValueError('History start is after history end.')
    if (df['planned_followup_end'] != df['prediction_date'] + pd.Timedelta(days=horizon)).any():
        raise ValueError('Planned follow-up ends do not match the configured horizon; '
                         'check --horizon-days against Script 10.')
    positive = df['event'].eq(1)
    if df.loc[positive, 'first_future_injury_date'].isna().any():
        raise ValueError('Positive events must have first_future_injury_date.')
    injury_offsets = (df.loc[positive, 'first_future_injury_date'] -
                      df.loc[positive, 'prediction_date']).dt.days
    if not (injury_offsets == df.loc[positive, 'duration_days']).all():
        raise ValueError('Positive event duration disagrees with injury date.')
    if not injury_offsets.between(1, horizon).all():
        raise ValueError('A positive injury falls outside the prediction horizon.')
    # Script 10 intentionally retains calendar-end censored rows (132 in our cohort),
    # so calendar_eligible need not be True for every survival target.
    if 'has_full_history_calendar' in df and not df['has_full_history_calendar'].fillna(False).astype(bool).all():
        raise ValueError('A survival target lacks its full historical calendar window.')
    df = df.sort_values(['prediction_date', 'player_name'], kind='stable').reset_index(drop=True)
    # Stable ID determined only by immutable player/date keys, not CSV row order.
    df['sample_id'] = (df['player_name'].astype(str) + '|' +
                       df['prediction_date'].dt.strftime('%Y-%m-%d'))
    return df


def summarize(part: pd.DataFrame, protocol: str, fold_id: str,
              role: str, cutoff: pd.Timestamp | None = None) -> dict:
    positive = part.loc[part['event'].eq(1)]
    unique_injuries = positive[['player_name', 'first_future_injury_date']].drop_duplicates()
    return {
        'protocol': protocol, 'fold_id': fold_id, 'split': role,
        'samples': len(part),
        'players': part['player_name'].nunique(),
        'positive_windows': len(positive),
        'distinct_injury_events': len(unique_injuries),
        'players_with_positive_windows': positive['player_name'].nunique(),
        'censored_windows': int(part['event'].eq(0).sum()),
        'first_prediction_date': part['prediction_date'].min(),
        'last_prediction_date': part['prediction_date'].max(),
        'cutoff_date': cutoff,
        'has_event_for_event_based_metric': len(unique_injuries) > 0,
        'warning': ('NO OBSERVED INJURY EVENT: event-based discrimination metrics '
                    'may be undefined' if unique_injuries.empty else
                    'VERY FEW DISTINCT INJURY EVENTS' if len(unique_injuries) < 3 else ''),
    }


def make_chronological(df: pd.DataFrame, fraction: float, horizon: int):
    distinct_dates = pd.DatetimeIndex(df['prediction_date'].drop_duplicates().sort_values())
    if len(distinct_dates) < 2:
        raise ValueError('At least two distinct prediction dates are needed for chronological split.')
    cut_index = min(max(int(np.floor(len(distinct_dates) * fraction)), 1), len(distinct_dates) - 1)
    cutoff = distinct_dates[cut_index]
    chrono = df[['sample_id', 'player_name', 'prediction_date',
                 'planned_followup_end', 'event', 'duration_days',
                 'first_future_injury_date']].copy()
    chrono['split'] = np.select(
        [chrono['prediction_date'].ge(cutoff),
         chrono['planned_followup_end'].ge(cutoff)],
        ['test', 'purged'], default='train'
    )
    chrono['protocol'] = 'chronological_80_20' if fraction == .8 else 'chronological'
    chrono['cutoff_date'] = cutoff
    chrono['purge_reason'] = np.where(chrono['split'].eq('purged'),
                                     'planned_outcome_horizon_reaches_test_start', '')
    diag = pd.DataFrame([
        summarize(df.loc[chrono['split'].eq(role)], 'chronological', 'chronological', role, cutoff)
        for role in ['train', 'purged', 'test']
    ])
    # Hard guarantee: no training target can require test-period dates.
    assert (chrono.loc[chrono['split'].eq('train'), 'planned_followup_end'] < cutoff).all()
    assert (chrono.loc[chrono['split'].eq('test'), 'prediction_date'] >= cutoff).all()
    assert len(chrono) == len(df)
    return chrono, diag, cutoff, len(distinct_dates)


def make_lopo(df: pd.DataFrame):
    folds, diagnostics = [], []
    base = df[['sample_id', 'player_name', 'prediction_date', 'event',
               'duration_days', 'first_future_injury_date']].copy()
    players = sorted(df['player_name'].unique().tolist())
    for number, heldout in enumerate(players, 1):
        fold_id = f'lopo_{number:02d}'
        fold = base.copy()
        fold['fold_id'] = fold_id
        fold['held_out_player'] = heldout
        fold['split'] = np.where(fold['player_name'].eq(heldout), 'test', 'train')
        assert (fold.loc[fold['split'].eq('test'), 'player_name'] == heldout).all()
        assert (fold.loc[fold['split'].eq('train'), 'player_name'] != heldout).all()
        assert fold['split'].eq('test').any() and fold['split'].eq('train').any()
        folds.append(fold)
        for role in ['train', 'test']:
            s = summarize(df.loc[fold['split'].eq(role)], 'lopo', fold_id, role)
            s['held_out_player'] = heldout
            diagnostics.append(s)
    lopo = pd.concat(folds, ignore_index=True)
    diag = pd.DataFrame(diagnostics)
    assert len(lopo) == len(df) * len(players)
    return lopo, diag, len(players)


def make_event_table(df: pd.DataFrame, chrono: pd.DataFrame, lopo: pd.DataFrame) -> pd.DataFrame:
    positives = df.loc[df['event'].eq(1), ['sample_id', 'player_name',
                        'prediction_date', 'first_future_injury_date']].copy()
    positives = positives.merge(chrono[['sample_id', 'split']].rename(columns={'split': 'chronological_split'}),
                                on='sample_id', how='left', validate='one_to_one')
    # Injury event identity is player+injury_date, NOT each daily positive window.
    events = (positives.groupby(['player_name', 'first_future_injury_date', 'chronological_split'],
                                as_index=False)
              .agg(positive_prediction_windows=('sample_id', 'size'),
                   earliest_prediction_date=('prediction_date', 'min'),
                   latest_prediction_date=('prediction_date', 'max')))
    heldout_map = lopo[['held_out_player', 'fold_id']].drop_duplicates()
    events = events.merge(heldout_map, left_on='player_name', right_on='held_out_player',
                          how='left', validate='many_to_one').drop(columns='held_out_player')
    events['lopo_role_in_own_fold'] = 'test'
    return events


def write_summary(path: Path, df: pd.DataFrame, chrono_diag: pd.DataFrame,
                  lopo_diag: pd.DataFrame, cutoff: pd.Timestamp, n_dates: int,
                  horizon: int, fraction: float) -> None:
    lines = [
        '=' * 78,
        'SoccerMon Script 11A - Evaluation Split Diagnostics',
        '=' * 78,
        f'Target samples: {len(df):,}',
        f'Players / LOPO folds: {df.player_name.nunique()}',
        f'Distinct prediction dates: {n_dates}',
        f'Prediction date range: {df.prediction_date.min():%Y-%m-%d} to {df.prediction_date.max():%Y-%m-%d}',
        f'Positive prediction windows: {int(df.event.sum())}',
        f'Distinct supporting injury events: {len(df.loc[df.event.eq(1), ["player_name", "first_future_injury_date"]].drop_duplicates())}',
        '',
        'CHRONOLOGICAL PROTOCOL',
        f'Requested training fraction of distinct dates: {fraction:.1%}',
        f'Shared cutoff / first test prediction date: {cutoff:%Y-%m-%d}',
        f'Forecast horizon used for purging: {horizon} days',
        'Training: prediction date earlier than cutoff AND planned follow-up end earlier than cutoff.',
        'Purged: earlier prediction date but planned follow-up overlaps test period.',
        'Test: prediction date on or after cutoff.',
    ]
    for row in chrono_diag.itertuples(index=False):
        lines.append(f'  {row.split}: {row.samples:,} samples, {row.positive_windows} positive windows, '
                     f'{row.distinct_injury_events} distinct injury events, {row.players} players')
    zero_test = lopo_diag[(lopo_diag['split'] == 'test') &
                          (lopo_diag['distinct_injury_events'] == 0)]
    lines += [
        '', 'LEAVE-ONE-PLAYER-OUT PROTOCOL',
        f'LOPO folds with zero observed test injury events: {len(zero_test)} / {df.player_name.nunique()}',
        'All available dates for the held-out player are test; all other players are training.',
        '', 'IMPORTANT LIMITATIONS',
        '- Positive daily windows may correspond to the same actual injury.',
        '- With zero observed test events, event-based metrics such as C-index may be undefined.',
        '- With extremely few injuries, estimates will be unstable even when mathematically defined.',
        '- Public injury-event surveillance completeness has not been verified.',
        '- Training/test history windows may overlap in calendar time; only future *label* overlap is purged.',
        '- No imputation, normalization, feature selection, or model training happens here.',
        '- Script 11B must fit preprocessing using training data only, separately for each protocol/fold.',
        '- LOPO and chronological evaluations are separate; not a nested combined evaluation.',
        '- A held-out player\'s past measurements can be used for inference, but not fitting.',
        '', 'Outputs: chronological_split.csv, lopo_folds.csv, split_diagnostics.csv,',
        '         positive_event_assignments.csv, split_summary.txt',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--train-fraction', type=float, default=.8,
                        help='Fraction of distinct prediction dates before chronological test (default: 0.8).')
    parser.add_argument('--horizon-days', type=int, default=7,
                        help='Must match Script 10 planned follow-up horizon (default: 7).')
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        parser.error('--train-fraction must be between 0 and 1.')
    if args.horizon_days < 1:
        parser.error('--horizon-days must be >=1.')
    df = load_targets(args.input, args.horizon_days)
    chrono, chrono_diag, cutoff, n_dates = make_chronological(df, args.train_fraction, args.horizon_days)
    lopo, lopo_diag, n_folds = make_lopo(df)
    events = make_event_table(df, chrono, lopo)
    chrono_diag['held_out_player'] = ''
    lopo_diag['cutoff_date'] = pd.NaT
    diagnostics = pd.concat([chrono_diag, lopo_diag], ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    chrono.to_csv(args.output / 'chronological_split.csv', index=False)
    lopo.to_csv(args.output / 'lopo_folds.csv', index=False)
    diagnostics.to_csv(args.output / 'split_diagnostics.csv', index=False)
    events.to_csv(args.output / 'positive_event_assignments.csv', index=False)
    write_summary(args.output / 'split_summary.txt', df, chrono_diag, lopo_diag,
                  cutoff, n_dates, args.horizon_days, args.train_fraction)
    print((args.output / 'split_summary.txt').read_text(encoding='utf-8'))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
