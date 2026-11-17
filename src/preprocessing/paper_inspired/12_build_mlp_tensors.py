#!/usr/bin/env python3
"""SoccerMon paper-inspired Script 12: assemble DeepHit MLP survival tensors.

Inputs: Script 10 survival_targets.csv, and each Script 11B fold's
sample_assignments.csv plus median/linear/bespoke scaled_player_days.csv and
preprocessing_parameters.json.

Outputs: one train/test tensor set per fold and method, survival durations,
event indicators, ordered sample metadata, feature-order metadata, and QA.

No fitting, feature selection, resampling, or relabeling occurs here. Only
21 player-days ending at t-1 contribute to a sample on prediction date t.
The saved feature tensors have shape [samples, 21, 40]. Script 13 can flatten
with X.reshape(len(X), -1) to obtain [samples, 840] for an MLP.

Methodological limitations carried forward from prior steps: rolling derived
features were not recomputed after imputation; the 'linear' method is causal
extrapolation and 'bespoke' is a reconstruction. Only two distinct injury
events contribute positive windows in the current cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

DEFAULT_PREPROCESSED = Path('data/modeling/paper_inspired/preprocessed')
DEFAULT_TARGETS = Path('data/modeling/paper_inspired/survival/survival_targets.csv')
DEFAULT_OUTPUT = Path('data/modeling/paper_inspired/tensors')
METHODS = ('median', 'linear', 'bespoke')
EXPECTED_FEATURES = 40
EXPECTED_HISTORY_DAYS = 21
FORECAST_DAYS = 7


def load_targets(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing Script 10 targets: {path}')
    t = pd.read_csv(path, low_memory=False)
    required = {'player_name', 'prediction_date', 'event', 'duration_days'}
    if required - set(t):
        raise ValueError(f'Targets missing: {sorted(required - set(t))}')
    t['prediction_date'] = pd.to_datetime(t['prediction_date'], errors='raise').dt.normalize()
    if t[['player_name', 'prediction_date']].isna().any().any():
        raise ValueError('Targets contain null sample keys.')
    if t.duplicated(['player_name', 'prediction_date']).any():
        raise ValueError('Targets contain duplicate player/prediction-date pairs.')
    t['sample_id'] = t['player_name'].astype(str) + '|' + t['prediction_date'].dt.strftime('%Y-%m-%d')
    if not t['event'].isin([0, 1]).all():
        raise ValueError('Target event must be binary 0/1.')
    dur = pd.to_numeric(t['duration_days'], errors='raise')
    if dur.isna().any() or not np.isfinite(dur).all() or not ((dur >= 1) & (dur <= FORECAST_DAYS)).all():
        raise ValueError(f'Target durations must be finite integers 1..{FORECAST_DAYS}.')
    if not np.equal(dur.to_numpy(), np.floor(dur.to_numpy())).all():
        raise ValueError('Target durations must be integers.')
    t['duration_days'] = dur.astype(np.int16)
    t['event'] = t['event'].astype(np.uint8)
    return t


def discover_folds(base: Path, protocol: str, fold_id: str | None):
    result = []
    if protocol in ('all', 'chronological'):
        folder = base / 'chronological'
        if not (folder / 'sample_assignments.csv').is_file():
            raise FileNotFoundError(f'Missing Script 11B chronological assignments: {folder}')
        result.append(('chronological', 'chronological', folder))
    if protocol in ('all', 'lopo'):
        root = base / 'lopo'
        folders = sorted(root.glob('lopo_*')) if root.exists() else []
        if fold_id:
            folders = [p for p in folders if p.name == fold_id]
        if not folders:
            raise FileNotFoundError(f'No LOPO fold folders found in {root} (fold_id={fold_id}).')
        for folder in folders:
            if not (folder / 'sample_assignments.csv').is_file():
                raise FileNotFoundError(f'Missing assignments for {folder.name}: {folder}')
            result.append(('lopo', folder.name, folder))
    return result


def load_assignments(path: Path, targets: pd.DataFrame, protocol: str, fold: str) -> pd.DataFrame:
    a = pd.read_csv(path, low_memory=False)
    required = {'sample_id', 'player_name', 'prediction_date', 'split'}
    if required - set(a):
        raise ValueError(f'{fold}: assignments missing {sorted(required - set(a))}')
    a['prediction_date'] = pd.to_datetime(a['prediction_date'], errors='raise').dt.normalize()
    expected_ids = a['player_name'].astype(str) + '|' + a['prediction_date'].dt.strftime('%Y-%m-%d')
    if a['sample_id'].isna().any() or not a['sample_id'].eq(expected_ids).all():
        raise ValueError(f'{fold}: invalid assignment sample IDs.')
    if a['sample_id'].duplicated().any():
        raise ValueError(f'{fold}: duplicated assignment sample IDs.')
    allowed = {'train', 'test', 'purged'} if protocol == 'chronological' else {'train', 'test'}
    if set(a['split']) != (set(a['split']) & allowed):
        raise ValueError(f'{fold}: unexpected assignment split values.')
    if not {'train', 'test'}.issubset(set(a['split'])):
        raise ValueError(f'{fold}: training and testing samples are both required.')
    # The fold must cover exactly the samples from Script 10, including purged rows.
    if set(a['sample_id']) != set(targets['sample_id']):
        raise ValueError(f'{fold}: assignments must match Script 10 target sample IDs exactly.')
    # Never trust the labels copied into an assignment CSV in place of Script 10.
    for name in ('event', 'duration_days'):
        if name in a.columns:
            looked_up = a[['sample_id', name]].merge(
                targets[['sample_id', name]].rename(columns={name: '_original'}),
                on='sample_id', how='left', validate='one_to_one'
            )
            if not (looked_up[name] == looked_up['_original']).all():
                raise ValueError(f'{fold}: assignment {name} differs from original Script 10 targets.')
    a = a[['sample_id', 'player_name', 'prediction_date', 'split']].merge(
        targets.drop(columns=['player_name', 'prediction_date']),
        on='sample_id', how='left', validate='one_to_one'
    )
    # Keep player and date exactly as supplied by the authoritative Script 11B assignments.
    if a[['event', 'duration_days']].isna().any().any():
        raise ValueError(f'{fold}: assignments lack matching survival targets.')
    if protocol == 'lopo':
        if 'held_out_player' not in pd.read_csv(path, nrows=0).columns:
            raise ValueError(f'{fold}: missing held_out_player in source assignments.')
        info = pd.read_csv(path, usecols=['held_out_player', 'split', 'player_name'])
        held = info['held_out_player'].dropna().unique()
        if len(held) != 1:
            raise ValueError(f'{fold}: expected exactly one held-out player.')
        if not info.loc[info['split'] == 'test', 'player_name'].eq(held[0]).all():
            raise ValueError(f'{fold}: test contains players besides held-out player.')
        if info.loc[info['split'] == 'train', 'player_name'].eq(held[0]).any():
            raise ValueError(f'{fold}: held-out player included in training.')
        if info.loc[info['split'] == 'train', 'player_name'].nunique() != 21:
            print(f'WARNING: {fold}: expected 21 training players; found '
                  f"{info.loc[info['split'] == 'train', 'player_name'].nunique()}.", flush=True)
    return a.sort_values(['prediction_date', 'player_name'], kind='stable').reset_index(drop=True)


def make_history_lookup(assignments: pd.DataFrame, history: int) -> dict[str, np.ndarray]:
    """Build literal history keys; method-specific feature table resolves them to rows."""
    result = {}
    for partition in ('train', 'test'):
        samples = assignments.loc[assignments['split'] == partition]
        players = np.repeat(samples['player_name'].astype(str).to_numpy(), history)
        dates = np.repeat(samples['prediction_date'].to_numpy(dtype='datetime64[ns]'), history)
        # Oldest history row first: t-history, ..., t-1.
        offsets = np.tile(np.arange(history, 0, -1, dtype=np.int64), len(samples))
        dates = dates - offsets.astype('timedelta64[D]')
        result[partition] = pd.MultiIndex.from_arrays([players, dates], names=['player_name', 'date'])
    return result


def read_scaled(path: Path, param_path: Path, expected_method: str, expected_protocol: str,
                expected_fold: str, expected_heldout: str | None):
    if not path.is_file() or not param_path.is_file():
        raise FileNotFoundError(f'Missing Script 11B output: {path} or {param_path}')
    params = json.loads(param_path.read_text(encoding='utf-8'))
    names = params.get('feature_names')
    if not isinstance(names, list) or len(names) != EXPECTED_FEATURES or len(set(names)) != EXPECTED_FEATURES:
        raise ValueError(f'{path}: feature_names must contain {EXPECTED_FEATURES} unique features.')
    if params.get('method') != expected_method or params.get('protocol') != expected_protocol:
        raise ValueError(f'{path}: mismatched preprocessing method/protocol metadata.')
    if params.get('fold') != expected_fold:
        raise ValueError(f'{path}: preprocessing fold differs from assignments.')
    if params.get('held_out_player') != expected_heldout:
        raise ValueError(f'{path}: preprocessing held-out player differs from assignments.')
    if expected_heldout is not None and expected_heldout in params.get('training_players', []):
        raise ValueError(f'{path}: held-out player listed as a fitted preprocessing participant.')
    for name in ('imputer_median', 'scaler_mean', 'scaler_scale'):
        if set(params.get(name, {})) != set(names):
            raise ValueError(f'{path}: {name} parameter keys do not match feature order.')
    frame = pd.read_csv(path, usecols=['player_name', 'date', *names], low_memory=False)
    frame['date'] = pd.to_datetime(frame['date'], errors='raise').dt.normalize()
    if frame[['player_name', 'date']].isna().any().any():
        raise ValueError(f'{path}: feature key missing.')
    if frame.duplicated(['player_name', 'date']).any():
        raise ValueError(f'{path}: duplicate player/day.')
    # Explicit order avoids changes from CSV column ordering or alphabetical sorting.
    frame = frame[['player_name', 'date', *names]]
    matrix = frame[names].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError(f'{path}: nonfinite scaled features.')
    return frame, matrix, names, params


def write_partition(folder: Path, partition: str, samples: pd.DataFrame,
                    history_index: pd.MultiIndex, frame: pd.DataFrame,
                    scaled: np.ndarray, history: int):
    feature_index = pd.MultiIndex.from_frame(frame[['player_name', 'date']])
    idx = feature_index.get_indexer(history_index)
    missing = idx < 0
    if np.any(missing):
        bad = history_index[np.flatnonzero(missing)[0]]
        raise ValueError(f'{folder}: missing historical feature row for {partition}: {bad}.')
    x = scaled[idx].reshape(len(samples), history, scaled.shape[1])
    # The history is ordered old -> recent and is STRICTLY earlier than t.
    if len(samples) and x.shape != (len(samples), history, EXPECTED_FEATURES):
        raise AssertionError('Incorrect tensor dimensions.')
    if not np.isfinite(x).all():
        raise AssertionError('Nonfinite values in constructed model tensor.')
    durations = samples['duration_days'].to_numpy(dtype=np.int16)
    events = samples['event'].to_numpy(dtype=np.uint8)
    if not set(np.unique(events)).issubset({0, 1}):
        raise AssertionError('Invalid event labels.')
    if not np.all((durations >= 1) & (durations <= FORECAST_DAYS)):
        raise AssertionError('Invalid survival durations.')
    np.save(folder / f'X_{partition}.npy', x, allow_pickle=False)
    np.save(folder / f'duration_{partition}.npy', durations, allow_pickle=False)
    np.save(folder / f'event_{partition}.npy', events, allow_pickle=False)
    metadata_columns = ['sample_id', 'player_name', 'prediction_date', 'event', 'duration_days']
    for optional in ['first_future_injury_date', 'censor_reason', 'observed_followup_end',
                     'has_full_followup_calendar', 'calendar_eligible']:
        if optional in samples:
            metadata_columns.append(optional)
    metadata = samples[metadata_columns].copy()
    metadata.insert(3, 'history_start', samples['prediction_date'] - pd.Timedelta(days=history))
    metadata.insert(4, 'history_end', samples['prediction_date'] - pd.Timedelta(days=1))
    metadata.to_csv(folder / f'sample_index_{partition}.csv', index=False)
    if len(samples) != len(durations) or len(samples) != len(events):
        raise AssertionError('Misaligned labels and tensors.')
    return {
        'samples': len(samples),
        'observed_event_windows': int(events.sum()),
        'players': int(samples['player_name'].nunique()),
        'unique_supporting_events': int(samples.loc[samples['event'] == 1, ['player_name', 'first_future_injury_date']].drop_duplicates().shape[0])
            if 'first_future_injury_date' in samples else None,
        'short_censored_samples': int(((events == 0) & (durations < FORECAST_DAYS)).sum()),
    }


def execute_method(protocol: str, fold: str, method: str, source_folder: Path,
                   output_base: Path, assignments: pd.DataFrame, history_indexes: dict,
                   history: int, heldout: str | None):
    src = source_folder / method
    frame, scaled, feature_names, params = read_scaled(
        src / 'scaled_player_days.csv', src / 'preprocessing_parameters.json',
        method, protocol, fold, heldout
    )
    # CSV and JSON row orders are verified; no re-fitting, scaling, or fill here.
    out = (output_base / 'chronological' / method if protocol == 'chronological'
           else output_base / 'lopo' / fold / method)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for part in ('train', 'test'):
        s = assignments.loc[assignments['split'] == part].reset_index(drop=True)
        stats[part] = write_partition(out, part, s, history_indexes[part], frame, scaled, history)
    param_sha = hashlib.sha256((src / 'preprocessing_parameters.json').read_bytes()).hexdigest()
    payload = {
        'script': '12_build_mlp_tensors.py',
        'protocol': protocol, 'fold': fold, 'method': method,
        'held_out_player': heldout,
        'history_days': history,
        'input_dates': f't-{history} through t-1 (no prediction-day features)',
        'forecast_days': FORECAST_DAYS,
        'tensor_shape_per_sample': [history, len(feature_names)],
        'flattened_mlp_features': history * len(feature_names),
        'tensor_order': 'oldest history date first; feature_names order within each day',
        'feature_names': feature_names,
        'dtype': 'float32',
        'preprocessing_parameters_sha256': param_sha,
        'source_scaled_features': str(src / 'scaled_player_days.csv'),
        'survival_label_semantics': 'event=1 injury observed on duration day; event=0 right-censored at duration day',
        'partitions': stats,
        'limitations': [
            'No oversampling or synthetic injury events.',
            'Derived rolling features were imputed independently in Script 11B.',
            'Current labels represent only two distinct observed injury events.',
            'Eligibility and surveillance caveats from Script 10 are retained, not changed.'
        ]
    }
    (out / 'tensor_metadata.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return {
        'protocol': protocol, 'fold': fold, 'method': method,
        'held_out_player': heldout or '',
        'train_samples': stats['train']['samples'], 'test_samples': stats['test']['samples'],
        'purged_samples': int(assignments.split.eq('purged').sum()),
        'train_positive_windows': stats['train']['observed_event_windows'],
        'test_positive_windows': stats['test']['observed_event_windows'],
        'train_distinct_events': stats['train']['unique_supporting_events'],
        'test_distinct_events': stats['test']['unique_supporting_events'],
        'train_short_censored': stats['train']['short_censored_samples'],
        'test_short_censored': stats['test']['short_censored_samples'],
        'features': len(feature_names), 'history_days': history,
        'input_dimensions': history * len(feature_names),
        'nonfinite_tensor_values': 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--targets', type=Path, default=DEFAULT_TARGETS)
    parser.add_argument('--preprocessed', type=Path, default=DEFAULT_PREPROCESSED)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--protocol', choices=['all', 'chronological', 'lopo'], default='all')
    parser.add_argument('--fold-id', help='Specific LOPO fold, e.g. lopo_01 (only with --protocol lopo).')
    parser.add_argument('--history-days', type=int, default=EXPECTED_HISTORY_DAYS)
    args = parser.parse_args()
    if args.fold_id and args.protocol != 'lopo':
        parser.error('--fold-id requires --protocol lopo.')
    if args.history_days != EXPECTED_HISTORY_DAYS:
        parser.error('The benchmark requires 21-day histories; change the benchmark spec before using other lengths.')
    targets = load_targets(args.targets)
    folds = discover_folds(args.preprocessed, args.protocol, args.fold_id)
    # When running one fold at a time, keep previous records in diagnostics.
    records = []
    for protocol, fold, source_folder in folds:
        assignments = load_assignments(source_folder / 'sample_assignments.csv', targets, protocol, fold)
        heldout = None
        if protocol == 'lopo':
            raw_assignments = pd.read_csv(source_folder / 'sample_assignments.csv', usecols=['held_out_player'])
            heldout = str(raw_assignments['held_out_player'].dropna().unique()[0])
        histories = make_history_lookup(assignments, args.history_days)
        for method in METHODS:
            rec = execute_method(protocol, fold, method, source_folder, args.output,
                                 assignments, histories, args.history_days, heldout)
            records.append(rec)
            print(f"{protocol}/{fold}/{method}: train={rec['train_samples']:,}, "
                  f"test={rec['test_samples']:,}; positive={rec['train_positive_windows']}/"
                  f"{rec['test_positive_windows']}; X=(N,21,40)", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame.from_records(records)
    diagnostics_path = args.output / 'tensor_diagnostics.csv'
    if diagnostics_path.is_file():
        previous = pd.read_csv(diagnostics_path)
        report = (pd.concat([previous, report], ignore_index=True)
                  .drop_duplicates(['protocol', 'fold', 'method'], keep='last')
                  .sort_values(['protocol', 'fold', 'method']))
    report.to_csv(diagnostics_path, index=False)
    lines = [
        'SoccerMon Script 12 - DeepHit MLP Tensor Construction',
        f'Input targets: {len(targets):,}; players: {targets.player_name.nunique()}',
        f'Executed combinations in this invocation: {len(records)}',
        f'Total combinations present in diagnostics: {len(report)}',
        f'Historical window: t-{args.history_days}..t-1; forecast t+1..t+{FORECAST_DAYS}',
        f'Features: {EXPECTED_FEATURES}; tensor dimensions: (samples, 21, 40); MLP flattened: 840',
        'Files: X_train/test.npy, duration_train/test.npy, event_train/test.npy,',
        '       sample_index_train/test.csv, tensor_metadata.json',
        'Training/test split and survival labels preserved; purged rows excluded.',
        'No imputation or normalization refitted; uses Script 11B scaled inputs.',
        'No NaNs, infs or incomplete histories allowed.',
        'IMPORTANT: Event counts are positive windows, not independent injuries.',
        'IMPORTANT: Only two distinct supporting injuries in current cohort.',
        'IMPORTANT: The DeepHit MLP architecture and loss are implemented later in Script 13.',
    ]
    (args.output / 'tensor_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, AssertionError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
