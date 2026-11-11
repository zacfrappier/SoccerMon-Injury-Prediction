#!/usr/bin/env python3
"""SoccerMon paper-inspired Script 11B: fold-specific, causal feature preprocessing.

Inputs: Script 08 unimputed player-day features and Script 11A split assignments.
Produces separate chronological/LOPO, median/causal-linear/bespoke feature tables,
feature-missingness masks, training-only scaler/imputer parameters and diagnostics.

IMPORTANT: The original paper's exact bespoke procedure is not publicly reproduced
here. 'linear' means past-only linear extrapolation (NOT bidirectional interpolation).
Derived rolling features are imputed as existing columns, not recomputed from imputed
underlying daily values. These are explicitly documented reconstruction choices.
The training-corpus median/scaler use permitted training-history player-days; on
training dates, globally fitted medians can use later *training-period* rows, as is
normal for fitted preprocessing. No test-period/held-out-player rows fit parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

DEFAULT_FEATURES = Path('data/modeling/paper_inspired/features/paper_features_unimputed.csv')
DEFAULT_SPLITS = Path('data/modeling/paper_inspired/splits')
DEFAULT_OUTPUT = Path('data/modeling/paper_inspired/preprocessed')

FEATURES = [
    'fatigue', 'mood', 'readiness', 'sleep_duration', 'sleep_quality', 'soreness', 'stress',
    'rpe_mean', 'rpe_max', 'duration_total', 'duration_mean', 'srpe_total', 'srpe_mean', 'srpe_max',
    'duration_obj', 'Speed_km_h_mean', 'Speed_km_h_max', 'Speed_km_h_std',
    'sp_lir_p', 'sp_lir_t', 'sp_lir_d', 'sp_mir_p', 'sp_mir_t', 'sp_mir_d',
    'sp_hir_p', 'sp_hir_t', 'sp_hir_d', 'sp_spr_p', 'sp_spr_t', 'sp_spr_d',
    'Distance', 'distance_per_min', 'daily_load', 'atl_7d', 'weekly_load_7d',
    'monotony_7d', 'strain_7d', 'ctl28', 'ctl42', 'acwr_7_28',
]
# Percentage/proportion feature bounds. No arbitrary clinical bounds applied.
PROPORTIONS = {'sp_lir_p', 'sp_mir_p', 'sp_hir_p', 'sp_spr_p'}
NONNEGATIVE = set(FEATURES) - set()  # All 40 current candidates represent nonnegative quantities.


def load_data(features_path: Path, splits_dir: Path):
    for path in [features_path, splits_dir / 'chronological_split.csv', splits_dir / 'lopo_folds.csv']:
        if not path.is_file():
            raise FileNotFoundError(f'Missing input {path}. Run Scripts 08 and 11A first.')
    f = pd.read_csv(features_path, low_memory=False)
    required = {'player_name', 'date', *FEATURES}
    missing = required - set(f.columns)
    if missing:
        raise ValueError(f'Feature CSV missing columns: {sorted(missing)}')
    f['date'] = pd.to_datetime(f['date'], errors='raise').dt.normalize()
    if f[['player_name', 'date']].isna().any().any() or f.duplicated(['player_name', 'date']).any():
        raise ValueError('Feature file needs unique, non-null (player_name, date) keys.')
    for name in FEATURES:
        f[name] = pd.to_numeric(f[name], errors='raise')
        if np.isinf(f[name].to_numpy(float)).any():
            raise ValueError(f'Infinity in original feature {name}; inspect source before preprocessing.')
    f = f[['player_name', 'date', *FEATURES]].sort_values(['player_name', 'date']).reset_index(drop=True)
    chrono = pd.read_csv(splits_dir / 'chronological_split.csv', parse_dates=['prediction_date', 'cutoff_date'])
    lopo = pd.read_csv(splits_dir / 'lopo_folds.csv', parse_dates=['prediction_date'])
    for n, s in [('chronological', chrono), ('lopo', lopo)]:
        required_s = {'player_name', 'prediction_date', 'sample_id', 'split'}
        if required_s - set(s):
            raise ValueError(f'{n} missing {sorted(required_s - set(s))}')
        if not s['sample_id'].eq(s['player_name'].astype(str) + '|' + s['prediction_date'].dt.strftime('%Y-%m-%d')).all():
            raise ValueError(f'{n} contains inconsistent sample IDs.')
    if set(chrono['split']) - {'train', 'test', 'purged'}:
        raise ValueError('Unrecognized chronological split values.')
    if set(lopo['split']) - {'train', 'test'}:
        raise ValueError('Unrecognized LOPO split values.')
    if chrono['sample_id'].duplicated().any():
        raise ValueError('Chronological sample IDs must be unique.')
    if lopo.duplicated(['fold_id', 'sample_id']).any():
        raise ValueError('Duplicate LOPO sample/fold rows.')
    if not set(chrono['player_name']).issubset(set(f['player_name'])):
        raise ValueError('Chronological targets have players not found in features.')
    if not set(lopo['player_name']).issubset(set(f['player_name'])):
        raise ValueError('LOPO targets have players not found in features.')
    # Every sample needs exactly the 21 consecutive calendar days ending at t-1.
    # This validates the source and avoids silently constructing incomplete tensors in Script 12.
    key_set = set(zip(f['player_name'], f['date']))
    checks = chrono[['player_name', 'prediction_date']].drop_duplicates()
    for r in checks.itertuples(index=False):
        for lag in range(1, 22):
            if (r.player_name, r.prediction_date - pd.Timedelta(days=lag)) not in key_set:
                raise ValueError(f'Missing 21-day history row for {r.player_name} prediction {r.prediction_date:%Y-%m-%d}, lag {lag}')
    return f, chrono, lopo


def training_history_mask(f: pd.DataFrame, targets: pd.DataFrame, history_days: int = 21) -> np.ndarray:
    """Only dates that appear in training samples' historical input windows fit parameters."""
    training = targets.loc[targets['split'].eq('train'), ['player_name', 'prediction_date']].drop_duplicates()
    if training.empty:
        raise ValueError('No train samples.')
    pieces = []
    for lag in range(1, history_days + 1):
        tmp = training.copy()
        tmp['date'] = tmp['prediction_date'] - pd.Timedelta(days=lag)
        pieces.append(tmp[['player_name', 'date']])
    keys = pd.concat(pieces, ignore_index=True).drop_duplicates()
    idx = pd.MultiIndex.from_frame(f[['player_name', 'date']])
    allowed = idx.isin(pd.MultiIndex.from_frame(keys))
    if not allowed.any():
        raise ValueError('No training feature-history days matched.')
    return np.asarray(allowed, dtype=bool)


def fit_training_stats(f: pd.DataFrame, fit_mask: np.ndarray):
    train = f.loc[fit_mask, FEATURES]
    if train.empty:
        raise ValueError('Training feature-day set is empty.')
    meds = train.median(axis=0, skipna=True)
    missing_all = meds[meds.isna()].index.tolist()
    if missing_all:
        raise ValueError(f'No observed training values for features: {missing_all}; cannot fit imputer.')
    obs_min = train.min(axis=0, skipna=True)
    obs_max = train.max(axis=0, skipna=True)
    return meds, obs_min, obs_max


def past_only_linear(f: pd.DataFrame, medians: pd.Series, lo: pd.Series,
                     hi: pd.Series, max_gap_days: int) -> tuple[pd.DataFrame, dict]:
    """Per-player strictly past-only linear extrapolation, capped to training min/max.

    If last two observed values are within max_gap_days of the target day,
    project from their calendar-day slope. Otherwise use last observation within
    the gap; after that use train median. A missing day is NEVER an observed anchor.
    """
    result = f[FEATURES].copy()
    counts = {'estimated': 0, 'last_observation': 0, 'fallback': 0}
    for feature in FEATURES:
        for _, g in f.groupby('player_name', sort=False):
            indices = g.index.to_numpy()
            dates = g['date'].to_numpy(dtype='datetime64[D]').astype('int64')
            vals = g[feature].to_numpy(float)
            last = prev = None
            for pos, value in enumerate(vals):
                if np.isfinite(value):
                    prev, last = last, pos
                    continue
                estimate = None
                if last is not None and dates[pos] - dates[last] <= max_gap_days:
                    if prev is not None and dates[last] > dates[prev]:
                        slope = (vals[last] - vals[prev]) / (dates[last] - dates[prev])
                        estimate = vals[last] + slope * (dates[pos] - dates[last])
                        counts['estimated'] += 1
                    else:
                        estimate = vals[last]
                        counts['last_observation'] += 1
                else:
                    estimate = float(medians[feature])
                    counts['fallback'] += 1
                estimate = float(np.clip(estimate, lo[feature], hi[feature]))
                if feature in NONNEGATIVE:
                    estimate = max(0., estimate)
                if feature in PROPORTIONS:
                    estimate = min(1., estimate)
                result.at[indices[pos], feature] = estimate
    return result, counts


def teammate_relative(f: pd.DataFrame, train_players: set[str], medians: pd.Series,
                      lo: pd.Series, hi: pd.Series, lookback: int = 14,
                      min_pairs: int = 3) -> tuple[pd.DataFrame, dict]:
    """Causal reconstructed teammate-relative imputer.

    For missing feature on player-day d, use ONLY observed peer values on day d
    among TRAINING players, and paired observed player/peer values on days
    [d-lookback, d-1]. Scale today's peer median by the median of (player/peer)
    historical ratios. If no ratios or today's peer median, fall back to past
    player observation (within lookback), then training median. Heldout player is
    excluded from peer aggregate and from fitted population statistics. Their
    prior observations may be used for test-time causal per-player adaptation.
    """
    result = f[FEATURES].copy()
    counts = {'teammate_estimated': 0, 'personal_last': 0, 'fallback': 0}
    dates = f['date'].to_numpy(dtype='datetime64[D]').astype('int64')
    source_players = f['player_name'].astype(str).to_numpy()
    for feature in FEATURES:
        # Peer aggregate per calendar date; exclude target player if they are a trainer.
        peers_by_day: dict[int, dict[str, float]] = {}
        tmp = f.loc[f['player_name'].isin(train_players), ['date', 'player_name', feature]]
        for day, g in tmp.groupby('date', sort=False):
            d = int(day.to_datetime64().astype('datetime64[D]').astype('int64'))
            peers_by_day[d] = dict(zip(g['player_name'].astype(str), g[feature].astype(float)))
        def peer_median(day: int, player: str) -> float:
            v = [x for p, x in peers_by_day.get(day, {}).items() if p != player and np.isfinite(x)]
            return float(np.median(v)) if v else np.nan
        for player, g in f.groupby('player_name', sort=False):
            indices = g.index.to_numpy()
            day_values = dates[indices]
            source = g[feature].to_numpy(float)
            last_observed = None
            history: list[tuple[int, float, float]] = []
            for j, v in enumerate(source):
                d = int(day_values[j])
                peer = peer_median(d, str(player))
                if np.isfinite(v):
                    if np.isfinite(peer) and peer > 0:
                        history.append((d, float(v), peer))
                    last_observed = (d, float(v))
                    continue
                pairs = [(pv / pp) for pd_day, pv, pp in history
                         if 0 < d - pd_day <= lookback and pp > 0 and np.isfinite(pv)]
                if len(pairs) >= min_pairs and np.isfinite(peer):
                    value = float(np.median(pairs)) * peer
                    counts['teammate_estimated'] += 1
                elif last_observed is not None and 0 < d - last_observed[0] <= lookback:
                    value = last_observed[1]
                    counts['personal_last'] += 1
                else:
                    value = float(medians[feature])
                    counts['fallback'] += 1
                value = float(np.clip(value, lo[feature], hi[feature]))
                if feature in NONNEGATIVE:
                    value = max(0., value)
                if feature in PROPORTIONS:
                    value = min(1., value)
                result.at[indices[j], feature] = value
    return result, counts


def handle_protocol(f: pd.DataFrame, assignments: pd.DataFrame,
                    output: Path, protocol: str, fold: str, heldout: str | None,
                    linear_gap: int, teammate_lookback: int, min_pairs: int):
    assignment = assignments.copy()
    fit_mask = training_history_mask(f, assignment)
    training_players = set(assignment.loc[assignment.split.eq('train'), 'player_name'].astype(str))
    if heldout is not None:
        if heldout in training_players or not assignment.loc[assignment.split.eq('test'), 'player_name'].eq(heldout).all():
            raise AssertionError('LOPO heldout player leaked into training assignments.')
        if f.loc[fit_mask, 'player_name'].eq(heldout).any():
            raise AssertionError('Held-out player leaked into train-fit feature days.')
    else:
        cutoff = pd.Timestamp(assignment['cutoff_date'].dropna().iloc[0]).normalize()
        if f.loc[fit_mask, 'date'].ge(cutoff).any():
            raise AssertionError('Test-period dates leaked into chronological train-fit feature days.')
    medians, low, high = fit_training_stats(f, fit_mask)
    original = f[FEATURES]
    missing = original.isna()
    output.mkdir(parents=True, exist_ok=True)
    # Save shared missingness mask once per fold, never use as model input by default.
    mask_frame = f[['player_name', 'date']].copy()
    for c in FEATURES:
        mask_frame[c] = missing[c].astype('uint8')
    mask_frame.to_csv(output / 'original_missingness_mask.csv', index=False)
    # Give Script 12 the exact sample assignment list for this fold.
    assignment.to_csv(output / 'sample_assignments.csv', index=False)
    diagnostics = []
    for method in ['median', 'linear', 'bespoke']:
        if method == 'median':
            processed = original.fillna(medians)
            fallback = {'fallback': int(missing.to_numpy().sum())}
        elif method == 'linear':
            processed, fallback = past_only_linear(f, medians, low, high, linear_gap)
        else:
            processed, fallback = teammate_relative(f, training_players, medians, low, high,
                                                   teammate_lookback, min_pairs)
        if processed.isna().any().any() or not np.isfinite(processed.to_numpy(float)).all():
            raise ValueError(f'{protocol}/{fold}/{method} contains nonfinite model features.')
        # Preserve observed values precisely, even if an observed value falls outside
        # the range of the permitted TRAINING data.
        if not np.allclose(processed.where(~missing).fillna(0).to_numpy(),
                           original.where(~missing).fillna(0).to_numpy(), rtol=0, atol=0):
            raise AssertionError('Imputation changed observed data.')
        if not pd.DataFrame({c: processed[c] for c in PROPORTIONS}).ge(0).all().all():
            raise ValueError('Negative proportions in features.')
        # Fit AFTER imputation; fit_mask excludes all held-out/test-period samples.
        train_mat = processed.loc[fit_mask, FEATURES].to_numpy(float)
        mean = train_mat.mean(axis=0)
        scale = train_mat.std(axis=0, ddof=0)
        constant = scale <= 1e-12
        scale[constant] = 1.0
        scaled = (processed.to_numpy(float) - mean) / scale
        if not np.isfinite(scaled).all():
            raise ValueError('Scaled outputs contain invalid values.')
        target_dir = output / method
        target_dir.mkdir(parents=True, exist_ok=True)
        transformed = f[['player_name', 'date']].copy()
        transformed[FEATURES] = scaled
        transformed.to_csv(target_dir / 'scaled_player_days.csv', index=False)
        params = {
            'method': method, 'protocol': protocol, 'fold': fold,
            'held_out_player': heldout, 'feature_count': len(FEATURES),
            'feature_names': FEATURES, 'training_feature_days': int(fit_mask.sum()),
            'training_players': sorted(training_players),
            'imputer_median': {c: float(medians[c]) for c in FEATURES},
            'training_observed_min': {c: float(low[c]) for c in FEATURES},
            'training_observed_max': {c: float(high[c]) for c in FEATURES},
            'scaler_mean': dict(zip(FEATURES, mean.tolist())),
            'scaler_scale': dict(zip(FEATURES, scale.tolist())),
            'constant_training_features': [FEATURES[i] for i in np.flatnonzero(constant)],
            'imputation_actions': fallback,
            'method_notes': (
                'Training-history medians' if method == 'median' else
                'Past-only linear extrapolation, max gap, clamped to training range; fallback train median' if method == 'linear' else
                'Past 14-day player/peer ratios; same-day observed training teammates; personal previous, then training median'
            ),
            'limitations': 'Rolling derived features were imputed independently; not recomputed after imputing daily load.',
        }
        (target_dir / 'preprocessing_parameters.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
        diagnostics.append({
            'protocol': protocol, 'fold': fold, 'method': method,
            'held_out_player': heldout or '',
            'train_samples': int(assignment.split.eq('train').sum()),
            'test_samples': int(assignment.split.eq('test').sum()),
            'purged_samples': int(assignment.split.eq('purged').sum()),
            'training_fit_player_days': int(fit_mask.sum()),
            'original_missing_feature_cells': int(missing.to_numpy().sum()),
            'final_missing_feature_cells': int(processed.isna().to_numpy().sum()),
            **fallback,
        })
    print(f'  {protocol}/{fold}: {len(assignment):,} assignments; {int(fit_mask.sum()):,} train-fit days; 3 strategies', flush=True)
    return diagnostics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--features', type=Path, default=DEFAULT_FEATURES)
    p.add_argument('--splits', type=Path, default=DEFAULT_SPLITS)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--protocol', choices=['all', 'chronological', 'lopo'], default='all')
    p.add_argument('--fold-id', default=None, help='For --protocol lopo, process one fold (e.g. lopo_01).')
    p.add_argument('--linear-max-gap-days', type=int, default=14)
    p.add_argument('--teammate-lookback-days', type=int, default=14)
    p.add_argument('--min-paired-observations', type=int, default=3)
    args = p.parse_args()
    if min(args.linear_max_gap_days, args.teammate_lookback_days, args.min_paired_observations) < 1:
        p.error('Lookback, gap, min paired observations must be positive.')
    if args.fold_id and args.protocol != 'lopo':
        p.error('--fold-id requires --protocol lopo')
    f, chrono, lopo = load_data(args.features, args.splits)
    summaries = []
    print(f'SoccerMon Script 11B: {len(f):,} player-days, {f.player_name.nunique()} players, {len(FEATURES)} features', flush=True)
    if args.protocol in ('all', 'chronological'):
        summaries += handle_protocol(f, chrono, args.output / 'chronological',
                                     'chronological', 'chronological', None,
                                     args.linear_max_gap_days, args.teammate_lookback_days,
                                     args.min_paired_observations)
    if args.protocol in ('all', 'lopo'):
        for fold_id in sorted(lopo.fold_id.unique()):
            if args.fold_id and fold_id != args.fold_id:
                continue
            subset = lopo.loc[lopo.fold_id.eq(fold_id)].copy()
            held = subset.held_out_player.dropna().unique()
            if len(held) != 1:
                raise ValueError(f'Expected one held-out player per fold: {fold_id}')
            summaries += handle_protocol(f, subset, args.output / 'lopo' / fold_id,
                                         'lopo', fold_id, str(held[0]),
                                         args.linear_max_gap_days, args.teammate_lookback_days,
                                         args.min_paired_observations)
    if not summaries:
        raise ValueError('No requested folds found.')
    summary = pd.DataFrame(summaries)
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep previous summaries when running one fold so no misleading full coverage claim.
    path = args.output / 'preprocessing_diagnostics.csv'
    if path.exists():
        prev = pd.read_csv(path)
        summary = (pd.concat([prev, summary], ignore_index=True)
                   .drop_duplicates(['protocol', 'fold', 'method'], keep='last')
                   .sort_values(['protocol', 'fold', 'method']))
    summary.to_csv(path, index=False)
    lines = [
        'SoccerMon 11B: Fold-Specific Imputation and Normalization',
        f'Input rows: {len(f):,}, players: {f.player_name.nunique()}, features: {len(FEATURES)}',
        f'Protocol in this execution: {args.protocol}',
        f'Fold-strategy combinations executed: {len(summaries)}',
        'Strategies: train-median; causal past-only linear; reconstructed teammate-relative.',
        'All 40 inputs scaled using training-only fitting; original missingness saved separately.',
        'Training fit dates: union of 21-day histories belonging to train samples.',
        'Chronological future test dates and LOPO held-out players excluded from fitting.',
        'All source features preserved; imputation replaces only missing values.',
        'Missingness mask is not part of the 40-feature model tensor by default.',
        'LIMITATION: rolling features were NOT recomputed after imputing daily values.',
        'LIMITATION: causal linear differs from bidirectional interpolation in Script 09.',
        'LIMITATION: bespoke is a documented reconstruction, not confirmed paper source code.',
        'LIMITATION: teammate values for a player-day assume measurements available by day end.',
        'LIMITATION: model evaluation with only two injuries remains statistically unstable.',
    ]
    (args.output / 'preprocessing_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'Diagnostics saved to {path}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
