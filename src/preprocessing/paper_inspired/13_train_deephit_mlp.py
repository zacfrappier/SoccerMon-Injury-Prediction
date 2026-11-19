#!/usr/bin/env python3
"""SoccerMon paper-inspired Script 13: train a single-event discrete-time DeepHit-style MLP.

Consumes Script 12 tensors, labels, indices, and metadata; trains a fresh model
for each evaluation protocol/fold/imputation method; saves probabilities and QA.

Architecture defaults (840 -> 128 -> 64 -> 8) are RECONSTRUCTION choices,
not verified hyperparameters from the published SoccerMon DeepHit paper.

The 8-category softmax models injury on each of days 1..7, or survival beyond
7 days. Event at day d: -log p[d]. Censoring through day c: -log P(T > c),
including short follow-up c < 7. Optional pairwise DeepHit-style ranking loss
can be enabled with --ranking-weight, but is off by default because only two
unique injury episodes underpin the available positive prediction windows.

No validation/early-stopping or hyperparameter search uses held-out test data.
Set hyperparameters a priori. Test metrics are DESCRIPTIVE, not suitable for
model selection. Overlapping windows are dependent observations; LOPO folds
with zero events cannot supply a meaningful C-index.

Example:
 python src/preprocessing/paper_inspired/13_train_deephit_mlp.py --protocol chronological
 python src/preprocessing/paper_inspired/13_train_deephit_mlp.py --protocol lopo --fold-id lopo_01
 python src/preprocessing/paper_inspired/13_train_deephit_mlp.py --protocol all
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_TENSORS = Path('data/modeling/paper_inspired/tensors')
DEFAULT_OUTPUT = Path('data/modeling/paper_inspired/models')
METHODS = ('median', 'linear', 'bespoke')
HORIZON = 7


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DeepHitMLP(nn.Module):
    """Single-risk discrete-time MLP with logits for 7 event days + beyond-day-7."""
    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, HORIZON + 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.flatten(start_dim=1))  # raw logits, not probabilities


def censoring_nll(logits: torch.Tensor, duration: torch.Tensor,
                  event: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood for event day d or right censoring through day c.

    Logit indices 0..6 represent event on days 1..7; index 7 represents T > 7.
    For a censored sample with duration c, the admissible categories are
    {c+1,...,7,>7}; duration c=7 admits only the last category.
    """
    if logits.ndim != 2 or logits.shape[1] != HORIZON + 1:
        raise ValueError('Expected logits [batch, 8].')
    if duration.shape != event.shape or duration.ndim != 1:
        raise ValueError('Expected one duration and event per sample.')
    if not bool(((duration >= 1) & (duration <= HORIZON)).all()):
        raise ValueError('Durations must be 1..7.')
    if not bool(((event == 0) | (event == 1)).all()):
        raise ValueError('Events must be 0 or 1.')

    logp = torch.log_softmax(logits, dim=1)
    event_logp = logp.gather(1, (duration - 1).unsqueeze(1)).squeeze(1)
    categories = torch.arange(1, HORIZON + 2, device=logits.device)
    admissible = categories.unsqueeze(0) > duration.unsqueeze(1)
    censor_logp = torch.logsumexp(logp.masked_fill(~admissible, float('-inf')), dim=1)
    return -torch.where(event.bool(), event_logp, censor_logp).mean()


def ranking_loss(logits: torch.Tensor, duration: torch.Tensor,
                 event: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Optional pairwise loss for event i and j with follow-up strictly > time_i.

    At event time d_i, encourage F_i(d_i) > F_j(d_i). This is a simplified
    single-risk DeepHit-inspired ranking term; it is NOT a verified copy of
    the SoccerMon paper's implementation. The pair count is usually tiny.
    """
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    cdf = torch.softmax(logits, dim=1)[:, :HORIZON].cumsum(dim=1)
    comparable = event.bool().unsqueeze(1) & (duration.unsqueeze(1) < duration.unsqueeze(0))
    if not bool(comparable.any()):
        return logits.sum() * 0.0
    idx = (duration - 1).clamp(0, HORIZON - 1)
    score_i = cdf.gather(1, idx[:, None]).squeeze(1)
    score_j_at_i = cdf[:, idx].T  # [i, j] = F_j(d_i)
    pair_penalty = torch.exp(torch.clamp(-(score_i[:, None] - score_j_at_i) / sigma, -40, 40))
    return pair_penalty[comparable].mean()


def load_partition(folder: Path, partition: str, expected_history: int, expected_features: int):
    paths = {key: folder / f'{key}_{partition}.npy' for key in ('X', 'duration', 'event')}
    paths['index'] = folder / f'sample_index_{partition}.csv'
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing Script 12 input: {path}')
    X = np.load(paths['X'], allow_pickle=False)
    durations = np.load(paths['duration'], allow_pickle=False)
    events = np.load(paths['event'], allow_pickle=False)
    index = pd.read_csv(paths['index'], low_memory=False)
    if X.ndim != 3 or X.shape[1:] != (expected_history, expected_features):
        raise ValueError(f'Unexpected {partition} feature tensor shape: {X.shape}')
    if len(X) != len(durations) or len(X) != len(events) or len(X) != len(index):
        raise ValueError(f'{partition}: data / labels / sample index counts differ.')
    if X.dtype.kind != 'f' or not np.isfinite(X).all():
        raise ValueError(f'{partition}: nonfinite or non-floating input tensor')
    if not set(np.unique(events)).issubset({0, 1}):
        raise ValueError(f'{partition}: nonbinary events')
    if not np.all((durations >= 1) & (durations <= HORIZON)):
        raise ValueError(f'{partition}: durations outside 1..7')
    if not {'sample_id', 'player_name', 'prediction_date', 'event', 'duration_days'}.issubset(index.columns):
        raise ValueError(f'{partition}: missing provenance fields in sample index')
    if index['sample_id'].duplicated().any():
        raise ValueError(f'{partition}: duplicate sample IDs')
    if not np.array_equal(events.astype(int), index['event'].to_numpy(dtype=int)):
        raise ValueError(f'{partition}: event array and sample_index disagree')
    if not np.array_equal(durations.astype(int), index['duration_days'].to_numpy(dtype=int)):
        raise ValueError(f'{partition}: duration array and sample_index disagree')
    if 'history_end' in index and 'history_start' in index:
        pred = pd.to_datetime(index.prediction_date)
        start = pd.to_datetime(index.history_start)
        end = pd.to_datetime(index.history_end)
        if not ((pred - start).dt.days == expected_history).all() or not ((pred - end).dt.days == 1).all():
            raise ValueError(f'{partition}: invalid historical feature date boundaries')
    return (X.astype(np.float32, copy=False), durations.astype(np.int64, copy=False),
            events.astype(np.int64, copy=False), index)


def validate_fold(folder: Path, meta: dict, train: tuple, test: tuple,
                  protocol: str, fold: str, method: str) -> None:
    if (meta.get('protocol'), meta.get('fold'), meta.get('method')) != (protocol, fold, method):
        raise ValueError(f'{folder}: incorrect tensor_metadata protocol/fold/method')
    if meta.get('history_days') != 21 or meta.get('forecast_days') != HORIZON:
        raise ValueError(f'{folder}: unexpected history or forecast duration')
    n_features = len(meta.get('feature_names', []))
    if n_features != 40 or meta.get('flattened_mlp_features') != 21 * n_features:
        raise ValueError(f'{folder}: unexpected feature metadata')
    ti, vi = train[3], test[3]
    if set(ti.sample_id).intersection(vi.sample_id):
        raise ValueError(f'{folder}: training and testing contain shared sample IDs')
    if protocol == 'lopo':
        held = meta.get('held_out_player')
        if not held or held in set(ti.player_name) or set(vi.player_name) != {held}:
            raise ValueError(f'{folder}: LOPO held-out player contamination')
    if protocol == 'chronological':
        if pd.to_datetime(ti.prediction_date).max() >= pd.to_datetime(vi.prediction_date).min():
            raise ValueError(f'{folder}: chronological split out of order')
    for p, loaded in [('train', train), ('test', test)]:
        expected = meta.get('partitions', {}).get(p, {}).get('samples')
        if expected is not None and int(expected) != len(loaded[0]):
            raise ValueError(f'{folder}: metadata count mismatch for {p}')


@torch.no_grad()
def predict(net: nn.Module, X: np.ndarray, device: torch.device,
            batch_size: int) -> np.ndarray:
    net.eval()
    outputs = []
    for begin in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[begin:begin + batch_size]).to(device)
        outputs.append(torch.softmax(net(batch), dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 8), dtype=np.float32)


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based AUC without an optional sklearn dependency."""
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return None
    ranks = pd.Series(scores).rank(method='average').to_numpy()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def concordance(duration: np.ndarray, event: np.ndarray, probability: np.ndarray):
    """Time-dependent comparable-pair C-index using F_j(t_i) for event i.

    Valid pairs: i has event at d_i, other j is observed longer than d_i.
    A point estimate is descriptive only; overlapping injury windows are not
    independent. Returns None if no comparable pairs.
    """
    cdf = probability[:, :HORIZON].cumsum(axis=1)
    numerator, denominator = 0.0, 0
    for i in np.flatnonzero(event == 1):
        competitors = np.flatnonzero(duration > duration[i])
        if len(competitors) == 0:
            continue
        a = cdf[i, duration[i] - 1]
        b = cdf[competitors, duration[i] - 1]
        numerator += float((a > b).sum() + 0.5 * (a == b).sum())
        denominator += len(competitors)
    return ((numerator / denominator) if denominator else None), int(denominator)


def metrics(duration: np.ndarray, event: np.ndarray, p: np.ndarray) -> dict:
    if p.shape != (len(event), HORIZON + 1):
        raise ValueError('Predictions do not have [samples, 8] shape')
    if not np.isfinite(p).all() or np.min(p) < 0 or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError('Invalid predicted probability distributions')
    risks = p[:, :HORIZON].sum(axis=1)
    full = (event == 1) | ((event == 0) & (duration == HORIZON))
    y = event[full]
    scored = risks[full]
    ci, comparable = concordance(duration, event, p)
    out = {
        'samples': int(len(event)), 'positive_windows': int(event.sum()),
        'short_censored': int(((event == 0) & (duration < HORIZON)).sum()),
        'known_seven_day_outcomes': int(full.sum()),
        'seven_day_brier': float(np.mean((scored - y) ** 2)) if len(y) else None,
        'seven_day_auc': safe_auc(y, scored),
        'concordance_index': ci, 'comparable_pairs': comparable,
        'mean_predicted_seven_day_risk': float(np.mean(risks)) if len(risks) else None,
    }
    return out


def output_predictions(index: pd.DataFrame, probs: np.ndarray, path: Path):
    output = index.copy()
    for d in range(1, HORIZON + 1):
        output[f'p_injury_day_{d}'] = probs[:, d - 1]
        output[f'cumulative_risk_day_{d}'] = probs[:, :d].sum(axis=1)
    output['p_no_injury_through_day_7'] = probs[:, HORIZON]
    output['risk_within_7_days'] = probs[:, :HORIZON].sum(axis=1)
    output.to_csv(path, index=False)


def train_one(folder: Path, output: Path, args, protocol: str, fold: str, method: str,
              device: torch.device) -> dict:
    meta_path = folder / 'tensor_metadata.json'
    if not meta_path.is_file():
        raise FileNotFoundError(f'Missing Script 12 metadata: {meta_path}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    features = len(meta.get('feature_names', []))
    history = int(meta.get('history_days', 0))
    tr = load_partition(folder, 'train', history, features)
    te = load_partition(folder, 'test', history, features)
    validate_fold(folder, meta, tr, te, protocol, fold, method)
    X, dur, ev, train_index = tr
    X_test, test_duration, test_event, test_index = te
    if len(X) == 0 or len(X_test) == 0:
        raise ValueError(f'{folder}: no training or testing samples')
    if ev.sum() == 0:
        print(f'WARNING: {folder}: no training injury windows, learning may be degenerate', file=sys.stderr)

    # Fresh network and optimizer for EVERY (protocol, fold, method) experiment.
    seed_all(args.seed)
    network = DeepHitMLP(history * features, args.hidden1, args.hidden2, args.dropout).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(dur), torch.from_numpy(ev))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=False, generator=generator, num_workers=0)
    output.mkdir(parents=True, exist_ok=True)
    hist = []
    for epoch in range(1, args.epochs + 1):
        network.train()
        nll_sum, rank_sum, total_sum, count = 0., 0., 0., 0
        for features_x, duration_y, event_y in loader:
            features_x, duration_y, event_y = (features_x.to(device), duration_y.to(device),
                                                event_y.to(device))
            optimizer.zero_grad(set_to_none=True)
            logits = network(features_x)
            nll = censoring_nll(logits, duration_y, event_y)
            rank = (ranking_loss(logits, duration_y, event_y, args.rank_sigma)
                    if args.ranking_weight > 0 else logits.sum() * 0)
            loss = nll + args.ranking_weight * rank
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'{folder}: nonfinite training loss at epoch {epoch}')
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
            optimizer.step()
            n = len(features_x)
            nll_sum += n * float(nll.detach())
            rank_sum += n * float(rank.detach())
            total_sum += n * float(loss.detach())
            count += n
        hist.append({'epoch': epoch, 'training_nll': nll_sum / count,
                     'training_rank_loss': rank_sum / count,
                     'training_total_loss': total_sum / count})
        if args.verbose:
            print(f'    epoch {epoch:3d}/{args.epochs}: training_loss={total_sum/count:.6f}')

    train_prob = predict(network, X, device, args.batch_size)
    test_prob = predict(network, X_test, device, args.batch_size)
    train_metrics = metrics(dur, ev, train_prob)
    test_metrics = metrics(test_duration, test_event, test_prob)
    output_predictions(train_index, train_prob, output / 'predictions_train.csv')
    output_predictions(test_index, test_prob, output / 'predictions_test.csv')
    pd.DataFrame(hist).to_csv(output / 'training_history.csv', index=False)
    torch.save({'model_state_dict': network.cpu().state_dict(),
                'architecture': {'input_dim': history * features,
                                 'hidden1': args.hidden1, 'hidden2': args.hidden2,
                                 'dropout': args.dropout, 'output_dim': HORIZON + 1}},
               output / 'model.pt')
    config = {
        'script': '13_train_deephit_mlp.py', 'protocol': protocol, 'fold': fold,
        'method': method, 'held_out_player': meta.get('held_out_player'),
        'tensor_metadata_path': str(meta_path),
        'tensor_preprocessing_sha256': meta.get('preprocessing_parameters_sha256'),
        'feature_names': meta['feature_names'],
        'tensor_order': meta.get('tensor_order'),
        'architecture': {'history_days': history, 'features_per_day': features,
                         'input_dim': history * features, 'hidden_layers': [args.hidden1, args.hidden2],
                         'hidden_activation': 'ReLU', 'dropout': args.dropout,
                         'logits': HORIZON + 1, 'output_probabilities': 'softmax'},
        'loss': {'censoring_aware_nll': True,
                 'ranking_weight': args.ranking_weight, 'ranking_sigma': args.rank_sigma},
        'training': {'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
                     'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
                     'max_grad_norm': args.max_grad_norm, 'device': str(device),
                     'validation_policy': 'fixed hyperparameters/epochs, no test-based stopping'},
        'train_metrics': train_metrics, 'test_metrics': test_metrics,
        'limitations': [
            'Paper-inspired reconstruction, not verified original published architecture.',
            'There are only two independent supporting injury events in the current cohort.',
            'Positive windows for a given injury are dependent; do not treat as independent injuries.',
            'AUC and C-index may be undefined when test outcomes contain no observed injuries.',
            'Seven-day metrics exclude short-censored samples; no IPCW adjustment is used.',
            'No model selection on held-out test data; fixed epochs are not early stopping.',
            'Ranking loss is an optional simplified single-risk DeepHit-style regularizer.',
        ],
    }
    (output / 'run_metadata.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf-8')
    result = {
        'protocol': protocol, 'fold': fold, 'method': method,
        'train_samples': train_metrics['samples'], 'test_samples': test_metrics['samples'],
        'train_positive_windows': train_metrics['positive_windows'],
        'test_positive_windows': test_metrics['positive_windows'],
        'test_known_seven_day_outcomes': test_metrics['known_seven_day_outcomes'],
        'test_brier_7d': test_metrics['seven_day_brier'],
        'test_auc_7d': test_metrics['seven_day_auc'],
        'test_c_index': test_metrics['concordance_index'],
        'test_comparable_pairs': test_metrics['comparable_pairs'],
        'final_training_loss': hist[-1]['training_total_loss'],
    }
    print(f'{protocol}/{fold}/{method}: train={len(X):,} test={len(X_test):,}; '
          f'positive={int(ev.sum())}/{int(test_event.sum())}; '
          f'test_7d_brier={result["test_brier_7d"]:.5f} '
          f'test_C={result["test_c_index"] if result["test_c_index"] is not None else "undefined"}')
    return result


def discover(input_dir: Path, protocol: str, fold_id: str | None, method: str):
    methods = METHODS if method == 'all' else (method,)
    selected = []
    if protocol in ('all', 'chronological'):
        for m in methods:
            selected.append(('chronological', 'chronological', m, input_dir / 'chronological' / m))
    if protocol in ('all', 'lopo'):
        if fold_id:
            folds = [input_dir / 'lopo' / fold_id]
        else:
            folds = sorted(p for p in (input_dir / 'lopo').glob('lopo_*') if p.is_dir())
            if not folds:
                raise FileNotFoundError(f'No LOPO tensor directories in {input_dir / "lopo"}')
        for f in folds:
            for m in methods:
                selected.append(('lopo', f.name, m, f / m))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tensors', type=Path, default=DEFAULT_TENSORS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--protocol', choices=['chronological', 'lopo', 'all'], default='chronological')
    parser.add_argument('--fold-id', help='LOPO fold name (e.g. lopo_01); --protocol lopo only')
    parser.add_argument('--method', choices=['all', *METHODS], default='all')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden1', type=int, default=128)
    parser.add_argument('--hidden2', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--ranking-weight', type=float, default=0.0)
    parser.add_argument('--rank-sigma', type=float, default=0.1)
    parser.add_argument('--max-grad-norm', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--verbose', action='store_true', help='Print each training epoch')
    args = parser.parse_args()
    if args.fold_id and args.protocol != 'lopo':
        parser.error('--fold-id can only be used with --protocol lopo')
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden1 <= 0 or args.hidden2 <= 0:
        parser.error('epochs, batch-size, hidden layer widths must be positive')
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error('dropout must be in [0,1), LR positive, weight decay nonnegative')
    if args.ranking_weight < 0 or args.rank_sigma <= 0:
        parser.error('ranking weight must be nonnegative, sigma positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA selected but unavailable')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else ('cpu' if args.device == 'auto' else args.device))
    jobs = discover(args.tensors, args.protocol, args.fold_id, args.method)
    if not jobs:
        parser.error('No tensor jobs found')
    print(f'SoccerMon Script 13: {len(jobs)} runs; device={device}; '
          f'epochs={args.epochs}; ranking_weight={args.ranking_weight}')
    results = []
    for protocol, fold, method, folder in jobs:
        out = (args.output / 'chronological' / method if protocol == 'chronological'
               else args.output / 'lopo' / fold / method)
        results.append(train_one(folder, out, args, protocol, fold, method, device))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / 'model_diagnostics.csv'
    existing = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    current = pd.DataFrame(results)
    if not existing.empty:
        existing = existing.loc[~existing.set_index(['protocol', 'fold', 'method']).index.isin(
            current.set_index(['protocol', 'fold', 'method']).index)]
    combined = pd.concat([existing, current], ignore_index=True)
    combined.sort_values(['protocol', 'fold', 'method']).to_csv(summary, index=False)
    lines = [
        'SoccerMon Script 13 - Paper-inspired DeepHit-style MLP',
        f'Runs in this invocation: {len(results)}',
        f'Runs in diagnostics: {len(combined)}',
        f'Input: [N,21,40] -> 840; hidden: {args.hidden1},{args.hidden2}; output: 8',
        f'Epochs: {args.epochs}, batch size: {args.batch_size}, seed: {args.seed}',
        f'Censor-aware NLL, ranking weight: {args.ranking_weight}',
        'All test data excluded from training and hyperparameter decisions.',
        'Undefined event-based metrics are written as missing, NOT as zero.',
        'WARNING: Positive windows are not independent injuries; only two supporting injury events.',
        'WARNING: This is a paper-inspired reconstruction, not an exact architecture replication.',
    ]
    (args.output / 'model_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FloatingPointError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
