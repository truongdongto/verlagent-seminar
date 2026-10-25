"""Evaluation metrics for trading episodes and HGPO-style context-group statistics."""

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

TRADING_DAYS = 252


def series_metrics(daily_returns: Sequence[float]) -> Dict[str, float]:
    r = np.asarray(daily_returns, dtype=np.float64)
    if r.size == 0:
        return {}
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    max_dd = float(np.max(1.0 - equity / peak))
    years = r.size / TRADING_DAYS
    annual_return = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
    std = r.std(ddof=1) if r.size > 1 else 0.0
    downside = r[r < 0].std(ddof=1) if (r < 0).sum() > 1 else 0.0
    return {
        "cumulative_return": float(equity[-1] - 1.0),
        "annual_return": annual_return,
        "annual_vol": float(std * np.sqrt(TRADING_DAYS)),
        "sharpe": float(r.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0,
        "sortino": float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else 0.0,
        "max_drawdown": max_dd,
        "calmar": float(annual_return / max_dd) if max_dd > 0 else 0.0,
    }


def episode_metrics(episodes: List[dict]) -> Dict[str, float]:
    """`episodes` are final-step infos from TradingSim (with step_returns, bh_return, turnover)."""
    excess = np.array([e["episode_return"] - e["bh_return"] for e in episodes])
    out = {
        "num_episodes": len(episodes),
        "mean_episode_return": float(np.mean([e["episode_return"] for e in episodes])),
        "mean_bh_return": float(np.mean([e["bh_return"] for e in episodes])),
        "mean_excess_return": float(excess.mean()),
        "win_rate_vs_bh": float(np.mean([e["won"] for e in episodes])),
        "mean_turnover": float(np.mean([e["turnover"] for e in episodes])),
        "mean_position": float(np.mean([np.mean(e["positions"]) for e in episodes])),
    }
    lo, hi = bootstrap_ci(excess)
    out["excess_return_ci95_low"], out["excess_return_ci95_high"] = lo, hi
    # windows are non-overlapping in evaluation, so concatenating them gives one daily series
    daily = np.concatenate([e["step_returns"] for e in episodes])
    out.update(series_metrics(daily))
    return out


def bootstrap_ci(x: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = x[rng.randint(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def context_group_stats(anchor_seqs: List[List[str]], max_k: int) -> Dict[str, float]:
    """
    Hierarchy-of-groups statistics for ONE rollout group (same task), as in HGPO Table 2 / Fig. 4.
    anchor_seqs[i][t] is the anchor of trajectory i at step t. The k-context key of step t is the
    tuple of anchors t-k..t (k=0 is GiGPO's step-level group). A step is "utilized" at level k if
    its k-context group has more than one member.
    """
    out = {}
    for k in range(max_k + 1):
        keys = [tuple(seq[max(0, t - k): t + 1]) for seq in anchor_seqs for t in range(len(seq))]
        counts = Counter(keys)
        sizes = np.array([counts[key] for key in keys])
        out[f"{k}-context/utilization"] = float(np.mean(sizes > 1))
        out[f"{k}-context/mean_group_size"] = float(np.mean(sizes))
    return out


def batch_context_group_stats(non_tensor_batch: Dict, max_k: int) -> Dict[str, float]:
    """
    context_group_stats averaged over the rollout groups of a training batch. Expects the
    verl-agent batch before any reordering (steps of a trajectory in time order).
    """
    groups: Dict[str, Dict[str, List[str]]] = {}
    for uid, traj, anchor in zip(non_tensor_batch["uid"], non_tensor_batch["traj_uid"], non_tensor_batch["anchor_obs"]):
        groups.setdefault(uid, {}).setdefault(traj, []).append(str(anchor))
    per_group = [context_group_stats(list(trajs.values()), max_k) for trajs in groups.values()]
    return {f"context_groups/{key}": float(np.mean([g[key] for g in per_group])) for key in per_group[0]}
