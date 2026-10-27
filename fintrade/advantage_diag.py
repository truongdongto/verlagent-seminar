"""Advantage-bias diagnostics (claim C2): how far is each estimator from the Oracle advantage?

Reproduces the analysis of HGPO Fig. 2 on the rollouts of ANY training run. All estimators are
recomputed on the same batch with the same normalization ((r - mean) / (std + eps), population
std, groups of size 1 get 0), so the only difference between them is the grouping:

    traj       GRPO     episode return, normalized over the rollout group (uid)
    step0      GiGPO    step return, normalized over steps with the same current anchor (0-context)
    gigpo      GiGPO    traj + step0 (joint advantage with step_advantage_w = 1)
    hgpo       HGPO     weighted sum over 0..K-context groups, same rule as recipe/hgpo/core_hgpo.py

and two references:

    oracleK    steps sharing the current anchor AND the K previous anchors (the memory window);
               this is the deepest level used by HGPO and the Oracle of the HGPO paper
    oraclefull steps sharing the whole anchor prefix since the start of the episode; not one of
               HGPO's levels once t > K, so it is a check that does not favour HGPO by construction

Metrics are taken over steps whose oracle group has more than one member.
"""

from collections import defaultdict
from typing import Dict, Sequence

import numpy as np

ESTIMATORS = ("traj", "step0", "gigpo", "hgpo")
ORACLES = ("oracleK", "oraclefull")


def _normalize(values: np.ndarray, eps: float) -> np.ndarray:
    if values.size <= 1:
        return np.zeros_like(values)
    return (values - values.mean()) / (values.std() + eps)


def discounted_step_returns(rewards: Sequence[float], traj_uid: Sequence, gamma: float) -> np.ndarray:
    """Per-step discounted return; steps of a trajectory must appear in time order."""
    rewards = np.asarray(rewards, dtype=np.float64)
    out = np.zeros_like(rewards)
    by_traj = defaultdict(list)
    for i, t in enumerate(traj_uid):
        by_traj[t].append(i)
    for idx in by_traj.values():
        running = 0.0
        for i in reversed(idx):
            running = rewards[i] + gamma * running
            out[i] = running
    return out


def compute_estimators(
    uid: Sequence,
    traj_uid: Sequence,
    anchors: Sequence,
    step_returns: np.ndarray,
    episode_scores: np.ndarray,
    history_length: int,
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Returns {name: advantages [N]} for ESTIMATORS and ORACLES, plus "<oracle>_defined" masks.
    Level k = 1..history_length+1 is the number of trailing anchors in the key (k=1 is 0-context),
    and HGPO weights are (k+1)^alpha over levels with non-zero advantage, as in core_hgpo.py.
    """
    n = len(anchors)
    anchors = [str(a) for a in anchors]
    step_returns = np.asarray(step_returns, dtype=np.float64)
    episode_scores = np.asarray(episode_scores, dtype=np.float64)
    out = {name: np.zeros(n) for name in ESTIMATORS + ORACLES}
    defined = {name: np.zeros(n, dtype=bool) for name in ORACLES}
    max_k = history_length + 1

    groups = defaultdict(list)
    for i, g in enumerate(uid):
        groups[g].append(i)

    for members in groups.values():
        members = np.asarray(members)
        out["traj"][members] = _normalize(episode_scores[members], eps)

        trajs = defaultdict(list)
        for i in members:
            trajs[traj_uid[i]].append(int(i))

        level_clusters = defaultdict(list)  # (k, key) -> step indices
        full_clusters = defaultdict(list)   # full prefix -> step indices
        deepest = {}                        # step -> key of its deepest available level
        for steps in trajs.values():
            seq = [anchors[i] for i in steps]
            for si, i in enumerate(steps):
                k_upper = min(max_k, si + 1)
                for k in range(1, k_upper + 1):
                    level_clusters[(k, tuple(seq[si - k + 1: si + 1]))].append(i)
                deepest[i] = (k_upper, tuple(seq[si - k_upper + 1: si + 1]))
                full_clusters[tuple(seq[: si + 1])].append(i)

        level_adv = defaultdict(dict)  # step -> {k: adv}
        cluster_adv = {}
        for key, idx in level_clusters.items():
            if len(idx) <= 1:
                continue
            adv = _normalize(step_returns[idx], eps)
            cluster_adv[key] = dict(zip(idx, adv))
            for i, a in zip(idx, adv):
                level_adv[i][key[0]] = a

        for idx in full_clusters.values():
            if len(idx) <= 1:
                continue
            out["oraclefull"][idx] = _normalize(step_returns[idx], eps)
            defined["oraclefull"][idx] = True

        for i in members:
            items = level_adv.get(int(i), {})
            out["step0"][i] = items.get(1, 0.0)
            valid = [(k, a) for k, a in items.items() if a != 0]
            if valid:
                w = np.array([(k + 1) ** alpha for k, _ in valid], dtype=np.float64)
                w = w / (w.sum() + eps)
                out["hgpo"][i] = float(np.dot(w, [a for _, a in valid]))
            key = deepest[int(i)]
            if key in cluster_adv:
                out["oracleK"][i] = cluster_adv[key][int(i)]
                defined["oracleK"][i] = True

    out["gigpo"] = out["traj"] + out["step0"]
    for name in ORACLES:
        out[f"{name}_defined"] = defined[name]
    return out


def bias_metrics(est: Dict[str, np.ndarray], prefix: str = "adv_bias") -> Dict[str, float]:
    metrics = {}
    for oracle in ORACLES:
        mask = est[f"{oracle}_defined"]
        metrics[f"{prefix}/{oracle}/coverage"] = float(mask.mean()) if mask.size else 0.0
        if mask.sum() < 2:
            continue
        ref = est[oracle][mask]
        nonzero = ref != 0
        for name in ESTIMATORS:
            a = est[name][mask]
            metrics[f"{prefix}/{oracle}/{name}_mae"] = float(np.mean(np.abs(a - ref)))
            if a.std() > 0 and ref.std() > 0:
                metrics[f"{prefix}/{oracle}/{name}_corr"] = float(np.corrcoef(a, ref)[0, 1])
            if nonzero.any():
                metrics[f"{prefix}/{oracle}/{name}_sign_agree"] = float(np.mean(np.sign(a[nonzero]) == np.sign(ref[nonzero])))
    return metrics


def batch_advantage_bias_stats(non_tensor_batch: Dict, episode_scores: np.ndarray, gamma: float,
                               history_length: int, alpha: float = 1.0) -> Dict[str, float]:
    """Entry point for the trainer: verl-agent batch before any reordering."""
    step_returns = discounted_step_returns(non_tensor_batch["rewards"], non_tensor_batch["traj_uid"], gamma)
    est = compute_estimators(
        non_tensor_batch["uid"], non_tensor_batch["traj_uid"], non_tensor_batch["anchor_obs"],
        step_returns, episode_scores, history_length, alpha,
    )
    return bias_metrics(est)
