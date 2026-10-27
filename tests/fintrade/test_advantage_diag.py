"""Tests for the advantage-bias diagnostics (claim C2).

    python -m pytest tests/fintrade -q

The last test compares against the real HGPO implementation and needs torch (training server).
"""

import numpy as np
import pytest

from fintrade.advantage_diag import (
    bias_metrics,
    batch_advantage_bias_stats,
    compute_estimators,
    discounted_step_returns,
)


def make_batch(paths, step_rewards, uid="g"):
    """paths[i] = list of anchors of trajectory i; step_rewards[i] = list of per-step rewards."""
    batch = {"uid": [], "traj_uid": [], "anchor_obs": [], "rewards": []}
    for i, (path, rews) in enumerate(zip(paths, step_rewards)):
        for a, r in zip(path, rews):
            batch["uid"].append(uid)
            batch["traj_uid"].append(f"{uid}-{i}")
            batch["anchor_obs"].append(a)
            batch["rewards"].append(r)
    return batch


def test_discounted_step_returns():
    out = discounted_step_returns([1.0, 0.0, 2.0, 5.0], ["a", "a", "a", "b"], gamma=0.5)
    assert out == pytest.approx([1 + 0.5 * 0 + 0.25 * 2, 0 + 0.5 * 2, 2.0, 5.0])


def test_consistent_history_means_no_bias():
    # every trajectory shares the same anchor path, so all groups coincide with the oracle
    path = ["t=0|pos=0", "t=1|pos=50", "t=2|pos=100"]
    rng = np.random.RandomState(0)
    rews = rng.normal(size=(6, 3))
    b = make_batch([path] * 6, rews.tolist())
    ret = discounted_step_returns(b["rewards"], b["traj_uid"], 0.95)
    est = compute_estimators(b["uid"], b["traj_uid"], b["anchor_obs"], ret, np.zeros(len(ret)), history_length=2)
    assert est["oracleK_defined"].all() and est["oraclefull_defined"].all()
    np.testing.assert_allclose(est["step0"], est["oracleK"], atol=1e-6)
    np.testing.assert_allclose(est["hgpo"], est["oracleK"], atol=1e-5)
    m = bias_metrics(est)
    assert m["adv_bias/oracleK/hgpo_mae"] < 1e-5 and m["adv_bias/oracleK/step0_mae"] < 1e-6


def test_hgpo_aggregation_by_hand():
    # two levels for the step at t=1: 0-context {A,B,C} and 1-context {A,B}
    paths = [["t=0|pos=0", "t=1|pos=50"],      # A
             ["t=0|pos=0", "t=1|pos=50"],      # B
             ["t=0|pos=100", "t=1|pos=50"]]    # C (different history)
    rews = [[0.0, 1.0], [0.0, 3.0], [0.0, 8.0]]
    b = make_batch(paths, rews)
    ret = discounted_step_returns(b["rewards"], b["traj_uid"], 1.0)
    est = compute_estimators(b["uid"], b["traj_uid"], b["anchor_obs"], ret, np.zeros(len(ret)), history_length=1, alpha=1.0)

    r0 = np.array([1.0, 3.0, 8.0])
    a0 = (r0 - r0.mean()) / (r0.std() + 1e-6)
    r1 = np.array([1.0, 3.0])
    a1 = (r1 - r1.mean()) / (r1.std() + 1e-6)
    w = np.array([2.0, 3.0]) / (5.0 + 1e-6)  # levels k=1,2 -> weights (k+1)^alpha
    step_a = 1  # trajectory A, t=1 is the 2nd sample
    assert est["step0"][step_a] == pytest.approx(a0[0])
    assert est["oracleK"][step_a] == pytest.approx(a1[0])
    assert est["hgpo"][step_a] == pytest.approx(w[0] * a0[0] + w[1] * a1[0])
    step_c = 5
    assert not est["oracleK_defined"][step_c]  # C's history is unique
    assert est["hgpo"][step_c] == pytest.approx(w[0] / w[0] * a0[2], rel=1e-5)  # only one valid level


def test_inconsistent_history_ranks_estimators():
    # rewards depend on the position path; a coarser grouping mixes different histories
    rng = np.random.RandomState(1)
    paths, rews = [], []
    for _ in range(8):
        pos = rng.choice([0, 50, 100], size=6)
        paths.append([f"t={t}|pos={p}" for t, p in enumerate(pos)])
        rews.append([p / 50.0 + rng.normal(scale=0.1) for p in pos])
    b = make_batch(paths, rews)
    episode = np.repeat([sum(r) for r in rews], 6)
    m = batch_advantage_bias_stats(b, episode, gamma=0.95, history_length=2)
    assert 0 < m["adv_bias/oracleK/coverage"] < 1
    assert m["adv_bias/oracleK/hgpo_mae"] <= m["adv_bias/oracleK/step0_mae"]
    assert m["adv_bias/oracleK/step0_mae"] < m["adv_bias/oracleK/traj_mae"]


def test_matches_real_hgpo_implementation():
    try:  # needs the full verl stack (torch, ray, ...), i.e. the training server
        import torch
        from recipe.hgpo.core_hgpo import hgpo_advantage_estimate
    except ImportError as e:
        pytest.skip(f"verl not installed: {e}")

    rng = np.random.RandomState(2)
    paths, rews = [], []
    for g in range(3):
        for _ in range(8):
            pos = rng.choice([0, 50, 100], size=5)
            paths.append((g, [f"t={t}|pos={p}" for t, p in enumerate(pos)]))
            rews.append(rng.normal(size=5).tolist())
    batch = {"uid": [], "traj_uid": [], "anchor_obs": [], "rewards": []}
    for i, ((g, path), r) in enumerate(zip(paths, rews)):
        for a, x in zip(path, r):
            batch["uid"].append(f"g{g}")
            batch["traj_uid"].append(f"tr{i}")
            batch["anchor_obs"].append(a)
            batch["rewards"].append(x)
    ret = discounted_step_returns(batch["rewards"], batch["traj_uid"], 0.95)
    n = len(ret)
    real = hgpo_advantage_estimate(
        token_level_rewards=torch.zeros(n, 4),
        step_rewards=torch.tensor(ret, dtype=torch.float32),
        response_mask=torch.ones(n, 4),
        anchor_obs=np.array(batch["anchor_obs"], dtype=object),
        index=np.array(batch["uid"], dtype=object),
        traj_index=np.array(batch["traj_uid"], dtype=object),
        history_length=2,
        mode="mean_std_norm",
        length_weight_alpha=1.0,
    )[:, 0].numpy()
    ours = compute_estimators(batch["uid"], batch["traj_uid"], batch["anchor_obs"], ret, np.zeros(n), history_length=2)["hgpo"]
    np.testing.assert_allclose(ours, real, atol=1e-4)
