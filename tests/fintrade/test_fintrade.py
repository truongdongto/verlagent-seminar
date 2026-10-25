"""CPU-only tests for the trading simulator, prompts and parsing.

    python -m pytest tests/fintrade -q
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from fintrade.data import FEATURE_COLUMNS
from fintrade.metrics import context_group_stats, series_metrics
from fintrade.prompts import build_prompt, parse_action, task_description
from fintrade.sim import MarketData, TradingSim, make_windows


def synthetic_market(n=400, seed=0):
    rng = np.random.RandomState(seed)
    rets = rng.normal(0.0005, 0.02, size=n)
    close = 100 * np.cumprod(1 + rets)
    feats = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    feats[:, FEATURE_COLUMNS.index("ret")] = np.concatenate([[0.0], close[1:] / close[:-1] - 1])
    dates = np.arange(np.datetime64("2023-06-01"), np.datetime64("2023-06-01") + n).astype("datetime64[ns]")
    return MarketData("TEST", dates, close, feats)


def run(sim, actions):
    infos, rewards = [], []
    for a in actions:
        _, r, _, info = sim.step(a)
        rewards.append(r)
        infos.append(info)
    return rewards, infos


def test_buy_and_hold_matches_hand_computation():
    m = synthetic_market()
    sim = TradingSim(m, start=20, window_len=20, cost_bps=10)
    _, infos = run(sim, [100] * 20)
    expected = (1 - 1e-3) * m.close[40] / m.close[20] - 1
    assert sim.episode_return == pytest.approx(expected, rel=1e-10)
    assert sim.episode_return == pytest.approx(sim.bh_return, rel=1e-10)
    assert not sim.won  # replicating buy-and-hold is not a win
    assert sim.turnover == pytest.approx(1.0)
    assert len(infos[-1]["step_returns"]) == 20


def test_flat_and_costs():
    m = synthetic_market()
    sim = TradingSim(m, start=20, window_len=4, cost_bps=10)
    rewards, _ = run(sim, [0, 0, 0, 0])
    assert sim.value == 1.0 and all(r == 0 for r in rewards)

    sim = TradingSim(m, start=20, window_len=2, cost_bps=10)
    run(sim, [100, 0])
    r1 = m.close[21] / m.close[20] - 1
    assert sim.value == pytest.approx((1 - 1e-3) * (1 + r1) * (1 - 1e-3))
    assert sim.turnover == pytest.approx(2.0)


def test_invalid_action_keeps_position_and_episode_ends():
    m = synthetic_market()
    sim = TradingSim(m, start=20, window_len=3)
    _, infos = run(sim, [50, None, 100])
    assert infos[1]["position"] == 50 and "invalid" in infos[1]["action_summary"]
    assert sim.done
    _, r, done, _ = sim.step(100)  # stepping a finished episode is a no-op
    assert done and r == 0.0 and sim.position == 100


def test_reward_types():
    m = synthetic_market()
    sparse = TradingSim(m, start=20, window_len=5, reward_type="excess_terminal")
    rewards, _ = run(sparse, [0, 100, 0, 100, 0])
    assert all(r == 0 for r in rewards[:-1]) and rewards[-1] in (0.0, 10.0)
    assert (rewards[-1] == 10.0) == sparse.won

    a = TradingSim(m, start=20, window_len=20, reward_type="log_return")
    b = TradingSim(m, start=20, window_len=20, reward_type="risk_adjusted", risk_lambda=0.5)
    ra, _ = run(a, [100] * 20)
    rb, _ = run(b, [100] * 20)
    assert sum(ra) == pytest.approx(100 * np.log(a.value))
    assert all(y <= x + 1e-12 for x, y in zip(ra, rb))


def test_no_look_ahead_in_observation():
    m = synthetic_market()
    sim = TradingSim(m, start=20, window_len=5)
    obs_before = sim.render()
    m.close[21:] *= 2.0  # change the future only
    m.features[21:] += 1.0
    assert sim.render() == obs_before


def test_anchor_is_day_and_position():
    m = synthetic_market()
    a, b = TradingSim(m, 20, 5), TradingSim(m, 20, 5)
    run(a, [100, 50])
    run(b, [0, 50])
    assert a.anchor() == b.anchor() == "t=2|pos=50"
    assert a.render() != b.render()  # same state, different history (episode return differs)


def test_windows_stay_inside_split():
    m = synthetic_market(n=800)
    split_end = np.datetime64("2025-01-01")
    for s in make_windows(m, "val", 20, 10, stride=20):
        assert m.dates[s] >= np.datetime64("2024-01-01")
        assert m.dates[s + 20] < split_end


@pytest.mark.parametrize("text,expected", [
    ("<think>up</think><action>100</action>", (100, 1)),
    ("<think>x</think><action> 50% </action>", (50, 1)),
    ("<think>x</think><action>0</action>", (0, 1)),
    ("<action>100</action>", (100, 0)),  # missing think: executed but penalized
    ("<think>x</think><action>75</action>", (None, 0)),
    ("<think>x</think><action>50 or 100</action>", (None, 0)),
    ("<think>x</think>buy", (None, 0)),
])
def test_parse_action(text, expected):
    assert parse_action(text) == expected


def test_prompt_with_history():
    task = task_description(20, "log_return", 10)
    p0 = build_prompt("OBS", task)
    p1 = build_prompt("OBS", task, "[Observation 1: 'x', Action 1: 'y']", 1, 1)
    assert "Prior to this step" not in p0 and "Prior to this step" in p1 and "step 2" in p1
    assert "<think>" in p0 and "<action>" in p0


def test_context_group_stats():
    seqs = [["a", "b", "c"], ["a", "b", "d"], ["a", "x", "c"]]
    s = context_group_stats(seqs, 2)
    assert s["0-context/utilization"] == pytest.approx(7 / 9)  # "x" and "d" are singletons
    assert s["1-context/utilization"] == pytest.approx(5 / 9)
    assert s["2-context/utilization"] <= s["1-context/utilization"] < s["0-context/utilization"]


def test_batch_context_group_stats_matches_single_group():
    from fintrade.metrics import batch_context_group_stats

    seqs = [["a", "b", "c"], ["a", "b", "d"], ["a", "x", "c"]]
    batch = {"uid": [], "traj_uid": [], "anchor_obs": []}
    for g in ("g1", "g2"):
        for i, seq in enumerate(seqs):
            for a in seq:
                batch["uid"].append(g)
                batch["traj_uid"].append(f"{g}-{i}")
                batch["anchor_obs"].append(a)
    s = batch_context_group_stats(batch, 2)
    assert s["context_groups/1-context/utilization"] == pytest.approx(5 / 9)


def test_series_metrics():
    m = series_metrics([0.01, -0.02, 0.01])
    assert m["max_drawdown"] == pytest.approx(0.02)
    assert m["cumulative_return"] == pytest.approx(1.01 * 0.98 * 1.01 - 1)


def test_vec_env_groups_share_start(tmp_path):
    import pandas as pd
    pytest.importorskip("torch")  # agent_system.environments imports torch
    from agent_system.environments.env_package.trading.envs import TradingVecEnv

    m = synthetic_market(n=900)
    df = pd.DataFrame(m.features, columns=FEATURE_COLUMNS, index=pd.DatetimeIndex(m.dates, name="date"))
    df["close"] = m.close
    df.to_parquet(tmp_path / "TEST.parquet")
    cfg = OmegaConf.create({"trading": {
        "data_dir": str(tmp_path), "train_tickers": ["TEST"], "val_tickers": ["TEST"], "train_split": "val",
        "val_split": "val", "window_len": 5, "val_window_len": 5, "lookback": 10, "cost_bps": 10,
        "reward_type": "log_return", "risk_lambda": 0.5}})
    env = TradingVecEnv(seed=0, env_num=3, group_n=4, is_train=True, env_config=cfg)
    obs, infos = env.reset()
    starts = [i["start_date"] for i in infos]
    for g in range(3):
        assert len(set(starts[g * 4:(g + 1) * 4])) == 1
        assert len(set(obs[g * 4:(g + 1) * 4])) == 1
    obs, rewards, dones, infos = env.step([0, 50, 100, None] * 3)
    assert len(obs) == 12 and not any(dones)
