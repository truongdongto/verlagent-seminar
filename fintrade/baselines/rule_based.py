"""Rule-based reference policies on the shared TradingSim.

    python -m fintrade.baselines.rule_based --split val
    python -m fintrade.baselines.rule_based --split test --tickers TSLA AMD KO UNH --tag ood
"""

import argparse
import os

import numpy as np
import pandas as pd

from fintrade.baselines.common import eval_windows, run_policy, save_episodes, summarize
from fintrade.data import ID_TICKERS


def _feature(sim, name):
    return sim._features()[name]


def buy_and_hold(sim):
    return 100


def always_flat(sim):
    return 0


def macd_trend(sim):
    return 100 if _feature(sim, "macd_pct") > 0 else 0


def sma_trend(sim):
    return 100 if _feature(sim, "sma30_gap") > 0 else 0


def rsi_reversion(sim):
    rsi = _feature(sim, "rsi")
    if rsi < 45:
        return 100
    if rsi > 60:
        return 0
    return sim.position


def make_random(seed):
    rng = np.random.RandomState(seed)
    return lambda sim: int(rng.choice([0, 50, 100]))


POLICIES = {
    "buy_and_hold": buy_and_hold,
    "always_flat": always_flat,
    "macd_trend": macd_trend,
    "sma_trend": sma_trend,
    "rsi_reversion": rsi_reversion,
    "random": make_random(0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/fin")
    parser.add_argument("--tickers", nargs="+", default=ID_TICKERS)
    parser.add_argument("--split", default="val")
    parser.add_argument("--window_len", type=int, default=20)
    parser.add_argument("--cost_bps", type=float, default=10.0)
    parser.add_argument("--tag", default="id")
    parser.add_argument("--out", default="results/fintrade")
    args = parser.parse_args()

    windows = eval_windows(args.data_dir, args.tickers, args.split, args.window_len)
    sim_kwargs = dict(window_len=args.window_len, cost_bps=args.cost_bps)
    episodes_path = f"{args.out}/rule_based_episodes_{args.split}_{args.tag}_w{args.window_len}.jsonl"
    if os.path.exists(episodes_path):
        os.remove(episodes_path)
    rows = []
    for name, policy in POLICIES.items():
        episodes = run_policy(policy, windows, sim_kwargs)
        save_episodes(episodes_path, name, episodes)
        rows.append(summarize(name, episodes))

    df = pd.DataFrame(rows).set_index("method")
    cols = ["num_episodes", "mean_episode_return", "mean_excess_return", "win_rate_vs_bh", "sharpe",
            "max_drawdown", "mean_turnover"]
    pd.set_option("display.width", 200)
    print(f"split={args.split} tickers={args.tickers} window_len={args.window_len}")
    print(df[cols].round(4).to_string())
    df.to_csv(f"{args.out}/rule_based_{args.split}_{args.tag}_w{args.window_len}.csv")


if __name__ == "__main__":
    main()
