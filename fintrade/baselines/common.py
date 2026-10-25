"""Shared evaluation loop: every baseline is scored on the same non-overlapping windows."""

import json
import os
from typing import Callable, Dict, List, Optional

import pandas as pd

from fintrade.metrics import episode_metrics
from fintrade.sim import TradingSim, load_market, make_windows

# policy(sim) -> target position in POSITION_LEVELS (or None to keep the position)
Policy = Callable[[TradingSim], Optional[int]]


def eval_windows(data_dir: str, tickers: List[str], split: str, window_len: int, lookback: int = 10):
    """(market, start) for every non-overlapping evaluation window."""
    out = []
    for t in tickers:
        market = load_market(data_dir, t)
        out += [(market, s) for s in make_windows(market, split, window_len, lookback, stride=window_len)]
    return out


def run_policy(policy: Policy, windows, sim_kwargs: Dict) -> List[dict]:
    episodes = []
    for market, start in windows:
        sim = TradingSim(market, start, **sim_kwargs)
        while not sim.done:
            _, _, _, info = sim.step(policy(sim))
        episodes.append(info)
    return episodes


def summarize(name: str, episodes: List[dict]) -> Dict:
    row = {"method": name, **episode_metrics(episodes)}
    by_ticker = pd.DataFrame(episodes).groupby("ticker")
    for ticker, df in by_ticker:
        row[f"excess/{ticker}"] = float((df["episode_return"] - df["bh_return"]).mean())
    return row


def save_episodes(path: str, name: str, episodes: List[dict]):
    """One JSON line per window, so methods can be compared window-by-window (paired tests)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for e in episodes:
            rec = {k: e[k] for k in ("ticker", "start_date", "episode_return", "bh_return", "won", "turnover",
                                     "step_returns", "positions")}
            f.write(json.dumps({"method": name, **rec}) + "\n")
