"""Single-stock trading simulator shared by the LLM environment and all baselines.

One episode replays `window_len` consecutive trading days of one ticker. At the
close of day d the agent picks a target position in {0, 50, 100}% of capital;
the position is held until the next close. Every rollout of the same task sees
identical market data (no market impact), so rollouts only diverge through
their own position path -- exactly the source of historical-context
inconsistency studied in HGPO.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS

SPLITS = {
    "train": ("2015-01-01", "2024-01-01"),
    "val": ("2024-01-01", "2025-01-01"),
    "test": ("2025-01-01", "2026-10-01"),  # after the Qwen2.5 knowledge cutoff
}
POSITION_LEVELS = (0, 50, 100)
REWARD_TYPES = ("log_return", "excess_terminal", "risk_adjusted")
SUCCESS_REWARD = 10.0  # same scale as the ALFWorld / WebShop success reward


@dataclass
class MarketData:
    ticker: str
    dates: np.ndarray  # datetime64[ns]
    close: np.ndarray
    features: np.ndarray  # (num_days, len(FEATURE_COLUMNS))

    def col(self, name: str) -> np.ndarray:
        return self.features[:, FEATURE_COLUMNS.index(name)]


def load_market(data_dir: str, ticker: str) -> MarketData:
    df = pd.read_parquet(os.path.join(data_dir, f"{ticker}.parquet"))
    return MarketData(
        ticker=ticker,
        dates=df.index.values,
        close=df["close"].to_numpy(dtype=np.float64),
        features=df[FEATURE_COLUMNS].to_numpy(dtype=np.float64),
    )


def make_windows(market: MarketData, split: str, window_len: int, lookback: int, stride: int = 1) -> List[int]:
    """Start indices whose decision days and final settlement close all fall inside `split`."""
    start, end = (np.datetime64(d) for d in SPLITS[split])
    inside = np.flatnonzero((market.dates >= start) & (market.dates < end))
    if len(inside) == 0:
        return []
    first = max(int(inside[0]), lookback - 1)
    last = int(inside[-1]) - window_len  # day s + window_len settles the last trade
    return list(range(first, last + 1, stride))


class TradingSim:
    def __init__(
        self,
        market: MarketData,
        start: int,
        window_len: int = 20,
        cost_bps: float = 10.0,
        reward_type: str = "log_return",
        risk_lambda: float = 0.5,
        lookback: int = 10,
    ):
        assert reward_type in REWARD_TYPES, f"unknown reward_type {reward_type}"
        assert start - lookback + 1 >= 0 and start + window_len < len(market.close)
        self.market = market
        self.start = start
        self.window_len = window_len
        self.cost_rate = cost_bps / 1e4
        self.reward_type = reward_type
        self.risk_lambda = risk_lambda
        self.lookback = lookback
        self.reset()

    # ------------------------------------------------------------------ state
    @property
    def day(self) -> int:
        return self.start + self.t

    @property
    def done(self) -> bool:
        return self.t >= self.window_len

    @property
    def episode_return(self) -> float:
        return self.value - 1.0

    @property
    def bh_return(self) -> float:
        """Buy-and-hold over the elapsed days, paying the entry cost once."""
        c = self.market.close
        return (1.0 - self.cost_rate) * c[self.day] / c[self.start] - 1.0

    @property
    def won(self) -> bool:
        # tolerance so that replicating buy-and-hold exactly does not count as beating it
        return bool(self.value > 1.0 + self.bh_return + 1e-9)

    def anchor(self) -> str:
        # market data is shared by the whole group, so (day, position) identifies the state
        return f"t={self.t}|pos={self.position}"

    def reset(self) -> Tuple[str, dict]:
        self.t = 0
        self.position = 0
        self.value = 1.0
        self.peak = 1.0
        self.turnover = 0.0
        self.step_returns: List[float] = []
        self.positions: List[int] = []
        return self.render(), self._info()

    # ------------------------------------------------------------------ dynamics
    def step(self, target: Optional[int]) -> Tuple[str, float, bool, dict]:
        """`target` is a level in POSITION_LEVELS, or None for an invalid output (position kept)."""
        if self.done:
            return self.render(), 0.0, True, self._info()

        valid = target in POSITION_LEVELS
        new_pos = target if valid else self.position
        traded = abs(new_pos - self.position) / 100.0
        cost = self.cost_rate * traded
        c = self.market.close
        mkt_ret = c[self.day + 1] / c[self.day] - 1.0
        gross = (1.0 - cost) * (1.0 + new_pos / 100.0 * mkt_ret)

        dd_prev = 1.0 - self.value / self.peak
        self.value *= gross
        self.peak = max(self.peak, self.value)
        dd_new = 1.0 - self.value / self.peak

        self.position = new_pos
        self.turnover += traded
        self.step_returns.append(gross - 1.0)
        self.positions.append(new_pos)
        self.t += 1

        log_ret = 100.0 * np.log(gross)
        if self.reward_type == "log_return":
            reward = log_ret
        elif self.reward_type == "risk_adjusted":
            reward = log_ret - self.risk_lambda * 100.0 * max(0.0, dd_new - dd_prev)
        else:  # excess_terminal: sparse success signal, mirrors the HGPO paper setting
            reward = SUCCESS_REWARD if (self.done and self.won) else 0.0

        info = self._info()
        tag = "" if valid else " (invalid output, position kept)"
        info["action_summary"] = f"target {new_pos}%{tag}, reward {reward:+.2f}"
        return self.render(), float(reward), self.done, info

    def _info(self) -> dict:
        info = {
            "anchor": self.anchor(),
            "summary": self.summary(),
            "won": self.won,
            "ticker": self.market.ticker,
            "start_date": str(self.market.dates[self.start])[:10],
            "position": self.position,
            "episode_return": self.episode_return,
            "bh_return": self.bh_return,
            "turnover": self.turnover,
        }
        if self.done:
            info["step_returns"] = list(self.step_returns)
            info["positions"] = list(self.positions)
        return info

    # ------------------------------------------------------------------ observations
    def _features(self) -> dict:
        d = min(self.day, len(self.market.close) - 1)
        return {name: self.market.features[d, i] for i, name in enumerate(FEATURE_COLUMNS)}

    def render(self) -> str:
        """Current observation as text. Ticker and calendar dates are hidden to limit look-ahead from pretraining."""
        f = self._features()
        rets = self.market.col("ret")[self.day - self.lookback + 1 : self.day + 1] * 100.0
        ret_str = ", ".join(f"{r:+.2f}%" for r in rets)
        day_no = min(self.t + 1, self.window_len)
        return (
            f"Trading day {day_no} of {self.window_len}.\n"
            f"Daily returns of Stock A over the last {self.lookback} sessions (oldest to newest): {ret_str}.\n"
            f"Close vs 30-day SMA: {f['sma30_gap'] * 100:+.2f}%; close vs 60-day SMA: {f['sma60_gap'] * 100:+.2f}%.\n"
            f"RSI(30): {f['rsi']:.1f}; MACD: {f['macd_pct']:+.2f}% of price; CCI(30): {f['cci']:.1f}; DX(30): {f['dx']:.1f}; "
            f"Bollinger position: {f['bb_pos']:.2f} (0 = lower band, 1 = upper band); volume vs 20-day average: {f['vol_ratio']:.2f}x.\n"
            f"Current position: {self.position}% of capital in Stock A. "
            f"Episode return so far: {self.episode_return * 100:+.2f}% (buy-and-hold over the same days: {self.bh_return * 100:+.2f}%)."
        )

    def summary(self) -> str:
        """Compact version of the current observation, stored in the agent's memory."""
        f = self._features()
        return (
            f"day return {f['ret'] * 100:+.2f}%, RSI {f['rsi']:.1f}, MACD {f['macd_pct']:+.2f}%, "
            f"position {self.position}%"
        )

    def numeric_state(self) -> np.ndarray:
        """Same information as `render`, as a vector for classic DRL baselines."""
        f = self._features()
        rets = self.market.col("ret")[self.day - self.lookback + 1 : self.day + 1] * 100.0
        extra = [
            f["sma30_gap"] * 100, f["sma60_gap"] * 100, f["rsi"] / 100, f["macd_pct"],
            f["cci"] / 100, f["dx"] / 100, f["bb_pos"], f["vol_ratio"],
            self.position / 100, self.episode_return * 100, self.bh_return * 100, self.t / self.window_len,
        ]
        return np.concatenate([rets, extra]).astype(np.float32)
