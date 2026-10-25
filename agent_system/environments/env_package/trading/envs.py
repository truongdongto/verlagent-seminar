from typing import List, Optional

import numpy as np

from fintrade.sim import TradingSim, load_market, make_windows


class TradingVecEnv:
    """
    Vectorized single-stock trading env. The simulator is pure numpy and cheap,
    so all sub-environments run in-process (no Ray workers).

    - env_num  : number of groups (tasks) per reset
    - group_n  : number of rollouts per task; all rollouts of a task share ticker and start day
    - Training samples tasks at random from every start day of the split; validation
      walks deterministically through non-overlapping windows.
    """

    def __init__(self, seed: int, env_num: int, group_n: int, is_train: bool, env_config):
        cfg = env_config.trading
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.window_len = cfg.window_len if is_train else cfg.val_window_len
        self.sim_kwargs = dict(
            window_len=self.window_len,
            cost_bps=cfg.cost_bps,
            reward_type=cfg.reward_type,
            risk_lambda=cfg.risk_lambda,
            lookback=cfg.lookback,
        )

        tickers = list(cfg.train_tickers if is_train else cfg.val_tickers)
        split = cfg.train_split if is_train else cfg.val_split
        stride = 1 if is_train else self.window_len
        self.markets = {t: load_market(cfg.data_dir, t) for t in tickers}
        self.tasks = [
            (t, s) for t in tickers for s in make_windows(self.markets[t], split, self.window_len, cfg.lookback, stride)
        ]
        if not self.tasks:
            raise ValueError(f"No {split} windows of length {self.window_len} for {tickers}")
        if not is_train and env_num != len(self.tasks):
            print(f"[trading] val_batch_size={env_num} but split '{split}' has {len(self.tasks)} windows; "
                  f"set data.val_batch_size={len(self.tasks)} to evaluate every window once per validation.")

        self._rng = np.random.RandomState(seed)
        self._cursor = 0
        self.sims: List[TradingSim] = []

    def reset(self):
        if self.is_train:
            idx = self._rng.choice(len(self.tasks), size=self.env_num, replace=len(self.tasks) < self.env_num)
        else:
            idx = [(self._cursor + i) % len(self.tasks) for i in range(self.env_num)]
            self._cursor = (self._cursor + self.env_num) % len(self.tasks)
        idx = np.repeat(idx, self.group_n)

        self.sims = []
        for i in idx:
            ticker, start = self.tasks[i]
            self.sims.append(TradingSim(self.markets[ticker], start, **self.sim_kwargs))

        obs_list, info_list = [], []
        for sim in self.sims:
            obs, info = sim.reset()
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def step(self, actions: List[Optional[int]]):
        if len(actions) != self.batch_size:
            raise ValueError(f"Got {len(actions)} actions, but the env was initialized with total_envs={self.batch_size}")

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for sim, action in zip(self.sims, actions):
            was_done = sim.done
            obs, reward, done, info = sim.step(action)
            if was_done:
                info["action_summary"] = "episode already finished"
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def close(self):
        self.sims = []


def build_trading_envs(seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True, env_config=None):
    return TradingVecEnv(seed=seed, env_num=env_num, group_n=group_n, is_train=is_train, env_config=env_config)
