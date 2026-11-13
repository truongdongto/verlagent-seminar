# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import gym
import numpy as np

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts one or more *WebAgentTextEnv* instances (``envs_per_worker``).
    Envs with the same seed share one ``SimServer`` (products, search index, goals), so packing a
    whole group into one actor costs about one env's RAM; each env then keeps its own browser
    state under its own session prefix.
    """
    
    def __init__(self, seeds, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        if isinstance(seeds, int):
            seeds = [seeds]
        env_kwargs = dict(env_kwargs)
        # 'binary': 10 on success, else 0 (paper); 'score': 10 x WebShop task score at the end (partial credit)
        self.reward_type = env_kwargs.pop('reward_type', 'binary')
        servers = {}
        self.envs = []
        for slot, seed in enumerate(seeds):
            kwargs = dict(env_kwargs, seed=seed)
            if len(seeds) > 1:
                kwargs['session_prefix'] = f'{slot}_'  # envs of a group reset with the same session index
            if seed in servers:
                kwargs['server'] = servers[seed]
            env = gym.make('WebAgentTextEnv-v0', **kwargs)
            servers.setdefault(seed, env.server)
            self.envs.append(env)
    
    def _step(self, env, action):
        """Execute a step in one environment"""
        obs, reward, done, info = env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        info['won'] = bool(done and reward == 1.0)
        if self.reward_type == 'score':
            reward = 10.0 * reward if done else 0  # WebShop scores the bought product in [0, 1]; 1 = success
        else:
            reward = 10.0 if info['won'] else 0

        return obs, reward, done, info

    def step(self, actions):
        """Execute a step in every hosted environment"""
        return [self._step(env, action) for env, action in zip(self.envs, actions)]
    
    def _reset(self, env, idx):
        obs, info = env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = env.get_available_actions()
        info['won'] = False
        return obs, info

    def reset(self, idxs):
        """Reset every hosted environment with the given session indices"""
        return [self._reset(env, idx) for env, idx in zip(self.envs, idxs)]
    
    def render(self, mode_for_render):
        """Render the hosted environments"""
        return [env.render(mode=mode_for_render) for env in self.envs]
    
    def get_available_actions(self):
        """Get available actions"""
        return [env.get_available_actions() for env in self.envs]
    
    def get_goals(self):
        """Get environment goals"""
        return self.envs[0].server.goals
    
    def close(self):
        """Close the environments"""
        for env in self.envs:
            env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        envs_per_worker: int = 1,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        # env i keeps seed + i // group_n as before; consecutive envs are packed envs_per_worker per actor
        self.envs_per_worker = max(1, int(envs_per_worker))
        k = self.envs_per_worker
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for start in range(0, self.num_processes, k):
            seeds = [seed + (i // self.group_n) for i in range(start, min(start + k, self.num_processes))]
            worker = env_worker.remote(seeds, self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        k = self.envs_per_worker
        futures = []
        for w, worker in enumerate(self._workers):
            future = worker.step.remote(actions[w * k:(w + 1) * k])
            futures.append(future)

        # Collect results
        results = [r for chunk in ray.get(futures) for r in chunk]
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        k = self.envs_per_worker
        futures = []
        for w, worker in enumerate(self._workers):
            future = worker.reset.remote(idx[w * k:(w + 1) * k])
            futures.append(future)

        # Collect results
        results = [r for chunk in ray.get(futures) for r in chunk]
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        k = self.envs_per_worker
        if env_idx is not None:
            future = self._workers[env_idx // k].render.remote(mode)
            return ray.get(future)[env_idx % k]

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return [r for chunk in ray.get(futures) for r in chunk]

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    envs_per_worker: int = 1,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        envs_per_worker=envs_per_worker,
    )