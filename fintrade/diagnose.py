"""Environment sanity report. Run from the verl-agent root, no GPU needed:

    python -m fintrade.diagnose                 # Mac or server
    python -m fintrade.diagnose --tokenizer Qwen/Qwen2.5-1.5B-Instruct   # also count prompt tokens

On a machine with torch + omegaconf installed (the training server) it drives the real
TradingEnvironmentManager through make_envs; otherwise it rebuilds prompts with SimpleMemory.
"""

import argparse
import os

import numpy as np

from fintrade.data import FEATURE_COLUMNS, ID_TICKERS, OOD_TICKERS
from fintrade.metrics import context_group_stats
from fintrade.prompts import build_prompt, task_description
from fintrade.sim import SPLITS, TradingSim, load_market, make_windows

WINDOW = 20
LOOKBACK = 10


def section(title):
    print(f"\n===== {title} =====")


def check_data(data_dir):
    section("1. Data coverage")
    ok = True
    for t in ID_TICKERS + OOD_TICKERS:
        path = os.path.join(data_dir, f"{t}.parquet")
        if not os.path.exists(path):
            print(f"{t:6s} MISSING {path}")
            ok = False
            continue
        m = load_market(data_dir, t)
        nan = int(np.isnan(m.features).sum())
        print(f"{t:6s} rows={len(m.close)} {str(m.dates[0])[:10]} -> {str(m.dates[-1])[:10]} "
              f"nan_features={nan} close[min,max]=[{m.close.min():.2f}, {m.close.max():.2f}]")
        ok &= nan == 0
    return ok


def count_windows(data_dir):
    section("2. Episode windows (window_len=20)")
    rows = [("train", ID_TICKERS, 1), ("val", ID_TICKERS, WINDOW), ("test", ID_TICKERS, WINDOW), ("test", OOD_TICKERS, WINDOW)]
    for split, tickers, stride in rows:
        n = sum(len(make_windows(load_market(data_dir, t), split, WINDOW, LOOKBACK, stride)) for t in tickers)
        label = "OOD" if tickers is OOD_TICKERS else "ID"
        hint = " (all start days)" if stride == 1 else f"  -> data.val_batch_size={n} to cover each window once"
        print(f"{split:5s} {label:3s} {SPLITS[split]}: {n} windows{hint}")


def rollout_with_manager(data_dir, history_length, rng):
    from omegaconf import OmegaConf
    from agent_system.environments.env_manager import make_envs

    cfg = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    trading = OmegaConf.to_container(cfg.env.trading)
    trading["data_dir"] = data_dir
    config = OmegaConf.create({
        "data": {"train_batch_size": 2, "val_batch_size": 4},
        "env": {"env_name": "fintrade", "seed": 0, "max_steps": WINDOW, "history_length": history_length,
                "resources_per_worker": {"num_cpus": 0.1, "num_gpus": 0}, "rollout": {"n": 8}, "trading": trading},
    })
    envs, val_envs = make_envs(config)
    obs, infos = envs.reset(kwargs=None)
    prompts, anchors = [obs["text"]], [obs["anchor"]]
    for _ in range(WINDOW):
        texts = [f"<think>diagnose</think><action>{rng.choice([0, 50, 100])}</action>" for _ in obs["text"]]
        obs, rewards, dones, infos = envs.step(texts)
        prompts.append(obs["text"])
        anchors.append(obs["anchor"])
    assert all(dones), "episodes should end after window_len steps"
    print(f"make_envs OK: train envs={len(infos)}, val windows={len(val_envs.envs.tasks)}")
    return [p[0] for p in prompts], infos


def rollout_without_manager(data_dir, history_length, rng):
    from agent_system.memory import SimpleMemory

    market = load_market(data_dir, ID_TICKERS[0])
    start = make_windows(market, "train", WINDOW, LOOKBACK)[0]
    sim = TradingSim(market, start, window_len=WINDOW)
    memory = SimpleMemory()
    memory.reset(batch_size=1)
    task = task_description(WINDOW, "log_return", 10)
    obs, info = sim.reset()
    prompts = [build_prompt(obs, task)]
    summary = info["summary"]
    while not sim.done:
        obs, reward, done, info = sim.step(int(rng.choice([0, 50, 100])))
        memory.store({"text_obs": [summary], "action": [info["action_summary"]]})
        summary = info["summary"]
        ctx, lens = memory.fetch(history_length)
        prompts.append(build_prompt(obs, task, ctx[0], lens[0], len(memory[0])))
    return prompts, [info]


def check_prompts(data_dir, history_length, tokenizer_name):
    section("3. Prompts and environment manager")
    rng = np.random.RandomState(0)
    try:
        prompts, infos = rollout_with_manager(data_dir, history_length, rng)
    except ImportError as e:
        print(f"(torch/omegaconf not available: {e}; rebuilding prompts with SimpleMemory)")
        prompts, infos = rollout_without_manager(data_dir, history_length, rng)

    print("--- prompt at step 1 (no history) ---")
    print(prompts[0])
    print(f"--- prompt at step 4 (history_length={history_length}) ---")
    print(prompts[3])
    longest = max(prompts, key=len)
    print(f"max prompt chars={len(longest)}")
    if tokenizer_name:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        ids = tok.apply_chat_template([{"role": "user", "content": longest}], add_generation_prompt=True)
        print(f"max prompt tokens ({tokenizer_name})={len(ids)}  (data.max_prompt_length is 2048)")
    final = infos[0]
    print(f"final info: return={final['episode_return']:+.4f} bh={final['bh_return']:+.4f} won={final['won']} "
          f"turnover={final['turnover']:.1f}")


def check_groups(data_dir, max_k=4, num_tasks=50, group_n=8):
    section("4. Hierarchy-of-groups statistics for synthetic policies (group of 8, K=4)")
    market = load_market(data_dir, ID_TICKERS[0])
    starts = make_windows(market, "train", WINDOW, LOOKBACK)
    rng = np.random.RandomState(0)
    policies = {
        "uniform random": lambda sim: int(rng.choice([0, 50, 100])),
        "sticky (keep w.p. 0.8)": lambda sim: sim.position if rng.rand() < 0.8 else int(rng.choice([0, 50, 100])),
    }
    for name, policy in policies.items():
        stats = []
        for s in rng.choice(starts, size=num_tasks, replace=False):
            seqs = []
            for _ in range(group_n):
                sim = TradingSim(market, int(s), window_len=WINDOW)
                seq = [sim.anchor()]
                while not sim.done:
                    sim.step(policy(sim))
                    if not sim.done:
                        seq.append(sim.anchor())
                seqs.append(seq)
            stats.append(context_group_stats(seqs, max_k))
        util = "  ".join(f"{k}-C={np.mean([x[f'{k}-context/utilization'] for x in stats]):.2f}" for k in range(max_k + 1))
        size = "  ".join(f"{k}-C={np.mean([x[f'{k}-context/mean_group_size'] for x in stats]):.2f}" for k in range(max_k + 1))
        print(f"{name:24s} step utilization: {util}")
        print(f"{'':24s} mean group size : {size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/fin")
    parser.add_argument("--history_length", type=int, default=2)
    parser.add_argument("--tokenizer", default=None, help="e.g. Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    print(f"features: {FEATURE_COLUMNS}")
    if not check_data(args.data_dir):
        print("\nData check failed: run `python -m fintrade.data --out data/fin` first.")
        return
    count_windows(args.data_dir)
    check_prompts(args.data_dir, args.history_length, args.tokenizer)
    check_groups(args.data_dir)
    print("\nDIAGNOSE DONE")


if __name__ == "__main__":
    main()
