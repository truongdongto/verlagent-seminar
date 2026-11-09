#!/usr/bin/env python3
"""Short GRPO + LoRA run on the fintrade env, without verl / vLLM / Ray.

Purpose: (1) measure how long one GRPO iteration takes on the GPU at hand and
extrapolate to longer runs, (2) check whether LoRA fine-tuning moves the policy
at all (reward, invalid-format rate, KL, loss) before paying for a full run.

The rollout mirrors TradingEnvironmentManager (same prompts, K-step memory,
<action> parsing, invalid penalty) and the update mirrors the verl GRPO actor
(episode-return advantage normalised within each group, PPO clip, low_var_kl to
the reference model, token-mean loss). The reference model is the base model
with the LoRA adapter switched off, so no second copy is kept in memory.

    # from the verl-agent root
    pip install "transformers>=4.46" "peft>=0.13" accelerate matplotlib
    python examples/fintrade/grpo_lora_lite.py                       # 10 iters, ~80 steps/iter
    python examples/fintrade/grpo_lora_lite.py --iters 30 --n_tasks 8 --window_len 10

Outputs in --out_dir: metrics.jsonl (one line per iteration), summary.json,
metrics.png, and the LoRA adapter (adapter_final/, plus adapter_iterN/ with --save_every).
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from fintrade.data import ID_TICKERS  # noqa: E402
from fintrade.prompts import build_prompt, parse_action, task_description  # noqa: E402
from fintrade.sim import TradingSim, load_market, make_windows  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Full verl config in examples/fintrade/train.sh, used for the extrapolation at the end
FULL_TASKS, FULL_GROUP, FULL_WINDOW, FULL_ITERS = 16, 8, 20, 160


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data/fin"))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "outputs/grpo_lora_lite"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu | mps")
    # env
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--n_tasks", type=int, default=4, help="tasks (ticker, start day) per iteration")
    p.add_argument("--group_n", type=int, default=4, help="rollouts per task (GRPO group size)")
    p.add_argument("--window_len", type=int, default=5, help="trading days per episode")
    p.add_argument("--history_length", type=int, default=2, help="K steps of memory in the prompt")
    p.add_argument("--reward_type", default="log_return")
    p.add_argument("--cost_bps", type=float, default=10.0)
    p.add_argument("--invalid_penalty", type=float, default=0.1)
    # generation
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen_batch", type=int, default=0, help="prompts per generate() call, 0 = all at once")
    # optimisation
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--mini_batch", type=int, default=32, help="samples (env steps) per optimizer step")
    p.add_argument("--micro_batch", type=int, default=4, help="samples per forward/backward")
    p.add_argument("--no_grad_ckpt", action="store_true")
    # validation / saving
    p.add_argument("--n_val", type=int, default=16, help="fixed 2024 windows evaluated before and after training")
    p.add_argument("--val_every", type=int, default=0, help="also validate every N iters (0 = only before/after)")
    p.add_argument("--val_temperature", type=float, default=0.4)
    p.add_argument("--save_every", type=int, default=0, help="save the adapter every N iters (0 = only at the end)")
    p.add_argument("--target_iters", type=int, default=FULL_ITERS, help="iteration count used for the ETA estimate")
    return p.parse_args()


# ---------------------------------------------------------------------- setup
def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    # T4 / P100 have no usable bf16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_model(args, device, dtype):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA weights in fp32 even when the base is fp16/bf16

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}
    return model, tok, eos_ids


# ---------------------------------------------------------------------- rollout
def format_history(mem, k):
    """Same string as agent_system.memory.SimpleMemory.fetch."""
    recent = mem[-k:] if k > 0 else []
    start = len(mem) - len(recent)
    lines = [f"[Observation {start + j + 1}: '{o}', Action {start + j + 1}: '{a}']" for j, (o, a) in enumerate(recent)]
    return "\n".join(lines), len(recent)


def encode_prompts(tok, prompts: List[str]) -> List[List[int]]:
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
             for p in prompts]
    return tok(texts, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def generate(model, tok, eos_ids, prompt_ids, args, temperature, device):
    """Returns (response ids trimmed after the first EOS, truncated flag) per prompt."""
    model.eval()
    bs = args.gen_batch or len(prompt_ids)
    out = []
    for i in range(0, len(prompt_ids), bs):
        batch = tok.pad({"input_ids": prompt_ids[i:i + bs]}, return_tensors="pt").to(device)
        gen = model.generate(**batch, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                             repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                             pad_token_id=tok.pad_token_id)
        for row in gen[:, batch["input_ids"].shape[1]:].tolist():
            end = next((j for j, t in enumerate(row) if t in eos_ids), None)
            out.append((row[:end + 1], False) if end is not None else (row, True))
    return out


def run_episodes(model, tok, eos_ids, sims: List[TradingSim], task: str, args, temperature, device):
    """Play every sim in lockstep (all windows have the same length). One sample per (episode, step)."""
    samples, memory = [], [[] for _ in sims]
    for sim in sims:
        sim.reset()
    while not sims[0].done:
        prompts = []
        for sim, mem in zip(sims, memory):
            hist, n_hist = format_history(mem, args.history_length)
            prompts.append(build_prompt(sim.render(), task, action_history=hist,
                                        history_length=n_hist, step_count=len(mem)))
        prompt_ids = encode_prompts(tok, prompts)
        responses = generate(model, tok, eos_ids, prompt_ids, args, temperature, device)
        texts = tok.batch_decode([r for r, _ in responses], skip_special_tokens=True)
        for i, sim in enumerate(sims):
            target, valid = parse_action(texts[i])
            pre_summary = sim.summary()
            _, reward, _, info = sim.step(target)
            memory[i].append((pre_summary, info["action_summary"]))
            samples.append({
                "traj": i, "prompt": prompt_ids[i], "resp": responses[i][0], "truncated": responses[i][1],
                "valid": valid, "position": sim.position,
                "reward": reward - (0.0 if valid else args.invalid_penalty),
            })
    return samples, [sim._info() for sim in sims]


def episode_stats(samples, infos, prefix) -> Dict[str, float]:
    return {
        f"{prefix}/return_pct": 100 * float(np.mean([e["episode_return"] for e in infos])),
        f"{prefix}/excess_vs_bh_pct": 100 * float(np.mean([e["episode_return"] - e["bh_return"] for e in infos])),
        f"{prefix}/win_rate": float(np.mean([e["won"] for e in infos])),
        f"{prefix}/invalid_rate": float(np.mean([1 - s["valid"] for s in samples])),
        f"{prefix}/truncated_rate": float(np.mean([s["truncated"] for s in samples])),
        f"{prefix}/resp_len": float(np.mean([len(s["resp"]) for s in samples])),
        f"{prefix}/mean_position": float(np.mean([s["position"] for s in samples])),
    }


def grpo_advantages(samples, n_traj, group_n):
    """GRPO: episode return (incl. invalid penalties) normalised within the group, shared by every step."""
    score = np.zeros(n_traj)
    for s in samples:
        score[s["traj"]] += s["reward"]
    adv = np.zeros(n_traj)
    zero_std_groups = 0
    for g in range(0, n_traj, group_n):
        x = score[g:g + group_n]
        std = x.std(ddof=1) if len(x) > 1 else 0.0
        zero_std_groups += std < 1e-6
        adv[g:g + group_n] = (x - x.mean()) / (std + 1e-6)
    for s in samples:
        s["adv"] = float(adv[s["traj"]])
    return {"reward/episode_score": float(score.mean()), "reward/episode_score_std": float(score.std()),
            "reward/zero_std_group_frac": zero_std_groups / max(1, n_traj // group_n)}


# ---------------------------------------------------------------------- update
def collate(batch, pad_id, device):
    """Left-pad prompts and right-pad responses so that every response starts at column P."""
    P = max(len(s["prompt"]) for s in batch)
    R = max(len(s["resp"]) for s in batch)
    ids = torch.full((len(batch), P + R), pad_id, dtype=torch.long)
    attn = torch.zeros_like(ids)
    mask = torch.zeros((len(batch), R))
    for j, s in enumerate(batch):
        p, r = s["prompt"], s["resp"]
        ids[j, P - len(p):P] = torch.tensor(p)
        ids[j, P:P + len(r)] = torch.tensor(r)
        attn[j, P - len(p):P + len(r)] = 1
        mask[j, :len(r)] = 1
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), attn.to(device), pos.to(device), mask.to(device), P, R


def token_logprobs(model, ids, attn, pos, P, R, temperature, with_entropy=False):
    """Log-probs of the response tokens; the lm_head is applied to the response slice only."""
    base = model.get_base_model()
    hidden = base.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
    logits = base.lm_head(hidden[:, P - 1:P + R - 1]).float() / temperature
    logp_all = logits.log_softmax(-1)
    logp = logp_all.gather(-1, ids[:, P:P + R].unsqueeze(-1)).squeeze(-1)
    entropy = None
    if with_entropy:
        with torch.no_grad():
            entropy = -(logp_all.exp() * logp_all).sum(-1)
    return logp, entropy


@torch.no_grad()
def fill_logprobs(model, samples, key, args, pad_id, device, use_adapter=True):
    model.eval()
    for i in range(0, len(samples), args.micro_batch):
        batch = samples[i:i + args.micro_batch]
        ids, attn, pos, _, P, R = collate(batch, pad_id, device)
        if use_adapter:
            logp, _ = token_logprobs(model, ids, attn, pos, P, R, args.temperature)
        else:
            with model.disable_adapter():
                logp, _ = token_logprobs(model, ids, attn, pos, P, R, args.temperature)
        for s, row in zip(batch, logp.cpu()):
            s[key] = row[:len(s["resp"])]


def pad_rows(rows, R, device):
    out = torch.zeros((len(rows), R))
    for j, r in enumerate(rows):
        out[j, :len(r)] = r
    return out.to(device)


def grpo_update(model, opt, scaler, samples, args, pad_id, device, rng):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    order = rng.permutation(len(samples))
    sums = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "clipfrac": 0.0, "entropy": 0.0}
    n_tok_total, grad_norms, skipped = 0, [], 0
    for m in range(0, len(order), args.mini_batch):
        mb = [samples[i] for i in order[m:m + args.mini_batch]]
        n_tok = sum(len(s["resp"]) for s in mb)  # token-mean over the whole mini-batch
        opt.zero_grad(set_to_none=True)
        for u in range(0, len(mb), args.micro_batch):
            batch = mb[u:u + args.micro_batch]
            ids, attn, pos, mask, P, R = collate(batch, pad_id, device)
            logp, entropy = token_logprobs(model, ids, attn, pos, P, R, args.temperature, with_entropy=True)
            old = pad_rows([s["old_logp"] for s in batch], R, device)
            ref = pad_rows([s["ref_logp"] for s in batch], R, device)
            adv = torch.tensor([s["adv"] for s in batch], device=device).unsqueeze(1)

            ratio = torch.exp(logp - old)
            pg = -torch.min(ratio * adv, ratio.clamp(1 - args.clip_ratio, 1 + args.clip_ratio) * adv)
            kl = (ref - logp).clamp(-20, 20)  # low_var_kl (k3), as in verl
            kld = (torch.exp(kl) - kl - 1).clamp(-10, 10)
            loss_tok = pg + args.kl_coef * kld
            loss = (loss_tok * mask).sum() / n_tok
            scaler.scale(loss).backward()

            with torch.no_grad():
                sums["loss"] += float((loss_tok * mask).sum())
                sums["pg_loss"] += float((pg * mask).sum())
                sums["kl"] += float((kld * mask).sum())
                sums["clipfrac"] += float((((ratio - 1).abs() > args.clip_ratio).float() * mask).sum())
                sums["entropy"] += float((entropy * mask).sum())
        scaler.unscale_(opt)
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
        n_tok_total += n_tok
        if all(s["adv"] == 0.0 for s in mb):
            # Only round-off KL gradients are left (log-probs recomputed with different padding);
            # Adam would rescale them into a full lr-sized step and make KL rise without any signal.
            skipped += 1
        else:
            scaler.step(opt)
        scaler.update()
    stats = {f"actor/{k}": v / n_tok_total for k, v in sums.items()}
    stats["actor/grad_norm"] = float(np.mean(grad_norms))
    stats["actor/optimizer_steps"] = len(grad_norms) - skipped
    stats["actor/skipped_steps"] = skipped
    return stats


# ---------------------------------------------------------------------- reporting
def plot_metrics(rows, val_rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    blue, orange, aqua, ink, grid = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df"
    it = np.array([r["iter"] for r in rows])

    def ma(y, w=3):
        y = np.asarray(y, dtype=float)
        return np.array([y[max(0, i - w + 1):i + 1].mean() for i in range(len(y))])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("reward/episode_score", "Train episode score (GRPO reward)"),
        ("train/excess_vs_bh_pct", "Excess return vs buy-and-hold (%)"),
        ("train/invalid_rate", "Invalid-format rate"),
        ("actor/loss", "Actor loss (pg + kl_coef * kl)"),
        ("actor/kl", "KL to reference (low_var_kl)"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        y = [r[key] for r in rows]
        ax.plot(it, y, "o", color=blue, alpha=0.35, ms=5, label="per iteration")
        ax.plot(it, ma(y), color=blue, lw=2, label="3-iter moving avg")
        if key == "train/excess_vs_bh_pct" and val_rows:
            ax.plot([v["iter"] for v in val_rows], [v["val/excess_vs_bh_pct"] for v in val_rows],
                    "s-", color=orange, lw=2, ms=8, label="val (fixed 2024 windows)")
        ax.set_title(title, color=ink, fontsize=11)
        ax.legend(fontsize=8, frameon=False)
    ax = axes.flat[5]
    bottom = np.zeros(len(it))
    for key, color, label in [("time/gen_s", blue, "generate"), ("time/logprob_s", orange, "old/ref log-prob"),
                              ("time/update_s", aqua, "update")]:
        y = np.array([r[key] for r in rows])
        ax.bar(it, y, bottom=bottom, color=color, edgecolor="white", linewidth=2, label=label)
        bottom += y
    ax.set_title("Seconds per iteration", color=ink, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    for ax in axes.flat:
        ax.set_xlabel("iteration", color=ink)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"plot -> {path}")


def fmt_hms(sec):
    h, rem = divmod(int(sec), 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def timing_report(rows, args, device):
    steady = rows[1:] if len(rows) > 1 else rows  # iteration 1 includes CUDA warm-up
    med = lambda k: float(np.median([r[k] for r in steady]))  # noqa: E731
    t_iter, t_gen, t_train = med("time/iter_s"), med("time/gen_s"), med("time/logprob_s") + med("time/update_s")
    n_traj = args.n_tasks * args.group_n
    n_samples = n_traj * args.window_len
    gen_call = t_gen / args.window_len  # one batched generate() for n_traj prompts
    full_traj, full_samples = FULL_TASKS * FULL_GROUP, FULL_TASKS * FULL_GROUP * FULL_WINDOW
    scale = full_traj / n_traj
    # generation of a larger batch costs between "same time" (GPU not saturated) and "linear in batch"
    full_gen_lo, full_gen_hi = gen_call * FULL_WINDOW * max(1.0, scale ** 0.5), gen_call * FULL_WINDOW * max(1.0, scale)
    full_train = t_train / n_samples * full_samples
    rep = {
        "device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "steady_iter_s": t_iter,
        "gen_s_per_iter": t_gen,
        "train_s_per_iter": t_train,
        "samples_per_iter": n_samples,
        "train_s_per_sample": t_train / n_samples,
        f"eta_this_config_{args.target_iters}_iters": fmt_hms(t_iter * args.target_iters),
        "full_config_iter_s_range": [full_gen_lo + full_train, full_gen_hi + full_train],
        f"full_config_{FULL_ITERS}_iters_range": [fmt_hms((full_gen_lo + full_train) * FULL_ITERS),
                                                  fmt_hms((full_gen_hi + full_train) * FULL_ITERS)],
    }
    print("\n================ TIMING ================")
    print(f"device                 : {rep['device']}")
    print(f"steady iteration       : {t_iter:.1f}s  (generate {t_gen:.1f}s, log-prob + update {t_train:.1f}s)")
    print(f"samples (env steps)    : {n_samples}/iter -> {t_train / n_samples:.2f}s training per sample")
    print(f"this config x {args.target_iters} iters: {rep[f'eta_this_config_{args.target_iters}_iters']}")
    print(f"verl config ({FULL_TASKS} tasks x {FULL_GROUP} rollouts x {FULL_WINDOW} days = {full_samples} samples/iter, "
          f"HF generate, LoRA, max_new_tokens={args.max_new_tokens}):")
    print(f"   per iteration       : {fmt_hms(full_gen_lo + full_train)} .. {fmt_hms(full_gen_hi + full_train)}")
    print(f"   x {FULL_ITERS} iterations    : {rep[f'full_config_{FULL_ITERS}_iters_range'][0]} .. "
          f"{rep[f'full_config_{FULL_ITERS}_iters_range'][1]}")
    print("   (rough: linear in samples for training; generation between sqrt and linear in batch size)")
    return rep


def learning_report(rows, val_rows):
    k = max(1, len(rows) // 3)
    first, last = rows[:k], rows[-k:]
    keys = ["reward/episode_score", "train/excess_vs_bh_pct", "train/win_rate", "train/invalid_rate",
            "train/resp_len", "actor/kl", "actor/loss", "actor/entropy"]
    rep = {key: {"first": float(np.mean([r[key] for r in first])), "last": float(np.mean([r[key] for r in last]))}
           for key in keys}
    print(f"\n================ LEARNING (mean of first {k} vs last {k} iters) ================")
    for key, v in rep.items():
        print(f"{key:26s}: {v['first']:+9.4f} -> {v['last']:+9.4f}  ({v['last'] - v['first']:+.4f})")
    if len(val_rows) >= 2:
        a, b = val_rows[0], val_rows[-1]
        print(f"val (fixed windows, iter {a['iter']} -> {b['iter']}):")
        for key in ["val/return_pct", "val/excess_vs_bh_pct", "val/win_rate", "val/invalid_rate"]:
            print(f"   {key:22s}: {a[key]:+8.3f} -> {b[key]:+8.3f}")
        rep["val"] = {"before": a, "after": b}
    print("Read it as: invalid_rate falling and KL rising = LoRA is changing the policy; "
          "episode score / val excess rising = it is learning the task. The GRPO loss itself "
          "stays near 0 by construction (group-normalised advantages), so do not expect it to decrease.")
    return rep


# ---------------------------------------------------------------------- main
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = pick_device(args.device)
    dtype = pick_dtype(device)
    model, tok, eos_ids = load_model(args, device, dtype)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"model {args.model} on {device} ({dtype}); LoRA r={args.lora_r}: "
          f"{n_train / 1e6:.2f}M trainable / {n_all / 1e6:.0f}M ({100 * n_train / n_all:.2f}%)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)

    sim_kwargs = dict(window_len=args.window_len, cost_bps=args.cost_bps, reward_type=args.reward_type)
    task = task_description(args.window_len, args.reward_type, args.cost_bps)
    markets = {t: load_market(args.data_dir, t) for t in ID_TICKERS}
    train_tasks = [(t, s) for t in ID_TICKERS for s in make_windows(markets[t], "train", args.window_len, 10)]
    val_all = [(t, s) for t in ID_TICKERS
               for s in make_windows(markets[t], "val", args.window_len, 10, stride=args.window_len)]
    val_tasks = [val_all[i] for i in np.unique(np.linspace(0, len(val_all) - 1, args.n_val).round().astype(int))]
    print(f"train tasks {len(train_tasks)}, val windows {len(val_tasks)} of {len(val_all)}; "
          f"{args.n_tasks} tasks x {args.group_n} rollouts x {args.window_len} days per iteration")

    def validate(it):
        t0 = time.time()
        sims = [TradingSim(markets[t], s, **sim_kwargs) for t, s in val_tasks]
        samples, infos = run_episodes(model, tok, eos_ids, sims, task, args, args.val_temperature, device)
        row = {"iter": it, **episode_stats(samples, infos, "val"), "time/val_s": time.time() - t0}
        print(f"[val iter {it}] return {row['val/return_pct']:+.2f}%  excess {row['val/excess_vs_bh_pct']:+.2f}%  "
              f"win {row['val/win_rate']:.2f}  invalid {row['val/invalid_rate']:.2f}  ({row['time/val_s']:.0f}s)")
        with open(os.path.join(args.out_dir, "val.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    for name in ("metrics.jsonl", "val.jsonl"):
        open(os.path.join(args.out_dir, name), "w").close()
    val_rows = [validate(0)] if args.n_val > 0 else []
    rows = []
    for it in range(1, args.iters + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sync(device)
        t0 = time.time()
        picks = rng.choice(len(train_tasks), size=args.n_tasks, replace=False)
        sims = [TradingSim(markets[train_tasks[i][0]], train_tasks[i][1], **sim_kwargs)
                for i in np.repeat(picks, args.group_n)]
        samples, infos = run_episodes(model, tok, eos_ids, sims, task, args, args.temperature, device)
        sync(device)
        t_gen = time.time() - t0

        adv_stats = grpo_advantages(samples, len(sims), args.group_n)
        fill_logprobs(model, samples, "old_logp", args, tok.pad_token_id, device, use_adapter=True)
        fill_logprobs(model, samples, "ref_logp", args, tok.pad_token_id, device, use_adapter=False)
        sync(device)
        t_lp = time.time() - t0 - t_gen

        actor_stats = grpo_update(model, opt, scaler, samples, args, tok.pad_token_id, device, rng)
        sync(device)
        t_iter = time.time() - t0

        n_gen_tok = sum(len(s["resp"]) for s in samples)
        row = {"iter": it, **adv_stats, **episode_stats(samples, infos, "train"), **actor_stats,
               "time/gen_s": t_gen, "time/logprob_s": t_lp, "time/update_s": t_iter - t_gen - t_lp,
               "time/iter_s": t_iter, "perf/gen_tokens_per_s": n_gen_tok / t_gen,
               "perf/peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
        rows.append(row)
        with open(os.path.join(args.out_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")

        steady = [r["time/iter_s"] for r in rows[1:]] or [t_iter]
        eta = float(np.mean(steady)) * (args.iters - it)
        print(f"[iter {it}/{args.iters}] score {row['reward/episode_score']:+.3f}  "
              f"excess {row['train/excess_vs_bh_pct']:+.2f}%  win {row['train/win_rate']:.2f}  "
              f"invalid {row['train/invalid_rate']:.2f}  len {row['train/resp_len']:.0f}  | "
              f"loss {row['actor/loss']:+.4f}  kl {row['actor/kl']:.5f}  clip {row['actor/clipfrac']:.3f}  "
              f"ent {row['actor/entropy']:.3f}  gnorm {row['actor/grad_norm']:.3f}  "
              f"steps {row['actor/optimizer_steps']}/{row['actor/optimizer_steps'] + row['actor/skipped_steps']}  | "
              f"{t_iter:.0f}s (gen {t_gen:.0f}s, lp {t_lp:.0f}s, upd {row['time/update_s']:.0f}s)  "
              f"mem {row['perf/peak_mem_gb']:.1f}GB  eta {fmt_hms(eta)}")

        if args.val_every and it % args.val_every == 0 and it != args.iters and args.n_val > 0:
            val_rows.append(validate(it))
        if args.save_every and it % args.save_every == 0:
            model.save_pretrained(os.path.join(args.out_dir, f"adapter_iter{it}"))

    if args.n_val > 0:
        val_rows.append(validate(args.iters))
    model.save_pretrained(os.path.join(args.out_dir, "adapter_final"))
    tok.save_pretrained(os.path.join(args.out_dir, "adapter_final"))

    summary = {"timing": timing_report(rows, args, device), "learning": learning_report(rows, val_rows)}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_metrics(rows, val_rows, os.path.join(args.out_dir, "metrics.png"))
    print(f"adapter -> {os.path.join(args.out_dir, 'adapter_final')}")


if __name__ == "__main__":
    main()
