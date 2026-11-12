"""Pre-flight checks for running HGPO on 2x T4 (Kaggle) before spending GPU quota.

Run from the verl-agent root:

    python examples/hgpo_t4/preflight.py env          # versions, GPUs, RAM, disk, data files
    python examples/hgpo_t4/preflight.py vllm         # vLLM fp16 generation on a T4
    python examples/hgpo_t4/preflight.py grad         # fp16 vs fp32 gradients -> choose TRAIN_DTYPE
    python examples/hgpo_t4/preflight.py webshop-mem  # RAM per WebShop env worker -> does 8x8 + val fit?
    python examples/hgpo_t4/preflight.py all

Each check prints PASS / WARN / FAIL lines; send the full output back for diagnosis.
"""

import argparse
import os
import shutil
import subprocess
import sys

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
WEBSHOP_DIR = "agent_system/environments/env_package/webshop/webshop"
SAMPLE_PROMPT = (
    "You are an expert autonomous agent operating in the WebShop e-commerce environment.\n"
    "Your task is to: Find me machine wash men's dress shirts with polyester heathers, and price lower than 50.00 dollars.\n"
    "Your current observation is: 'Search'.\nYour admissible actions of the current situation are: [\n'search[<your query>]',\n'click[search]',\n].\n"
    "Now it's your turn to take one action for the current step.\n"
    "You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.\n"
    "Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."
)


def report(level, msg):
    print(f"[{level}] {msg}", flush=True)


def check_env():
    print("===== env =====")
    report("INFO", f"python {sys.version.split()[0]}")
    for pkg in ["torch", "vllm", "transformers", "tensordict", "ray", "flash_attn", "xformers", "pyserini", "alfworld"]:
        try:
            mod = __import__(pkg)
            report("INFO", f"{pkg} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            level = "INFO" if pkg in ("flash_attn", "alfworld", "pyserini") else "FAIL"
            report(level, f"{pkg} not importable ({type(e).__name__}: {e})")

    try:
        import torch

        n = torch.cuda.device_count()
        report("PASS" if n == 2 else "WARN", f"{n} CUDA device(s); the scripts expect 2")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            report("INFO", f"GPU{i}: {p.name}, {p.total_memory / 1e9:.1f} GB, sm{p.major}{p.minor}")
        if n and torch.cuda.get_device_capability(0) < (8, 0):
            report("INFO", "pre-Ampere GPU: use attn_implementation=sdpa, fp16 and vLLM V0/XFORMERS (train.sh does this)")
    except Exception as e:  # noqa: BLE001
        report("FAIL", f"torch.cuda: {e}")

    try:
        import vllm
        import torch

        report("INFO", f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1', '<unset>')} "
                       f"VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND', '<unset>')}")
        report("INFO", f"torch {torch.__version__} with vllm {vllm.__version__}: they must come from the same install "
                       f"(pip check below)")
    except Exception:  # noqa: BLE001
        pass
    out = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    broken = [l for l in out.stdout.splitlines() if any(k in l.lower() for k in ("torch", "vllm", "tensordict", "transformers", "ray"))]
    report("PASS" if not broken else "WARN", "pip check (core packages): " + ("ok" if not broken else "; ".join(broken)))

    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    report("INFO", f"RAM {mem:.1f} GB, {os.cpu_count()} CPUs")
    for path in ["/kaggle/working", "/tmp", os.path.expanduser("~")]:
        if os.path.exists(path):
            free = shutil.disk_usage(path).free / 1e9
            report("PASS" if path != "/kaggle/working" or free > 15 else "WARN", f"disk free at {path}: {free:.1f} GB")

    data = os.path.join(WEBSHOP_DIR, "data")
    for f in ["items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json"]:
        report("PASS" if os.path.exists(os.path.join(data, f)) else "FAIL", f"webshop data/{f}")
    idx = os.path.join(WEBSHOP_DIR, "search_engine", "indexes")
    report("PASS" if os.path.isdir(idx) and os.listdir(idx) else "FAIL", f"webshop search_engine/indexes")
    alf = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    report("PASS" if os.path.isdir(alf) else "INFO", f"alfworld data at {alf} (only needed for ALFWorld)")


def check_vllm():
    print("===== vllm (fp16 generation) =====")
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE_NO_HIS
    from agent_system.environments.env_package.webshop.projection import webshop_projection

    # first step of a WebShop episode, rendered exactly like WebshopEnvironmentManager does
    first_step = WEBSHOP_TEMPLATE_NO_HIS.format(
        task_description="i need a long clip-in hair extension which is natural looking, and price lower than 40.00 dollars",
        current_observation="'Search'",
        available_actions="'search[<your query>]',\n'click[search]',",
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = tok.apply_chat_template([{"role": "user", "content": first_step}], tokenize=False, add_generation_prompt=True)
    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.35, enforce_eager=True, max_model_len=4096,
              enable_sleep_mode=True, seed=0)
    n = 64
    for temperature in (1.0, 0.4):  # training rollouts / validation
        outs = llm.generate([prompt] * n, SamplingParams(temperature=temperature, max_tokens=512))
        texts = [o.outputs[0].text for o in outs]
        _, valids = webshop_projection(list(texts))
        cut = sum(o.outputs[0].finish_reason == "length" for o in outs)
        report("PASS" if all(t.strip() for t in texts) else "FAIL", f"vLLM fp16 produced text (T={temperature})")
        report("INFO", f"T={temperature}: {sum(valids)}/{n} valid (<think> + <action>, as webshop_projection), "
                       f"{sum('<action>' in t for t in texts)}/{n} have <action>, {len(set(texts))}/{n} distinct, "
                       f"{cut}/{n} cut at 512 tokens")
        shown = [t for t, v in zip(texts, valids) if v][:1] + [t for t, v in zip(texts, valids) if not v][:2]
        for i, t in enumerate(shown):  # prefixed so the grep filter in KAGGLE.md keeps them
            print("\n".join(f"[SAMPLE T={temperature} #{i}] {line}" for line in t[:800].splitlines()))
    llm.sleep(level=1)  # free_cache_engine=True in train.sh relies on sleep/wake_up
    llm.wake_up()
    report("PASS", "vLLM sleep / wake_up works")


def check_grad():
    print("===== grad (fp16 vs fp32) =====")
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = tok.apply_chat_template(
        [{"role": "user", "content": SAMPLE_PROMPT},
         {"role": "assistant", "content": "<think>I should search for the shirt first.</think><action>search[men's dress shirt polyester heathers]</action>"}],
        tokenize=False)
    batch = tok([text] * 2, return_tensors="pt").to("cuda")

    def grads(dtype, scale=1.0):
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, attn_implementation="sdpa").cuda()
        model.gradient_checkpointing_enable()
        logits = model(**batch).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, batch["input_ids"][:, 1:, None]).squeeze(-1)
        loss = -logp.mean() * 1e-2  # PPO-sized loss: advantages ~1, many tokens
        (loss * scale).backward()   # scale > 1: loss scaling as in dp_actor (ShardedGradScaler)
        g = torch.cat([p.grad.float().flatten() for p in model.parameters() if p.grad is not None]) / scale
        del model
        torch.cuda.empty_cache()
        return loss.item(), g

    def compare(name, g16, g32):
        finite = bool(torch.isfinite(g16).all())
        cos = torch.nn.functional.cosine_similarity(g32, g16, dim=0).item() if finite else float("nan")
        underflow = ((g16 == 0) & (g32 != 0)).float().mean().item()
        report("INFO", f"{name}: finite={finite} grad cosine vs fp32={cos:.4f}, underflowed entries={underflow:.2%}")
        return finite and cos > 0.99 and underflow < 0.05

    l32, g32 = grads(torch.float32)
    l16, g16 = grads(torch.float16)
    report("INFO", f"loss fp32={l32:.6f} fp16={l16:.6f}")
    compare("fp16, no loss scaling", g16, g32)
    _, g16s = grads(torch.float16, scale=2.0**10)
    scaled_ok = compare("fp16, loss scale 1024 (what TRAIN_DTYPE=fp16 does)", g16s, g32)

    # speed of one forward+backward on a WebShop-sized micro-batch (2 x 2048 tokens, gradient checkpointing)
    ids = torch.randint(0, 150000, (2, 2048), device="cuda")
    times = {}
    for name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, attn_implementation="sdpa").cuda()
        model.gradient_checkpointing_enable()
        for i in range(3):  # first iteration is warm-up
            if i == 1:
                torch.cuda.synchronize(); t0 = time.time()
            model(input_ids=ids, logits_to_keep=513).logits.float().mean().backward()
        torch.cuda.synchronize()
        times[name] = (time.time() - t0) / 2
        del model
        torch.cuda.empty_cache()
    report("INFO", f"forward+backward 2x2048 tokens: fp32 {times['fp32']:.2f}s, fp16 {times['fp16']:.2f}s "
                   f"({times['fp32'] / times['fp16']:.1f}x)")
    if scaled_ok:
        report("PASS", "use TRAIN_DTYPE=fp16 (loss scaling keeps the gradients close to fp32)")
    else:
        report("WARN", "use TRAIN_DTYPE=fp32 (slower but safe); fp16 gradients deviate from fp32 even with loss scaling")


WORKER_SNIPPET = r"""
import os, sys
from agent_system.environments.env_package.webshop.envs import WebshopWorker
d = "agent_system/environments/env_package/webshop/webshop/data"
kwargs = dict(observation_mode="text", num_products=None, human_goals=False,
              file_path=os.path.join(d, "items_shuffle_1000.json"), attr_path=os.path.join(d, "items_ins_v2_1000.json"))
w = WebshopWorker([int(s) for s in sys.argv[1].split(",")], kwargs)  # the same actor class train.sh uses
w.reset(list(range(len(w.envs))))
w.step(["search[men dress shirt]"] * len(w.envs))
# VmHWM = peak RSS of this process; ru_maxrss would inherit the (large) parent's peak across fork/exec
print([l.split()[1] for l in open("/proc/self/status") if l.startswith("VmHWM")][0])
"""


def _worker_rss_gb(seeds):
    out = subprocess.run([sys.executable, "-c", WORKER_SNIPPET, ",".join(map(str, seeds))],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("could not start a WebShop worker:\n" + out.stderr[-2000:])
    return int(out.stdout.strip().splitlines()[-1]) / 1e6  # VmHWM is in kB


def check_webshop_mem(k=int(os.environ.get("ENVS_PER_WORKER", 16)), group_n=8, train_tasks=8, val_tasks=32, eval_tasks=128):
    """Peak RAM of the env actors, packed k envs per actor as in train.sh (env.webshop.envs_per_worker)."""
    print("===== webshop-mem =====")
    one = _worker_rss_gb([0])
    group = _worker_rss_gb([0] * k)                # a training group: k envs, one seed, one shared server
    mixed = _worker_rss_gb(list(range(1000, 1000 + k)))  # validation: k envs, k seeds, k servers
    report("INFO", f"actor RSS: 1 env {one:.2f} GB | {k} envs same seed {group:.2f} GB | {k} envs {k} seeds {mixed:.2f} GB "
                   f"(JAVA_TOOL_OPTIONS={os.environ.get('JAVA_TOOL_OPTIONS', '<unset>')})")
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    actors = lambda n: -(-n // k)  # noqa: E731
    train = actors(train_tasks * group_n) * (group if group_n >= k else mixed)
    per_server = max(mixed - one, 0.0) / max(k - 1, 1)  # extra RAM of one more seed (= one more WebShop server)
    eval_k = 32                                            # train.sh packs 32 envs per actor when EVAL=1
    eval_total = -(-eval_tasks // eval_k) * (one + (eval_k - 1) * per_server)
    for name, total in [(f"training: {train_tasks}x{group_n} train + {val_tasks} val envs", train + actors(val_tasks) * mixed),
                        (f"final eval: {eval_tasks} val envs, {eval_k} per actor (no train envs)", eval_total)]:
        level = "PASS" if total < 0.4 * mem else ("WARN" if total < 0.5 * mem else "FAIL")
        report(level, f"{name} ~ {total:.1f} GB of {mem:.1f} GB RAM (the 2 GPU workers, TaskRunner and Ray need ~14 GB more)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["env", "vllm", "grad", "webshop-mem", "all"])
    args = parser.parse_args()
    checks = {"env": check_env, "vllm": check_vllm, "grad": check_grad, "webshop-mem": check_webshop_mem}
    for name, fn in checks.items():
        if args.check in (name, "all"):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                report("FAIL", f"{name}: {type(e).__name__}: {e}")
    print("PREFLIGHT DONE")


if __name__ == "__main__":
    main()
