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

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = tok.apply_chat_template([{"role": "user", "content": SAMPLE_PROMPT}], tokenize=False, add_generation_prompt=True)
    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.35, enforce_eager=True, max_model_len=4096,
              enable_sleep_mode=True)
    outs = llm.generate([prompt] * 8, SamplingParams(temperature=1.0, max_tokens=512, seed=0))
    texts = [o.outputs[0].text for o in outs]
    ok = sum("<action>" in t and "</action>" in t for t in texts)
    think = sum("<think>" in t for t in texts)
    cut = sum(o.outputs[0].finish_reason == "length" for o in outs)
    report("PASS" if all(t.strip() for t in texts) else "FAIL", "vLLM fp16 produced text")
    report("INFO", f"{ok}/8 samples contain <action>...</action>, {think}/8 contain <think>, "
                   f"{cut}/8 cut at 512 tokens (low format rate is expected before training)")
    for i in range(2):  # prefixed so the grep filter in KAGGLE.md keeps them
        print("\n".join(f"[SAMPLE{i}] {line}" for line in texts[i][:1200].splitlines()))
    llm.sleep(level=1)  # free_cache_engine=True in train.sh relies on sleep/wake_up
    llm.wake_up()
    report("PASS", "vLLM sleep / wake_up works")


def check_grad():
    print("===== grad (fp16 vs fp32) =====")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = tok.apply_chat_template(
        [{"role": "user", "content": SAMPLE_PROMPT},
         {"role": "assistant", "content": "<think>I should search for the shirt first.</think><action>search[men's dress shirt polyester heathers]</action>"}],
        tokenize=False)
    batch = tok([text] * 2, return_tensors="pt").to("cuda")

    def grads(dtype):
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, attn_implementation="sdpa").cuda()
        model.gradient_checkpointing_enable()
        logits = model(**batch).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, batch["input_ids"][:, 1:, None]).squeeze(-1)
        loss = -logp.mean() * 1e-2  # PPO-sized loss: advantages ~1, many tokens
        loss.backward()
        g = torch.cat([p.grad.float().flatten() for p in model.parameters() if p.grad is not None])
        del model
        torch.cuda.empty_cache()
        return loss.item(), g

    l32, g32 = grads(torch.float32)
    l16, g16 = grads(torch.float16)
    finite = bool(torch.isfinite(g16).all())
    cos = torch.nn.functional.cosine_similarity(g32, g16, dim=0).item() if finite else float("nan")
    underflow = ((g16 == 0) & (g32 != 0)).float().mean().item()
    report("INFO", f"loss fp32={l32:.6f} fp16={l16:.6f}")
    report("PASS" if finite else "FAIL", f"fp16 gradients finite: {finite}")
    report("INFO", f"grad cosine(fp16, fp32)={cos:.4f}, underflowed entries={underflow:.2%}")
    if finite and cos > 0.99 and underflow < 0.05:
        report("PASS", "use TRAIN_DTYPE=fp16")
    else:
        report("WARN", "use TRAIN_DTYPE=fp32 (slower but safe); fp16 gradients deviate from fp32")


WORKER_SNIPPET = r"""
import os, sys, gym
sys.path.insert(0, os.path.join(os.getcwd(), "agent_system/environments/env_package/webshop/webshop"))
from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  registers WebAgentTextEnv-v0
d = "agent_system/environments/env_package/webshop/webshop/data"
env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None, human_goals=False, seed=0,
               file_path=os.path.join(d, "items_shuffle_1000.json"), attr_path=os.path.join(d, "items_ins_v2_1000.json"))
env.reset()
# VmHWM = peak RSS of this process; ru_maxrss would inherit the (large) parent's peak across fork/exec
print([l.split()[1] for l in open("/proc/self/status") if l.startswith("VmHWM")][0])
"""


def check_webshop_mem(train_workers=64, val_workers=32):
    print("===== webshop-mem =====")
    out = subprocess.run([sys.executable, "-c", WORKER_SNIPPET], capture_output=True, text=True)
    if out.returncode != 0:
        report("FAIL", "could not start a WebShop env:\n" + out.stderr[-2000:])
        return
    rss_gb = int(out.stdout.strip().splitlines()[-1]) / 1e6  # VmHWM is in kB
    total = rss_gb * (train_workers + val_workers)
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    report("INFO", f"one WebShop worker peaks at {rss_gb:.2f} GB (Python + JVM), "
                   f"JAVA_TOOL_OPTIONS={os.environ.get('JAVA_TOOL_OPTIONS', '<unset>')}")
    level = "PASS" if total < 0.6 * mem else ("WARN" if total < 0.85 * mem else "FAIL")
    report(level, f"{train_workers} train + {val_workers} val workers ~ {total:.1f} GB of {mem:.1f} GB RAM "
                  f"(the trainer itself needs ~8-10 GB more); if WARN/FAIL try JAVA_TOOL_OPTIONS=-Xmx256m")


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
