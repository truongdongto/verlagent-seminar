"""Collect results of the T4 reproduction from the console logs written by run_queue.sh / eval_all.sh.

    python examples/hgpo_t4/summarize.py --log_dir /kaggle/working/checkpoints/logs --out results/hgpo_t4

Writes
    metrics_<env>_<algo>.csv     every logged metric per training step
    table1_<env>.csv             final evaluation, mean +- std over eval seeds (paper Table 1 format)
    table2_<env>.csv             step utilization / mean group size per context level (paper Table 2)
    fig2_<env>.png               |advantage - Oracle advantage| over training (paper Fig. 2a-b)
and prints the tables.
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

STEP_RE = re.compile(r"^step:(\d+) - ")


def parse_log(path):
    """verl console logger lines: 'step:12 - key:val - key:val ...' (keys may contain spaces)."""
    rows = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = STEP_RE.match(line.strip())
            if not m:
                continue
            step = int(m.group(1))
            row = rows.setdefault(step, {"step": step})
            for part in line.strip().split(" - ")[1:]:
                key, _, val = part.rpartition(":")
                try:
                    row[key] = float(val)
                except ValueError:
                    pass
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r["step"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="/kaggle/working/checkpoints/logs")
    parser.add_argument("--out", default="results/hgpo_t4")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 200)

    # ---------------- training runs ----------------
    train = {}
    for path in sorted(glob.glob(os.path.join(args.log_dir, "*_*.log"))):
        name = os.path.basename(path)[:-4]
        if name.startswith("eval_"):
            continue
        env, algo = name.split("_", 1)
        df = parse_log(path)
        if df.empty:
            print(f"{name}: no step lines yet")
            continue
        df.to_csv(os.path.join(args.out, f"metrics_{env}_{algo}.csv"), index=False)
        train[(env, algo)] = df
        last = df.dropna(subset=[c for c in df.columns if c == "val/success_rate"] or ["step"]).iloc[-1]
        print(f"{name}: {int(df['step'].max())} steps logged; last val/success_rate="
              f"{last.get('val/success_rate', float('nan')):.3f}")

    for env in sorted({e for e, _ in train}):
        # Table 2: utilization per context level (averaged over training steps)
        rows = []
        for (e, algo), df in train.items():
            if e != env:
                continue
            row = {"algo": algo}
            for k in range(5):
                for stat in ("utilization", "mean_group_size"):
                    col = f"context_groups/{k}-context/{stat}"
                    if col in df:
                        row[f"{k}-C {stat}"] = df[col].mean()
            for col in ("adv_bias/oracleK/coverage", "adv_bias/oraclefull/coverage"):
                if col in df:
                    row[col.split("/")[1] + " coverage"] = df[col].mean()
            rows.append(row)
        t2 = pd.DataFrame(rows).set_index("algo")
        t2.to_csv(os.path.join(args.out, f"table2_{env}.csv"))
        print(f"\n=== {env}: hierarchy-of-groups statistics (mean over training) ===\n{t2.round(3).to_string()}")

        # Fig. 2: advantage error w.r.t. the Oracle, each algorithm on its own rollouts
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            for ax, oracle in zip(axes, ("oracleK", "oraclefull")):
                for (e, algo), df in sorted(train.items()):
                    if e != env:
                        continue
                    for est, style in (("traj", ":"), ("step0", "--"), ("hgpo", "-")):
                        col = f"adv_bias/{oracle}/{est}_mae"
                        if col in df:
                            ax.plot(df["step"], df[col].rolling(5, min_periods=1).mean(), style,
                                    label=f"{algo} run: {est}")
                ax.set_title(f"{env}: |A - A_{oracle}| (MAE, 5-step mean)")
                ax.set_xlabel("training step")
            axes[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"fig2_{env}.png"), dpi=150)
            plt.close(fig)
        except ImportError:
            print("matplotlib not installed: skipped fig2")

    # ---------------- final evaluation (Table 1) ----------------
    evals = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.log_dir, "eval_*.log"))):
        m = re.match(r"eval_(\w+?)_(\w+?)_seed(\d+)(?:_(eval_\w+))?\.log$", os.path.basename(path))
        if not m:
            continue
        env, algo, seed, split = m.groups()
        df = parse_log(path)
        if df.empty:
            continue
        row = df.iloc[-1]
        metrics = {k: row[k] for k in row.index if k.startswith("val/")}
        evals[(env, split or "-", algo)].append(metrics)
    if evals:
        rows = []
        for (env, split, algo), runs in sorted(evals.items()):
            df = pd.DataFrame(runs)
            row = {"env": env, "split": split, "algo": algo, "seeds": len(df)}
            for col in df.columns:
                short = col.replace("val/", "").replace(" (not success_rate)", "")
                row[short] = f"{100 * df[col].mean():.2f} ± {100 * df[col].std(ddof=0):.2f}" if "success" in col or "score" in col \
                    else f"{df[col].mean():.3f}"
            rows.append(row)
        t1 = pd.DataFrame(rows)
        for env, part in t1.groupby("env"):
            part.to_csv(os.path.join(args.out, f"table1_{env}.csv"), index=False)
        print(f"\n=== final evaluation (x100, mean ± std over seeds) ===\n{t1.to_string(index=False)}")


if __name__ == "__main__":
    main()
