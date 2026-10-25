#!/usr/bin/env bash
# Run a queue of training jobs one after another on a single GPU.
# A failed job is logged and the queue moves on; finished jobs are skipped on restart.
#
#   bash examples/fintrade/run_grid.sh main          # 7 algorithms x 3 seeds, log_return, K=2
#   bash examples/fintrade/run_grid.sh ablation_k    # K=4 for GiGPO / HGPO
#   bash examples/fintrade/run_grid.sh ablation_reward
#   bash examples/fintrade/run_grid.sh ablation_alpha
#   ALGOS="grpo hgpo" SEEDS="0" bash examples/fintrade/run_grid.sh main   # subset
set -uo pipefail

GRID=${1:-main}
LOG_DIR=${LOG_DIR:-logs/fintrade}
mkdir -p "$LOG_DIR"

jobs=()
case $GRID in
    main)
        for seed in ${SEEDS:-0 1 2}; do
            for algo in ${ALGOS:-grpo gigpo hgpo ppo rloo dapo gspo}; do
                jobs+=("$algo $seed REWARD=log_return K=2")
            done
        done ;;
    ablation_k)
        for seed in ${SEEDS:-0 1 2}; do
            for algo in ${ALGOS:-gigpo hgpo}; do jobs+=("$algo $seed REWARD=log_return K=4"); done
        done ;;
    ablation_reward)
        for reward in excess_terminal risk_adjusted; do
            for seed in ${SEEDS:-0 1}; do
                for algo in ${ALGOS:-grpo gigpo hgpo}; do jobs+=("$algo $seed REWARD=$reward K=2"); done
            done
        done ;;
    ablation_alpha)
        for alpha in 0.0 2.0; do
            for seed in ${SEEDS:-0 1}; do jobs+=("hgpo $seed REWARD=log_return K=4 ALPHA=$alpha"); done
        done ;;
    *) echo "unknown grid $GRID"; exit 1 ;;
esac

for job in "${jobs[@]}"; do
    read -r algo seed vars <<< "$job"
    name="${algo}_s${seed}_$(echo $vars | tr ' =' '_-')"
    if [ -f "$LOG_DIR/$name.done" ]; then
        echo "skip $name (done)"
        continue
    fi
    echo "[$(date '+%F %T')] start $name"
    if env $vars bash examples/fintrade/train.sh "$algo" "$seed" > "$LOG_DIR/$name.log" 2>&1; then
        touch "$LOG_DIR/$name.done"
        echo "[$(date '+%F %T')] done  $name"
    else
        echo "[$(date '+%F %T')] FAIL  $name (see $LOG_DIR/$name.log)"
    fi
done
