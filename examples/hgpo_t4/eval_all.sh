#!/usr/bin/env bash
# Final evaluation as in the paper: the latest checkpoint of each run, 3 evaluation seeds,
# validation temperature 0.4. ALFWorld is evaluated in- and out-of-distribution.
#
#   bash examples/hgpo_t4/eval_all.sh webshop
#   ALGOS="gigpo hgpo" bash examples/hgpo_t4/eval_all.sh alfworld
set -uo pipefail

ENV=${1:?usage: eval_all.sh <webshop|alfworld>}
ALGOS=${ALGOS:-grpo gigpo hgpo}
SEEDS=${SEEDS:-123 456 789}
if [ -d /kaggle/working ]; then CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-/kaggle/working/checkpoints}; else CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-$HOME/checkpoints}; fi
LOG_DIR=${LOG_DIR:-$CHECKPOINTS_DIR/logs}
export CHECKPOINTS_DIR
mkdir -p "$LOG_DIR"

splits="none"
[ "$ENV" = "alfworld" ] && splits="eval_in_distribution eval_out_of_distribution"

for algo in $ALGOS; do
    for split in $splits; do
        for seed in $SEEDS; do
            tag="eval_${ENV}_${algo}_seed${seed}"
            [ "$split" != "none" ] && tag="${tag}_${split}"
            if [ -f "$LOG_DIR/$tag.done" ]; then echo "skip $tag"; continue; fi
            echo "[$(date '+%F %T')] $tag"
            if [ "$split" != "none" ]; then export ALFWORLD_EVAL=$split; fi
            if EVAL=1 EVAL_SEED=$seed bash examples/hgpo_t4/train.sh "$ENV" "$algo" > "$LOG_DIR/$tag.log" 2>&1; then
                touch "$LOG_DIR/$tag.done"
            else
                echo "FAIL $tag (see $LOG_DIR/$tag.log)"
            fi
        done
    done
done
