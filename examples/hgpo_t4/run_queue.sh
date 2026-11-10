#!/usr/bin/env bash
# Run the WebShop (or ALFWorld) grid one run after another inside ONE Kaggle session, within a
# shared time budget. Re-running the same command in the next session skips finished runs and
# resumes the unfinished one from its latest checkpoint.
#
#   bash examples/hgpo_t4/run_queue.sh webshop                 # grpo gigpo hgpo
#   ALGOS="gigpo hgpo" bash examples/hgpo_t4/run_queue.sh alfworld
#   SESSION_BUDGET_MIN=660 (default, 11h)   EPOCHS=60   TRAIN_DTYPE=fp32
set -uo pipefail

ENV=${1:?usage: run_queue.sh <webshop|alfworld>}
ALGOS=${ALGOS:-grpo gigpo hgpo}
SESSION_BUDGET_MIN=${SESSION_BUDGET_MIN:-660}
EPOCHS=${EPOCHS:-60}
if [ -d /kaggle/working ]; then CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-/kaggle/working/checkpoints}; else CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-$HOME/checkpoints}; fi
LOG_DIR=${LOG_DIR:-$CHECKPOINTS_DIR/logs}
export CHECKPOINTS_DIR EPOCHS
mkdir -p "$LOG_DIR"
deadline=$(( $(date +%s) + SESSION_BUDGET_MIN * 60 ))

for algo in $ALGOS; do
    name="${ENV}_${algo}"
    if [ -f "$LOG_DIR/$name.done" ]; then
        echo "skip $name (done)"
        continue
    fi
    left=$(( (deadline - $(date +%s)) / 60 ))
    if [ $left -lt 30 ]; then
        echo "less than 30 min left in this session; stop here and rerun next session"
        break
    fi
    echo "[$(date '+%F %T')] start $name with ${left} min budget"
    TIME_BUDGET="${left}m" bash examples/hgpo_t4/train.sh "$ENV" "$algo" >> "$LOG_DIR/$name.log" 2>&1
    status=$?
    # train.sh prints this marker when it was cut by the time budget (resume next session)
    if tail -n 5 "$LOG_DIR/$name.log" | grep -q "TIME BUDGET REACHED"; then
        echo "[$(date '+%F %T')] paused $name (time budget); rerun next session"
        break
    elif [ $status -eq 0 ]; then
        touch "$LOG_DIR/$name.done"
        echo "[$(date '+%F %T')] done  $name"
    else
        echo "[$(date '+%F %T')] FAIL  $name (exit $status, see $LOG_DIR/$name.log)"
        break
    fi
done
echo "queue status:"; ls "$LOG_DIR"/*.done 2>/dev/null || echo "  no finished runs yet"
