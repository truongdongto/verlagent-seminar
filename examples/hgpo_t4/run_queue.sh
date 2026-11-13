#!/usr/bin/env bash
# Run the WebShop (or ALFWorld) grid one run after another inside ONE Kaggle session, within a
# shared time budget. Re-running the same command in the next session skips finished runs and
# resumes the unfinished one from its latest checkpoint.
#
#   bash examples/hgpo_t4/run_queue.sh webshop                 # grpo gigpo hgpo
#   ALGOS="gigpo hgpo" bash examples/hgpo_t4/run_queue.sh alfworld
#   SESSION_BUDGET_MIN=660 (default, 11h)   EPOCHS=60   TRAIN_DTYPE=fp16
set -uo pipefail

ENV=${1:?usage: run_queue.sh <webshop|alfworld>}
ALGOS=${ALGOS:-grpo gigpo hgpo}
REWARD=${REWARD:-binary}; export REWARD
rtag=""; [ "$REWARD" != "binary" ] && rtag="_$REWARD"  # keeps logs of different rewards apart
SESSION_BUDGET_MIN=${SESSION_BUDGET_MIN:-660}
EPOCHS=${EPOCHS:-60}
if [ -d /kaggle/working ]; then CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-/kaggle/working/checkpoints}; else CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-$HOME/checkpoints}; fi
LOG_DIR=${LOG_DIR:-$CHECKPOINTS_DIR/logs}
export CHECKPOINTS_DIR EPOCHS
mkdir -p "$LOG_DIR"
deadline=$(( $(date +%s) + SESSION_BUDGET_MIN * 60 ))

for algo in $ALGOS; do
    name="${ENV}_${algo}${rtag}"
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
    touch "$LOG_DIR/$name.log"
    # one short line per iteration on stdout (visible in Kaggle's version log); stops when this script exits.
    # grep/read instead of awk: Ubuntu's mawk buffers its pipe input, which held every line until the end
    tail -n 0 -F --pid=$$ "$LOG_DIR/$name.log" 2>/dev/null | grep --line-buffered -E "step:[0-9]+ - " |
        while IFS= read -r line; do
            echo "$(date +%T)  $name  $(echo "$line" | grep -oE "step:[0-9]+ |(episode|val)/(success_rate|webshop_task_score \(not success_rate\)):[-0-9.]+|episode/valid_action_ratio:[0-9.]+|actor/loss_scale:[0-9.]+|timing_s/step:[0-9.]+" |
                sed 's/webshop_task_score (not success_rate)/task_score/' | tr '\n' ' ')"
        done &
    TIME_BUDGET="${left}m" bash examples/hgpo_t4/train.sh "$ENV" "$algo" >> "$LOG_DIR/$name.log" 2>&1
    status=$?
    # train.sh prints this marker when it was cut by the time budget (resume next session)
    if tail -n 5 "$LOG_DIR/$name.log" | grep -q "TIME BUDGET REACHED"; then
        echo "[$(date '+%F %T')] paused $name (time budget); rerun next session"
        break
    elif [ $status -eq 0 ]; then
        touch "$LOG_DIR/$name.done"
        # the finished run only needs its weights for evaluation: drop the optimizer state (~4 GB of the 20 GB output)
        find "$CHECKPOINTS_DIR/hgpo_t4_$ENV" -path "*/${algo}_k*_s${SEED:-0}${rtag:+_r$REWARD}/*" -name "optim_world_size_*.pt" -delete 2>/dev/null
        echo "[$(date '+%F %T')] done  $name"
    else
        echo "[$(date '+%F %T')] FAIL  $name (exit $status, see $LOG_DIR/$name.log)"
        break
    fi
done
echo "queue status:"; ls "$LOG_DIR"/*.done 2>/dev/null || echo "  no finished runs yet"
du -sh "$CHECKPOINTS_DIR" /kaggle/working 2>/dev/null
