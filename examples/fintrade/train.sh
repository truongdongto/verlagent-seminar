#!/usr/bin/env bash
# Train one RL algorithm on the single-stock trading env with ONE GPU.
#
#   bash examples/fintrade/train.sh <algo> [seed] [extra hydra overrides...]
#   algo: ppo | rloo | grpo | dapo | gspo | gigpo | hgpo
#
# Environment variables (all optional):
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct   REWARD=log_return|excess_terminal|risk_adjusted
#   K=2 (history_length, also HGPO's deepest context level)   ALPHA=1.0 (HGPO weight exponent)
#   EPOCHS=160   GPU_MEM=0.4 (vLLM memory fraction)   CHECKPOINTS_DIR=$HOME/checkpoints
#   SMOKE=1  -> Qwen2.5-0.5B, 5-day episodes, 2 iterations, console logging only
#
# Every algorithm goes through recipe.hgpo.main_hgpo so that they share one trainer.
set -euo pipefail
set -x

ALGO=${1:?usage: train.sh <ppo|rloo|grpo|dapo|gspo|gigpo|hgpo> [seed] [overrides...]}
SEED=${2:-0}
shift $(( $# >= 2 ? 2 : 1 ))

export HYDRA_FULL_ERROR=1  # set VLLM_ATTENTION_BACKEND yourself only if vLLM picks a broken default

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
REWARD=${REWARD:-log_return}
K=${K:-2}
ALPHA=${ALPHA:-1.0}
EPOCHS=${EPOCHS:-160}
GPU_MEM=${GPU_MEM:-0.4}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-$HOME/checkpoints}

train_data_size=16   # tasks per iteration
group_size=8         # rollouts per task
val_data_size=96     # = number of non-overlapping 2024 windows for the 8 ID tickers
window_len=20
test_freq=10
logger="['console','wandb']"
[ -z "${WANDB_API_KEY:-}" ] && logger="['console']"

if [ "${SMOKE:-0}" = "1" ]; then
    MODEL=Qwen/Qwen2.5-0.5B-Instruct
    EPOCHS=2
    train_data_size=4
    val_data_size=8
    window_len=5
    test_freq=1
    logger="['console']"
fi

# ---------------- algorithm-specific settings ----------------
algo_args=()
case $ALGO in
    ppo)   # critic-based; no env grouping, same number of episodes per iteration (16 x 8)
        algo_args+=(algorithm.adv_estimator=gae
                    critic.optim.lr=1e-5
                    critic.model.path=$MODEL
                    critic.model.use_remove_padding=True
                    critic.model.enable_gradient_checkpointing=True
                    critic.ppo_micro_batch_size_per_gpu=4
                    critic.model.fsdp_config.param_offload=True
                    critic.model.fsdp_config.optimizer_offload=True
                    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True)
        train_data_size=$(( train_data_size * group_size ))
        group_size=1 ;;
    rloo)  algo_args+=(algorithm.adv_estimator=rloo) ;;
    grpo)  algo_args+=(algorithm.adv_estimator=grpo) ;;
    dapo)  # clip-higher + dynamic sampling; token-mean loss is the default loss_agg_mode
        algo_args+=(algorithm.adv_estimator=grpo
                    actor_rollout_ref.actor.clip_ratio_low=0.2
                    actor_rollout_ref.actor.clip_ratio_high=0.28
                    algorithm.filter_groups.enable=True
                    algorithm.filter_groups.max_num_gen_batches=10) ;;
    gspo)  # sequence-level importance ratio
        algo_args+=(algorithm.adv_estimator=grpo
                    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
                    actor_rollout_ref.actor.clip_ratio_low=0.0003
                    actor_rollout_ref.actor.clip_ratio_high=0.0004) ;;
    gigpo) algo_args+=(algorithm.adv_estimator=gigpo
                       algorithm.gigpo.step_advantage_w=1.0
                       algorithm.gigpo.mode=mean_std_norm) ;;
    hgpo)  algo_args+=(algorithm.adv_estimator=hgpo
                       algorithm.hgpo.mode=mean_std_norm
                       algorithm.hgpo.weight_type=length
                       algorithm.hgpo.length_weight_alpha=$ALPHA
                       algorithm.hgpo.base_group=False) ;;
    *) echo "unknown algo $ALGO"; exit 1 ;;
esac

project_name="fintrade_$(basename $MODEL)"
experiment_name="${ALGO}_${REWARD}_k${K}_s${SEED}"
[ "$ALGO" = "hgpo" ] && experiment_name="${ALGO}_a${ALPHA}_${REWARD}_k${K}_s${SEED}"
[ "${SMOKE:-0}" = "1" ] && experiment_name="smoke_${experiment_name}"

# The parquet files only set the modality and the number of tasks; the env produces the prompts.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m recipe.hgpo.main_hgpo \
    "${algo_args[@]}" \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=fintrade \
    env.seed=$SEED \
    env.max_steps=$window_len \
    env.history_length=$K \
    env.rollout.n=$group_size \
    env.trading.window_len=$window_len \
    env.trading.val_window_len=$window_len \
    env.trading.reward_type=$REWARD \
    trainer.critic_warmup=0 \
    trainer.logger=$logger \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$EPOCHS \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CHECKPOINTS_DIR}/${project_name}/${experiment_name}" \
    trainer.val_before_train=True "$@"
