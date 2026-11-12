#!/usr/bin/env bash
# Reproduce HGPO (ICLR 2026) on WebShop / ALFWorld with Qwen2.5-0.5B-Instruct on 2x T4 (Kaggle).
#
#   bash examples/hgpo_t4/train.sh <webshop|alfworld> <grpo|gigpo|hgpo> [extra hydra overrides...]
#
# Training keeps the paper's group size (8), history length (K=2), max steps, reward, gamma,
# alpha and normalization mode; it scales down the tasks per iteration (16 -> 8), the number of
# iterations (160 -> 60) and the model (1.5B -> 0.5B). T4 has no FlashAttention 2 and no bf16,
# so attention runs with SDPA, vLLM in fp16 (V0 engine, xformers) and FSDP with TRAIN_DTYPE.
#
# Environment variables (all optional):
#   SEED=0                 training seed (the paper trains once and evaluates with 3 seeds)
#   EPOCHS=60              training iterations
#   TRAIN_DTYPE=fp32       FSDP compute dtype: fp32 (safe, default) or fp16 (~3x faster on T4; dp_actor adds dynamic
#                          loss scaling, see preflight.py grad)
#   GPU_MEM=0.35           vLLM memory fraction per T4
#   TIME_BUDGET=11h        stop before Kaggle's 12h limit; the run resumes from the last checkpoint
#   SAVE_FREQ=5            checkpoint every N iterations (only the latest is kept)
#   ENVS_PER_WORKER=16     WebShop envs per Ray actor (a group of 8 shares one server: ~1.3 GB per actor instead of
#                          1.2 GB per env); 32 when EVAL=1 (128 val envs -> 4 actors)
#   RAY_CPUS=8             Ray CPU slots (each slot may get an idle prestarted Python worker, ~0.1 GB RAM)
#   CHECKPOINTS_DIR        default /kaggle/working/checkpoints on Kaggle, else $HOME/checkpoints
#   EVAL=1 EVAL_SEED=123   evaluate the latest checkpoint of this run instead of training
#   SMOKE=1                2 iterations, 2 tasks x 8 rollouts, 4 validation tasks, no checkpoint
#   WANDB_API_KEY          log to W&B in addition to the console
set -euo pipefail
set -x

ENV=${1:?usage: train.sh <webshop|alfworld> <grpo|gigpo|hgpo> [overrides...]}
ALGO=${2:?usage: train.sh <webshop|alfworld> <grpo|gigpo|hgpo> [overrides...]}
shift 2

ulimit -u 65536 || true
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=${VLLM_USE_V1:-0}                               # V0 engine runs on Turing (sm75)
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}  # FlashAttention needs sm80+

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-60}
TRAIN_DTYPE=${TRAIN_DTYPE:-fp32}
GPU_MEM=${GPU_MEM:-0.35}
TIME_BUDGET=${TIME_BUDGET:-11h}
SAVE_FREQ=${SAVE_FREQ:-5}
if [ -d /kaggle/working ]; then
    CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-/kaggle/working/checkpoints}
else
    CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-$HOME/checkpoints}
fi

group_size=8      # paper
history_length=2  # paper, K=2
train_data_size=8 # paper: 16
test_freq=10
logger="['console','wandb']"
[ -z "${WANDB_API_KEY:-}" ] && logger="['console']"

# ---------------- environment settings (from recipe/hgpo/run_qwen2.5_1.5b_*_train.sh) ----------------
case $ENV in
    webshop)
        env_args=(env.env_name=Webshop env.webshop.use_small=True env.webshop.human_goals=False
                  env.webshop.envs_per_worker=${ENVS_PER_WORKER:-$([ "${EVAL:-0}" = 1 ] && echo 32 || echo 16)})
        max_steps=15          # paper Table 7 setting (Table 1 uses 30)
        max_prompt_length=4096
        mode=mean_norm
        mini_batch=64
        val_data_size=32      # during training; final evaluation uses EVAL_VAL_SIZE
        eval_val_size=128 ;;
    alfworld)
        env_args=(env.env_name=alfworld/AlfredTWEnv env.alfworld.eval_dataset=${ALFWORLD_EVAL:-eval_in_distribution})
        max_steps=50
        max_prompt_length=2048
        mode=mean_std_norm
        mini_batch=256
        val_data_size=32
        eval_val_size=128 ;;
    *) echo "unknown env $ENV"; exit 1 ;;
esac
eval_val_size=${EVAL_VAL_SIZE:-$eval_val_size}

# ---------------- algorithm settings (paper defaults) ----------------
case $ALGO in
    grpo)  algo_args=(algorithm.adv_estimator=grpo) ;;
    gigpo) algo_args=(algorithm.adv_estimator=gigpo
                      algorithm.gigpo.step_advantage_w=1.0
                      algorithm.gigpo.mode=$mode) ;;
    hgpo)  algo_args=(algorithm.adv_estimator=hgpo
                      algorithm.hgpo.mode=$mode
                      algorithm.hgpo.weight_type=length
                      algorithm.hgpo.length_weight_alpha=1.0
                      algorithm.hgpo.base_group=False) ;;
    *) echo "unknown algo $ALGO"; exit 1 ;;
esac

project_name="hgpo_t4_${ENV}"
experiment_name="${ALGO}_k${history_length}_$(basename $MODEL)_s${SEED}"
run_dir="${CHECKPOINTS_DIR}/${project_name}/${experiment_name}"
log_name=$experiment_name
env_seed=$SEED
val_only=False
save_freq=$SAVE_FREQ

if [ "${EVAL:-0}" = "1" ]; then
    # evaluate the latest checkpoint in run_dir (loaded by resume_mode=auto), as recipe/hgpo/*_eval.sh
    [ -d "$run_dir" ] || { echo "no checkpoint at $run_dir"; exit 1; }
    EVAL_SEED=${EVAL_SEED:-123}
    env_seed=$EVAL_SEED
    val_data_size=$eval_val_size
    val_only=True
    save_freq=-1
    log_name="${experiment_name}_eval_seed${EVAL_SEED}"
    [ "$ENV" = "alfworld" ] && log_name="${log_name}_${ALFWORLD_EVAL:-eval_in_distribution}"
fi
if [ "${SMOKE:-0}" = "1" ]; then
    EPOCHS=2
    train_data_size=2
    val_data_size=4
    test_freq=1
    save_freq=-1
    run_dir="${CHECKPOINTS_DIR}/${project_name}/smoke_${experiment_name}"
    log_name="smoke_${experiment_name}"
fi

# The parquet files only set the modality and the number of tasks; the env produces the prompts.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# Leave room so the run can be cut cleanly before Kaggle's time limit; progress is kept in checkpoints.
set +e
timeout --signal=INT --kill-after=120 "$TIME_BUDGET" python3 -m recipe.hgpo.main_hgpo \
    "${algo_args[@]}" \
    "${env_args[@]}" \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=$TRAIN_DTYPE \
    actor_rollout_ref.actor.autocast_dtype=$TRAIN_DTYPE \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM \
    `# verl defaults (1024 seqs, 8192 tokens) make vLLM's profiling pass eat the whole T4 budget` \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + 512)) \
    `# vLLM pins 4 GB of host RAM per GPU for KV swap by default; preempted sequences are recomputed instead` \
    actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    `# T4: free the KV cache during the update; verl then requires eager mode (no CUDA graphs)` \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.seed=$env_seed \
    env.history_length=$history_length \
    env.max_steps=$max_steps \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=0.1 \
    ray_init.num_cpus=${RAY_CPUS:-8} \
    trainer.log_group_diagnostics=True \
    trainer.critic_warmup=0 \
    trainer.logger=$logger \
    trainer.project_name=$project_name \
    trainer.experiment_name=$log_name \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$EPOCHS \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="$run_dir" \
    trainer.val_only=$val_only \
    trainer.val_before_train=True \
    trainer.save_freq=$save_freq "$@"
status=$?
set -e

if [ $status -eq 124 ] || [ $status -eq 130 ]; then
    echo "TIME BUDGET REACHED ($TIME_BUDGET): rerun the same command in the next session to resume from $run_dir"
    exit 0
fi
exit $status
