#!/bin/bash
# Slime GRPO smoke test for Qwen2.5-1.5B-Instruct with /root/own round reward.

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1

SLIME_ROOT=${SLIME_ROOT:-/root/slime}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
MODEL_DIR=${MODEL_DIR:-/root/models/Qwen2.5-1.5B-Instruct}
LOAD_DIR=${LOAD_DIR:-}
REF_LOAD=${REF_LOAD:-/root/models/Qwen2.5-1.5B-Instruct_torch_dist}
PROMPT_DATA=${PROMPT_DATA:-/root/dapo-math-17k/dapo-math-17k.jsonl}
SAVE_DIR=${SAVE_DIR:-/root/own/checkpoints/qwen2.5-1.5B-round-smoke}
DEBUG_DIR=${DEBUG_DIR:-/root/own/debug_rollouts/qwen2.5-1.5B-round-smoke}

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NUM_GPUS=${NUM_GPUS:-2}
TP_SIZE=${TP_SIZE:-${NUM_GPUS}}
CLEANUP_EXISTING=${CLEANUP_EXISTING:-1}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.35}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-2048}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-512}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
NUM_ROLLOUT=${NUM_ROLLOUT:-1}
NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
SAVE_INTERVAL=${SAVE_INTERVAL:-9999}
SEED=${SEED:-1234}
ROLLOUT_SEED=${ROLLOUT_SEED:-42}
LR=${LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.00}
ENTROPY_COEF=${ENTROPY_COEF:-0.00}
USE_CHECKPOINT_OPT_PARAM_SCHEDULER=${USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-0}
WANDB_ENABLE=${WANDB_ENABLE:-0}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-slime-grpo-round}
WANDB_GROUP=${WANDB_GROUP:-qwen2.5-1.5B-round-smoke}
WANDB_DIR=${WANDB_DIR:-/root/own/wandb}
WANDB_TEAM=${WANDB_TEAM:-}
WANDB_API_KEY=${WANDB_API_KEY:-}
WANDB_RANDOM_SUFFIX=${WANDB_RANDOM_SUFFIX:-1}

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing HF checkpoint: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -d "${REF_LOAD}" ]]; then
  echo "Missing torch_dist checkpoint: ${REF_LOAD}" >&2
  exit 1
fi

if [[ ! -f "${PROMPT_DATA}" ]]; then
  echo "Missing prompt data: ${PROMPT_DATA}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}" "${DEBUG_DIR}"

if [[ "${CLEANUP_EXISTING}" == "1" ]]; then
  pkill -f sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  sleep 2
fi

cd "${SLIME_ROOT}"
source "${SLIME_ROOT}/scripts/models/qwen2.5-1.5B.sh"
MODEL_ARGS+=(--rotary-base 1000000)

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)

if [[ -n "${LOAD_DIR}" ]]; then
  if [[ ! -d "${LOAD_DIR}" ]]; then
    echo "Missing load checkpoint: ${LOAD_DIR}" >&2
    exit 1
  fi
  CKPT_ARGS+=(--load "${LOAD_DIR}")
fi

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rollout-seed "${ROLLOUT_SEED}"
  --custom-rm-path own.slime_adapters.round_reward.reward_func
  --custom-reward-post-process-path own.slime_adapters.round_reward.post_process_rewards

  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"

  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --save-debug-rollout-data "${DEBUG_DIR}/rollout_{rollout_id}.pt"
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef "${KL_LOSS_COEF}"
  --kl-loss-type low_var_kl
  --entropy-coef "${ENTROPY_COEF}"
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

if [[ "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--use-checkpoint-opt-param-scheduler)
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${TP_SIZE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
)

WANDB_ARGS=()
if [[ "${WANDB_ENABLE}" == "1" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-mode "${WANDB_MODE}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-dir "${WANDB_DIR}"
  )
  if [[ -n "${WANDB_TEAM}" ]]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
  if [[ "${WANDB_RANDOM_SUFFIX}" == "0" ]]; then
    WANDB_ARGS+=(--disable-wandb-random-suffix)
  fi
fi

MISC_ARGS=(
  --seed "${SEED}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

ray start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${MEGATRON_ROOT}:${SLIME_ROOT}:/root\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK\":\"${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-false}\",\"SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK\":\"${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK:-true}\",\"ROUND_REWARD_LENGTH_WEIGHT\":\"${ROUND_REWARD_LENGTH_WEIGHT:-0.1}\",\"ROUND_REWARD_SOFT_LENGTH_CAP\":\"${ROUND_REWARD_SOFT_LENGTH_CAP:-768}\",\"ROUND_REWARD_HARD_LENGTH_CAP\":\"${ROUND_REWARD_HARD_LENGTH_CAP:-1024}\",\"ROUND_REWARD_FORMAT_WEIGHT\":\"${ROUND_REWARD_FORMAT_WEIGHT:-0.05}\",\"ROUND_REWARD_ANTI_HACK_WEIGHT\":\"${ROUND_REWARD_ANTI_HACK_WEIGHT:-0.15}\"}}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${MISC_ARGS[@]}"
