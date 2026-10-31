#!/usr/bin/env bash
# Public entry point for the bundled 1.5B math experiment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
        echo "Usage: $0 [--dry-run]"
        echo "Loads .env from the repository root. See docs/reproduction.md."
        exit 0 ;;
    "") ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Too many arguments" >&2; exit 2; fi
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

export PROJECT_DIR="${PROJECT_DIR:-${REPO_ROOT}}"
export SLIME_PATH="${SLIME_PATH:-${PROJECT_DIR}/slime_tree}"
export SGLANG_PATH="${SGLANG_PATH:-${PROJECT_DIR}/sg_tree}"
export MEGATRON_PATH="${MEGATRON_PATH:-${PROJECT_DIR}/Megatron-LM}"
export TEACHER_NAME="${TEACHER_NAME:-math-teacher}"
export TREE_TEACHER_NAME="${TREE_TEACHER_NAME:-${TEACHER_NAME}}"
TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-8000}"
TEACHER_URL="http://${TEACHER_IP}:${TEACHER_PORT}/generate"
TEACHER_WAIT_SECONDS="${TEACHER_WAIT_SECONDS:-600}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${PROJECT_DIR}/models}"
HF_DATASET_DIR="${HF_DATASET_DIR:-${PROJECT_DIR}/data}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-${HF_MODEL_DIR}/DeepSeek-R1-Distill-Qwen-1.5B}"
REF_CHECKPOINT="${REF_CHECKPOINT:-${HF_MODEL_DIR}/DeepSeek-R1-Distill-Qwen-1.5B-dist}"
TRAIN_DATA="${TRAIN_DATA:-${HF_DATASET_DIR}/Polaris-Dataset-53K/polaris-data-53K_slime.jsonl}"
EVAL_DATA="${EVAL_DATA:-${HF_DATASET_DIR}/aime-2024/aime-2024.jsonl}"
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_DIR}/scripts/models/deepseek-r1-distill-qwen-1.5b.sh}"
EXP_NAME="${EXP_NAME:-exp258-distill-1.5B-v3}"
CHECKPOINT_DIR="${PROJECT_DIR}/shell/checkpoint/${EXP_NAME}"
WORLD_SIZE="${WORLD_SIZE:-1}" # Training nodes, not the torchrun process count.
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
UNION_TOPK="${UNION_TOPK:-20}"
LGR_NORMALIZE_STD="${LGR_NORMALIZE_STD:-0}"
LGR_ALTERNATE_STEPS="${LGR_ALTERNATE_STEPS:-2}"
MAX_TREE_TOKENS="${MAX_TREE_TOKENS:-18000}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
USE_WANDB="${USE_WANDB:-0}"
export WANDB_KEY="${WANDB_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-lgr}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PROJECT_DIR}/.cache/wandb/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${PROJECT_DIR}/.cache/wandb/config}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/.cache/wandb/run}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${PROJECT_DIR}/.cache/wandb/artifacts}"
export PYTHONUNBUFFERED=1

fail() { echo "ERROR: $*" >&2; exit 1; }
for name in WORLD_SIZE TENSOR_MODEL_PARALLEL_SIZE NUM_ROLLOUT UNION_TOPK MAX_TREE_TOKENS TEACHER_WAIT_SECONDS; do
    value="${!name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer"
done
[[ "$LGR_ALTERNATE_STEPS" =~ ^[0-9]+$ ]] || fail "LGR_ALTERNATE_STEPS must be nonnegative"
for name in USE_WANDB LGR_NORMALIZE_STD; do
    [[ "${!name}" == 0 || "${!name}" == 1 ]] || fail "$name must be 0 or 1"
done
[[ -f "$MODEL_CONFIG" ]] || fail "Missing model configuration: $MODEL_CONFIG"
# shellcheck disable=SC1090
source "$MODEL_CONFIG"

# Explicit tree endpoints take priority over optional Consul discovery.
CONFIDENCE_URL_ARGS=()
CONFIDENCE_URL_ARRAY=()
if [[ -n "${CONFIDENCE_URLS:-}" ]]; then
    read -r -a CONFIDENCE_URL_ARRAY <<< "$CONFIDENCE_URLS"
fi
HAS_NVLINK=0
if (( ! DRY_RUN )); then
    for cmd in python3 curl ray nvidia-smi; do command -v "$cmd" >/dev/null || fail "Missing command: $cmd"; done
    for dir in "$MEGATRON_PATH" "$SLIME_PATH" "$SGLANG_PATH/python" "$STUDENT_CHECKPOINT" "$REF_CHECKPOINT"; do
        [[ -d "$dir" ]] || fail "Missing directory: $dir"
    done
    for file in "$TRAIN_DATA" "$EVAL_DATA"; do [[ -f "$file" ]] || fail "Missing dataset: $file"; done
    if nvidia-smi topo -m 2>/dev/null | grep -q 'NV[0-9]'; then HAS_NVLINK=1; fi
    if [[ ${#CONFIDENCE_URL_ARRAY[@]} -eq 0 && -n "${CONSUL_URL:-}" ]]; then
        discovery=$(curl --connect-timeout 5 --max-time 15 --fail --silent --show-error "$CONSUL_URL")
        urls=$(printf '%s' "$discovery" | python3 -c '
import json, sys
model = sys.argv[1]
for entry in json.load(sys.stdin):
    key = entry.get("Key", "")
    if model in key:
        parts = key.rsplit("/", 1)[-1].split(":")
        if len(parts) >= 2:
            print("http://%s:%s/generate" % (parts[0], parts[1]))
' "$TREE_TEACHER_NAME")
        while IFS= read -r url; do
            [[ -z "$url" ]] || CONFIDENCE_URL_ARRAY+=("$url")
        done <<< "$urls"
    fi
fi
if [[ ${#CONFIDENCE_URL_ARRAY[@]} -gt 0 ]]; then
    CONFIDENCE_URL_ARGS=(--opd-union-topk-confidence-urls ${CONFIDENCE_URL_ARRAY[@]+"${CONFIDENCE_URL_ARRAY[@]}"})
fi
CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_CHECKPOINT}"
   --ref-load "${REF_CHECKPOINT}"
   --load "${CHECKPOINT_DIR}"
   --save "${CHECKPOINT_DIR}"
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 128
   --n-samples-per-prompt 1
   --rollout-max-prompt-len $((1*1024))
   --rollout-max-response-len $((9*1024))
   --rollout-temperature 1.0
   --num-steps-per-rollout 1
   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
    --eval-interval 10
    --eval-prompt-data aime "${EVAL_DATA}"
    --n-samples-per-eval-prompt 8
    --eval-max-response-len $((39*1024))
    --eval-top-p 1
    --eval-rm-type deepscaler
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu $((20*1024))
)


GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.0
   --opd-union-topk-kl
   --opd-teacher-topk-kl-coef 1.0
   --opd-teacher-topk-size "${UNION_TOPK}"
   --teacher-model-name "${TEACHER_NAME}"

   --opd-union-topk-confidence
   --opd-union-topk-confidence-coef 1
   --opd-union-topk-confidence-k 8
   --opd-union-topk-confidence-entropy-threshold 0.2
   --opd-union-topk-confidence-max-tree-tokens "${MAX_TREE_TOKENS}"
   --opd-union-topk-confidence-alternate-steps "${LGR_ALTERNATE_STEPS}"
   --opd-union-topk-confidence-model-name "${TREE_TEACHER_NAME}"
)

 RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.on_policy_distillation_gateway.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation_gateway.post_process_rewards
   --rm-url "${TEACHER_URL}"
   --rm-max-concurrent-requests 32
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()
if [[ "${USE_WANDB}" == 1 ]]; then
   WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT}" --wandb-group "${EXP_NAME}"
               --wandb-always-use-train-step --log-passrate)
   if [[ -n "${WANDB_KEY}" ]]; then
      WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
   fi
   if [[ -f "${CHECKPOINT_DIR}/wandb_run_id.txt" ]]; then
      WANDB_ARGS+=(--wandb-run-id "$(cat "${CHECKPOINT_DIR}/wandb_run_id.txt")")
   fi
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-enable-metrics
)

DEBUG_ARGS=(
   --dump-details "${PROJECT_DIR}/dump/${EXP_NAME}"
   --dump-student-topk-size 20
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)


if [[ "$LGR_NORMALIZE_STD" == 1 ]]; then
    GRPO_ARGS+=(--opd-union-topk-confidence-normalize-std)
fi
GRPO_ARGS+=(${CONFIDENCE_URL_ARGS[@]+"${CONFIDENCE_URL_ARGS[@]}"})
if [[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    EVAL_ARGS+=(--skip-eval-before-train)
    echo "Resuming checkpoint in ${CHECKPOINT_DIR}"
fi

export LGR_RUNTIME_PYTHONPATH="${MEGATRON_PATH}:${SLIME_PATH}:${SGLANG_PATH}/python:${PYTHONPATH:-}"
export LGR_HAS_NVLINK="$HAS_NVLINK"
# Serialize paths with JSON rather than interpolating shell strings into JSON.
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json, os
keys = ("WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DIR", "WANDB_ARTIFACT_DIR", "WANDB_ENTITY")
env = {key: os.environ[key] for key in keys}
env.update(PYTHONPATH=os.environ["LGR_RUNTIME_PYTHONPATH"], CUDA_DEVICE_MAX_CONNECTIONS="1",
           NCCL_NVLS_ENABLE=os.environ["LGR_HAS_NVLINK"], NCCL_DEBUG="INFO", PYTHONUNBUFFERED="1")
print(json.dumps({"env_vars": env}))
PY
)
COMMAND=(ray job submit --address="$RAY_ADDRESS" --runtime-env-json="$RUNTIME_ENV_JSON"
    -- python3 "${SLIME_PATH}/train.py"
    --actor-num-nodes "$WORLD_SIZE" --actor-num-gpus-per-node 8 --colocate
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}"
    "${GRPO_ARGS[@]}" ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} "${PERF_ARGS[@]}" "${RM_ARGS[@]}"
    "${EVAL_ARGS[@]}" "${SGLANG_ARGS[@]}" "${DEBUG_ARGS[@]}" "${MISC_ARGS[@]}")
if (( DRY_RUN )); then
    echo "Dry run: no service requests, training submission, or output directories."
    echo "Local model/data paths have not been checked. W&B key is redacted."
    redact_next=0
    for arg in "${COMMAND[@]}"; do
        if (( redact_next )); then printf '%q ' '<redacted>'; redact_next=0
        else printf '%q ' "$arg"; [[ "$arg" != --wandb-key ]] || redact_next=1
        fi
    done
    printf '\n'
    exit 0
fi

# One probe checks both ordinary scoring and the parent-indexed top-logprob API.
# The full semantic comparison is in sg_tree/test_tree_attention.py.
probe() {
    python3 - "$1" "$2" <<'PY'
import json, sys, urllib.request
url, model = sys.argv[1:]
payload = dict(input_ids=[16,10,28,18,0], parent_ids=[-1,0,1,1,3],
               sampling_params={"max_new_tokens":0}, return_logprob=True,
               logprob_start_len=0, top_logprobs_num=1, model=model)
request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
try:
    with urllib.request.urlopen(request, timeout=15) as response:
        entries = json.load(response)["meta_info"]["input_top_logprobs"]
    assert len(entries) == 5 and isinstance(entries[4][0][0], (int, float))
except Exception as error:
    print(f"Teacher probe failed for {url}: {error}", file=sys.stderr)
    sys.exit(1)
PY
}
start=$SECONDS
echo "Waiting for teacher scoring API: ${TEACHER_URL} (budget ${TEACHER_WAIT_SECONDS}s)"
until probe "$TEACHER_URL" "$TREE_TEACHER_NAME"; do
    (( SECONDS - start < TEACHER_WAIT_SECONDS )) || fail "Teacher readiness timeout"
    sleep 5
done
if [[ "$TEACHER_NAME" != "$TREE_TEACHER_NAME" ]]; then
    probe "$TEACHER_URL" "$TEACHER_NAME"
fi
for url in ${CONFIDENCE_URL_ARRAY[@]+"${CONFIDENCE_URL_ARRAY[@]}"}; do probe "$url" "$TREE_TEACHER_NAME"; done
RUN_DIR="${PROJECT_DIR}/shell/logs/${EXP_NAME}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
cp "$0" "$RUN_DIR/launch_script_backup.sh"
# Avoid copying .env or command arguments containing W&B credentials.
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/repository_commit.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/repository_status.txt"
echo "Submitting ${EXP_NAME}; logs: ${RUN_DIR}; checkpoints: ${CHECKPOINT_DIR}"
"${COMMAND[@]}"
