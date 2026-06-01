#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

: "${PROJECT_DIR:=${REPO_ROOT}}"
: "${TEACHER_IP:?Set TEACHER_IP to the teacher model server IP or hostname. See .env.example.}"

export PROJECT_DIR
export EXP_NAME="${EXP_NAME:-exp258-distill-1.5B-v3}"
export MEGATRON_PATH="${MEGATRON_PATH:-${PROJECT_DIR}/Megatron-LM-Slime}"
export SLIME_PATH="${SLIME_PATH:-${PROJECT_DIR}/slime_tree}"
export SGLANG_PATH="${SGLANG_PATH:-${PROJECT_DIR}/sg_tree}"
export WANDB_KEY="${WANDB_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-zuijiang}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-1.5b-guidev2-opd}"

export TEACHER_IP
export TEACHER_PORT="${TEACHER_PORT:-8000}"
export TEACHER_NAME="${TEACHER_NAME:-sky-7_dasd}"
export TREE_TEACHER_NAME="${TREE_TEACHER_NAME:-sky-7_tree}"

CONSUL_URL="${CONSUL_URL:-http://${TEACHER_IP}:8500/v1/kv/ray-nodes/?recurse=true}"



echo "Waiting for the teacher model server ($TEACHER_IP) to be ready..."

until curl -s -f -X POST "http://$TEACHER_IP:$TEACHER_PORT/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "input_ids": [16,10,16,28,18,30],
    "model": "'"${TEACHER_NAME}"'",
    "sampling_params": {
      "temperature": 1.0,
      "top_p": 1.0,
      "max_new_tokens": 0
    },
    "return_logprob": true,
    "logprob_start_len": 0
  }' > /dev/null; do

    echo "Teacher model ($TEACHER_NAME) is still initializing... (retrying in 5s)"

    if [ -f "$LOG_FILE" ]; then
        echo "--- Last 10 lines of $LOG_FILE ---"
        tail -n 10 "$LOG_FILE"
    fi
    sleep 5
done

echo "Teacher model $TEACHER_NAME is READY!"

# Discover confidence server URLs from Consul by filtering keys containing TREE_TEACHER_NAME
echo "Discovering confidence server URLs for model: ${TREE_TEACHER_NAME}..."
CONFIDENCE_URLS=""
CONSUL_RESPONSE=$(curl -s "$CONSUL_URL" 2>/dev/null)
if [ -n "$CONSUL_RESPONSE" ] && [ "$CONSUL_RESPONSE" != "null" ]; then
    CONFIDENCE_URLS=$(echo "$CONSUL_RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
urls = []
for entry in data:
    key = entry.get('Key', '')
    if '${TREE_TEACHER_NAME}' in key:
        # Key format: ray-nodes/IP:PORT:model_name:...
        parts = key.split('/')[-1].split(':')
        if len(parts) >= 2:
            ip, port = parts[0], parts[1]
            urls.append(f'http://{ip}:{port}/generate')
for u in urls:
    print(u)
")
fi

if [ -n "$CONFIDENCE_URLS" ]; then
    CONFIDENCE_URL_ARRAY=($CONFIDENCE_URLS)
    echo "Found ${#CONFIDENCE_URL_ARRAY[@]} confidence server(s):"
    for u in "${CONFIDENCE_URL_ARRAY[@]}"; do echo "  - $u"; done
    CONFIDENCE_URL_ARGS="--opd-union-topk-confidence-urls ${CONFIDENCE_URLS}"
else
    echo "No confidence servers found for ${TREE_TEACHER_NAME}, will use --rm-url as fallback"
    CONFIDENCE_URL_ARGS=""
fi

CHECKPOINT_DIR=${PROJECT_DIR}/shell/checkpoint/${EXP_NAME}

# ---- Tree Attention Validation ----
echo "Validating tree attention support..."

# Minimal tree: trunk [0->1->2], branch from node 1: node 3, probe child of 3: node 4
#   parent_ids: [-1, 0, 1, 1, 3]
TREE_TEST_PAYLOAD=$(cat <<'EOFPAYLOAD'
{
  "input_ids": [16, 10, 28, 18, 0],
  "parent_ids": [-1, 0, 1, 1, 3],
  "sampling_params": {"max_new_tokens": 0},
  "return_logprob": true,
  "logprob_start_len": 0,
  "top_logprobs_num": 1,
  "model": "PLACEHOLDER_MODEL"
}
EOFPAYLOAD
)
TREE_TEST_PAYLOAD=$(echo "$TREE_TEST_PAYLOAD" | sed "s/PLACEHOLDER_MODEL/${TREE_TEACHER_NAME}/")

# Helper: send tree request and extract probe logprob
extract_probe_logprob() {
    local url=$1
    local label=$2
    local resp
    resp=$(curl -s -X POST "$url" -H "Content-Type: application/json" -d "$TREE_TEST_PAYLOAD" 2>/dev/null)
    if [ -z "$resp" ]; then
        echo "ERROR: No response from $label ($url)"
        return 1
    fi
    echo "$resp" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    meta = data.get('meta_info', {})
    top_logprobs = meta.get('input_top_logprobs', None)
    if top_logprobs is None:
        print('FAIL: response missing input_top_logprobs'); sys.exit(1)
    if len(top_logprobs) != 5:
        print(f'FAIL: expected 5 entries, got {len(top_logprobs)}'); sys.exit(1)
    probe_entry = top_logprobs[4]
    if not probe_entry or len(probe_entry) == 0:
        print('FAIL: probe node has empty logprob data'); sys.exit(1)
    logp = probe_entry[0][0]
    if not isinstance(logp, (int, float)):
        print(f'FAIL: probe logprob is not numeric: {logp}'); sys.exit(1)
    print(f'{logp:.6f}')
except json.JSONDecodeError:
    print('FAIL: response is not valid JSON'); sys.exit(1)
except Exception as e:
    print(f'FAIL: {e}'); sys.exit(1)
"
}

# 1) Test TEACHER_IP directly
TEACHER_DIRECT_URL="http://$TEACHER_IP:$TEACHER_PORT/generate"
echo "  Testing TEACHER_IP: $TEACHER_DIRECT_URL"
TEACHER_LOGPROB=$(extract_probe_logprob "$TEACHER_DIRECT_URL" "TEACHER_IP")
if [ $? -ne 0 ] || [[ "$TEACHER_LOGPROB" == FAIL* ]]; then
    echo "Tree attention validation FAILED on TEACHER_IP: $TEACHER_LOGPROB"
    exit 1
fi
echo "  TEACHER_IP probe logprob: $TEACHER_LOGPROB  OK"

# 2) If Consul URLs exist, test first one and compare with TEACHER_IP
if [ -n "$CONFIDENCE_URLS" ]; then
    CONSUL_TEST_URL="${CONFIDENCE_URL_ARRAY[0]}"
    echo "  Testing Consul URL: $CONSUL_TEST_URL"
    CONSUL_LOGPROB=$(extract_probe_logprob "$CONSUL_TEST_URL" "Consul")
    if [ $? -ne 0 ] || [[ "$CONSUL_LOGPROB" == FAIL* ]]; then
        echo "Tree attention validation FAILED on Consul URL: $CONSUL_LOGPROB"
        exit 1
    fi
    echo "  Consul URL probe logprob: $CONSUL_LOGPROB  OK"

    # Compare TEACHER_IP vs Consul URL results
    CONSISTENT=$(python3 -c "
t, c = float('$TEACHER_LOGPROB'), float('$CONSUL_LOGPROB')
diff = abs(t - c)
if diff > 1e-4:
    print(f'FAIL: logprob mismatch: TEACHER_IP={t:.6f} vs Consul={c:.6f} (diff={diff:.6f})')
else:
    print(f'OK: consistent (diff={diff:.6f})')
")
    echo "  Consistency check: $CONSISTENT"
    if [[ "$CONSISTENT" == FAIL* ]]; then
        echo "Tree attention consistency check FAILED. TEACHER_IP and Consul URL return different results."
        echo "Ensure both serve the same model: ${TREE_TEACHER_NAME}"
        exit 1
    fi
else
    echo "  No Consul URLs found, skipping consistency check."
fi
echo "Tree attention validation PASSED."

echo "========================================="
echo "Training Script: ${EXP_NAME}"
echo "Method: Union TopK KL + Confidence Reward"
echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "========================================="



IS_RESUME=false
WANDB_RUN_ID_ARG=""
SKIP_EVAL=""
if [ -d "${CHECKPOINT_DIR}" ]; then
    IS_RESUME=true
    if [ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
      LAST_ITERATION=$(cat "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")
      echo "✓ Checkpoint detected: iter_${LAST_ITERATION}"
   else
      LAST_ITERATION=0
      echo "✓ No checkpoint found, starting from iter_0"
   fi
    echo "Mode: RESUME"

    WANDB_RUN_ID_FILE="${CHECKPOINT_DIR}/wandb_run_id.txt"
    if [ -f "${WANDB_RUN_ID_FILE}" ]; then
        SAVED_WANDB_RUN_ID=$(cat ${WANDB_RUN_ID_FILE})
        WANDB_RUN_ID_ARG="--wandb-run-id ${SAVED_WANDB_RUN_ID} "
        SKIP_EVAL="--skip-eval-before-train "
        echo "Saved wandb Run ID: ${SAVED_WANDB_RUN_ID}"
        echo "✓ wandb resume support detected!"
    else
        echo "⚠ WARNING: No wandb_run_id.txt found"
        SAVED_WANDB_RUN_ID=""
    fi
else
    echo "Mode: FRESH TRAINING (no checkpoint found)"
    SAVED_WANDB_RUN_ID=""
fi



if [ "$IS_RESUME" = true ]; then
    read -p "Continue with resume? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Resume cancelled."
        exit 0
    fi
fi

if [ -n "${WANDB_KEY}" ]; then
    wandb login "${WANDB_KEY}"
else
    echo "WANDB_KEY is not set; skipping wandb login."
fi
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

BASE_LOG_DIR="${PROJECT_DIR}/shell/logs/${EXP_NAME}"
if [ "$IS_RESUME" = true ]; then
    RUN_DIR="${BASE_LOG_DIR}/resume_${TIMESTAMP}"
else
    RUN_DIR="${BASE_LOG_DIR}/${TIMESTAMP}"
fi

mkdir -p "$RUN_DIR"
echo "Run Directory: $RUN_DIR"

CURRENT_SCRIPT="$0"
if [[ "$CURRENT_SCRIPT" != /* ]]; then
    CURRENT_SCRIPT=$(pwd)/"$CURRENT_SCRIPT"
fi

BACKUP_SCRIPT_PATH="${RUN_DIR}/launch_script_backup.sh"
if [ -f "$CURRENT_SCRIPT" ]; then
    cp "$CURRENT_SCRIPT" "$BACKUP_SCRIPT_PATH"
    echo "Backup launch script to: $BACKUP_SCRIPT_PATH"
fi

record_git_info() {
    local repo_name=$1
    local repo_path=$2
    local output_file="${RUN_DIR}/git_info_${repo_name}.txt"

    echo "Recording Git info for $repo_name to $output_file"
    echo "################################################################" > "$output_file"
    echo "Repository: $repo_name" >> "$output_file"
    echo "Path: $repo_path" >> "$output_file"

    if [ -d "$repo_path" ]; then
        (
            cd "$repo_path" || exit
            echo "Commit ID: $(git rev-parse HEAD 2>/dev/null)" >> "$output_file"
            echo "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)" >> "$output_file"
            echo -e "\n--- Git Status ---" >> "$output_file"
            git status --short >> "$output_file"
            echo -e "\n--- Git Diff (Unstaged) ---" >> "$output_file"
            git diff >> "$output_file"
            echo -e "\n--- Git Diff (Cached) ---" >> "$output_file"
            git diff --cached >> "$output_file"
        )
    else
        echo "Error: Directory not found!" >> "$output_file"
    fi
}

echo "Recording Git information..."
record_git_info "Megatron-LM-Slime" "$MEGATRON_PATH"
record_git_info "Slime" "$SLIME_PATH"
echo "Git info saved to $RUN_DIR."

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SLIME_SCRIPT_DIR="${SLIME_PATH}/scripts"
source "${SLIME_SCRIPT_DIR}/models/deepseek-distill-1.5B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_DIR}/DeepSeek-R1-Distill-Qwen-1.5B
   --ref-load ${HF_MODEL_DIR}/DeepSeek-R1-Distill-Qwen-1.5B-dist
   --load ${CHECKPOINT_DIR}
   --save ${CHECKPOINT_DIR}
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data ${HF_DATASET_DIR}/Polaris-Dataset-53K/polaris-data-53K_slime.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 3000
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
    --eval-prompt-data aime ${HF_DATASET_DIR}/aime-2024/aime-2024.jsonl
    --n-samples-per-eval-prompt 8
    --eval-max-response-len $((39*1024))
    --eval-top-p 1
    --eval-rm-type deepscaler
    ${SKIP_EVAL}
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}
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
   --opd-teacher-topk-size 20
   --teacher-model-name ${TEACHER_NAME}

   --opd-union-topk-confidence
   --opd-union-topk-confidence-coef 1
   --opd-union-topk-confidence-k 8
   --opd-union-topk-confidence-entropy-threshold 0.2
   --opd-union-topk-confidence-max-tree-tokens 18000
   --opd-union-topk-confidence-alternate-steps 2
   --opd-union-topk-confidence-model-name ${TREE_TEACHER_NAME}
)

 RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.on_policy_distillation_gateway.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation_gateway.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
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

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${WANDB_PROJECT}
   --wandb-group ${EXP_NAME}
   --wandb-always-use-train-step
   --log-passrate
   ${WANDB_RUN_ID_ARG}
)

if [ -n "${WANDB_KEY}" ]; then
   WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-enable-metrics
)

DEBUG_ARGS=(
   --dump-details ${PROJECT_DIR}/dump/${EXP_NAME}
   --dump-student-topk-size 20
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${SLIME_PATH}:${SGLANG_PATH}/python:${PYTHONPATH:-}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_CACHE_DIR\": \"${WANDB_CACHE_DIR}\",
    \"WANDB_CONFIG_DIR\": \"${WANDB_CONFIG_DIR}\",
   \"WANDB_DIR\": \"${WANDB_DIR}\",
   \"WANDB_ARTIFACT_DIR\": \"${WANDB_ARTIFACT_DIR}\",
   \"RAY_DEBUG\": \"legacy\",
   \"NCCL_DEBUG\": \"INFO\"
  }
}"

if [ "$IS_RESUME" = true ]; then
    echo "Resuming Training: ${EXP_NAME}"
    echo "From iteration: ${LAST_ITERATION}"
    echo "Target rollout: ${RESUME_NUM_ROLLOUT}"
    if [ -n "${SAVED_WANDB_RUN_ID}" ]; then
        echo "wandb run: ${SAVED_WANDB_RUN_ID} (continuing)"
    fi
    echo "Skip eval before train: YES"
else
    echo "Starting Fresh Training: ${EXP_NAME}"
    echo "Target rollout: ${RESUME_NUM_ROLLOUT}"
    echo "Skip eval before train: NO"
fi
echo "========================================="

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${SLIME_PATH}/train.py \
   --actor-num-nodes ${WORLD_SIZE} \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${RM_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${DEBUG_ARGS[@]} \
   ${MISC_ARGS[@]}

echo ""
echo "========================================="
if [ "$IS_RESUME" = true ]; then
    echo "Union TopK Confidence OPD Resume job submitted!"
else
    echo "Union TopK Confidence OPD Training job submitted!"
fi
echo "Monitor at: http://127.0.0.1:8265"
if [ -n "${SAVED_WANDB_RUN_ID}" ]; then
    echo "wandb: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${SAVED_WANDB_RUN_ID}"
fi
