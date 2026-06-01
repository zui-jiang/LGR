#!/bin/bash
# Offline distillation: use a teacher model (via SGLang) to generate responses for prompts,
# then SFT the student model on those teacher-generated responses.
#
# Architecture:
#   External teacher server  ←──── offline_distillation_rollout.py (concurrent HTTP calls, training rollout)
#                                         │
#                                   SFT data (tokens + loss_mask)
#                                         │
#                              Megatron student model training (4 GPUs)
#
#   Student SGLang router    ←──── sglang_rollout.eval_rollout (eval rollout, 4 GPUs)
#                                         │
#                                   Eval metrics
#
# Steps:
#   1.  Deploy teacher model externally (any size / architecture, separate node).
#   2.  Convert student model HF → torch_dist (if not done already).
#   3.  Set TEACHER_URL environment variable to point to your teacher server.
#   4.  Run this script.

# ── housekeeping ──────────────────────────────────────────────────────────────
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3

set -ex
export PYTHONBUFFERED=1
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: ${HAS_NVLINK}"

# ── teacher server config ─────────────────────────────────────────────────────
# Teacher model is deployed externally. Set via environment variable:
TEACHER_URL=${TEACHER_URL:-"http://127.0.0.1:30001/generate"}

# ── student model config ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh"   # student model architecture

STUDENT_HF_CKPT=${STUDENT_HF_CKPT:-"/root/Qwen3-4B-Base"}
STUDENT_TORCH_DIST=${STUDENT_TORCH_DIST:-"/root/Qwen3-4B-Base_torch_dist"}
STUDENT_SAVE=${STUDENT_SAVE:-"/root/Qwen3-4B-Base_offline_distill_slime"}

CKPT_ARGS=(
    --hf-checkpoint "${STUDENT_HF_CKPT}"
    --ref-load      "${STUDENT_TORCH_DIST}"
    --load          "${STUDENT_SAVE}/"
    --save          "${STUDENT_SAVE}/"
    --save-interval 500
)

# ── offline distillation rollout args ─────────────────────────────────────────
DISTILL_ARGS=(
    # use the teacher-calling rollout function
    --rollout-function-path \
        examples.offline_distillation.offline_distillation_rollout.generate_rollout

    # teacher server endpoint (reuses --rm-url)
    --rm-url "${TEACHER_URL}"
    # number of concurrent requests sent to the teacher server
    # tune down (e.g. 8) if the teacher server is on limited GPUs
    --rm-max-concurrent-requests 32

    # teacher generation hyperparameters
    # (read as getattr(args, "teacher_*") in offline_distillation_rollout.py)
    # Note: slime does not define these natively; they are forwarded via
    # --extra-args below, OR you can hard-code them in the rollout file.

    # prompt-only dataset (messages must NOT contain an assistant turn)
    --prompt-data /root/prompts_only.parquet
    --input-key messages
    --rollout-shuffle
    --num-epoch 3
    --rollout-batch-size 64
    --global-batch-size  64

    # SFT loss – train only on teacher-generated response tokens
    --loss-type sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
)

# ── student SGLang args (for eval only) ───────────────────────────────────────
SGLANG_ARGS=(
    --rollout-num-gpus 4
    --rollout-num-gpus-per-engine 4
    --sglang-mem-fraction-static 0.8
)

# ── eval args ─────────────────────────────────────────────────────────────────
EVAL_ARGS=(
    --eval-interval 50
    --skip-eval-before-train
    --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
    --n-samples-per-eval-prompt 4
    --eval-max-response-len 2048
    --eval-temperature 0.6
    --eval-top-p 0.9
)

# ── performance args ──────────────────────────────────────────────────────────
PERF_ARGS=(
    --tensor-model-parallel-size 1
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
)

# ── optimiser args ────────────────────────────────────────────────────────────
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 5e-6
    --lr-decay-style cosine
    --min-lr 5e-7
    --lr-warmup-fraction 0.1
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
)

# ── misc args ─────────────────────────────────────────────────────────────────
MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# ── launch ray + training ─────────────────────────────────────────────────────
ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus 8 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:.:${SCRIPT_DIR}/../..\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes         1 \
    --actor-num-gpus-per-node 4 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${DISTILL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${MISC_ARGS[@]}"

# ── cleanup ───────────────────────────────────────────────────────────────────
echo "Training finished."
ray stop
