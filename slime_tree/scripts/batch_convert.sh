#!/bin/bash

# ==============================================================================
# 使用方法：./batch_convert.sh <最大Step数> [HF模型名称]
# ==============================================================================

if [ -z "$1" ]; then
    echo "Usage: $0 <max_step> [hf_model_name]"
    exit 1
fi

MAX_STEP=$1
# 透传第三个参数给子脚本，如果没传，子脚本会使用它自己的默认值
HF_MODEL_NAME=$2

EXPERIMENTS=(
    "length-distill-1.5B-3k"
    "length-distill-1.5B-9k"
    "length-distill-1.5B-16k"
    "length-distill-1.5B-3k-code"
    "length-distill-1.5B-9k-code"
    "length-distill-1.5B-16k-code"
    "topk-distill-1.5B-math"
    "topk-distill-1.5B-code-16k"
    "reolopd-distill-1.5B-code"
    "reolopd-distill-1.5B-math"
    "topk-distill-1.5B-math-jsd"

)

CONVERT_SCRIPT="${PROJECT_DIR}/slime_for_guide_v3/scripts/auto_convert_model_hf.sh"

echo ">>> Batch Start. Target Model: ${HF_MODEL_NAME:-"Default (DeepSeek-R1-1.5B)"}"

for EXP in "${EXPERIMENTS[@]}"; do
    # 调用时带上模型名称参数
    $CONVERT_SCRIPT "$EXP" "$MAX_STEP" "$HF_MODEL_NAME"
done

echo ">>> All Batch tasks finished."