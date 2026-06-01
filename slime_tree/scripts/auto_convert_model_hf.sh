#!/bin/bash

# ==============================================================================
# 脚本功能：自动遍历实验目录下的 iter_ 文件夹并转换为 Hugging Face 格式
# 使用方法：./convert.sh <实验名称> <最大Step数> [HF模型名称]
# 示例 1（用默认）：./convert.sh exp91-qwen3-1.7B 50
# 示例 2（自定义）：./convert.sh exp91-qwen3-1.7B 50 Qwen2.5-7B
# ==============================================================================

# 1. 检查必要参数
if [ "$#" -lt 2 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <experiment_name> <max_step> [hf_model_name]"
    exit 1
fi

# 2. 接收参数
EXP_NAME=$1
MAX_STEP=$2
# 如果 $3 为空，则默认使用 DeepSeek-R1-Distill-Qwen-1.5B
HF_MODEL_NAME=${3:-"DeepSeek-R1-Distill-Qwen-1.5B"}

# 3. 路径配置
BASE_DIR="${PROJECT_DIR}/shell/checkpoint/${EXP_NAME}"
HF_SOURCE="${HF_MODEL_DIR}/${HF_MODEL_NAME}"
CONVERT_SCRIPT="${PROJECT_DIR}/slime/tools/convert_torch_dist_to_hf.py"

# 4. 环境准备
cd "${PROJECT_DIR}/slime" || { echo "Error: Cannot cd to ${PROJECT_DIR}/slime"; exit 1; }
export PYTHONPATH="${PROJECT_DIR}/Megatron-LM-Slime"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Experiment directory not found: $BASE_DIR"
    exit 1
fi

echo "----------------------------------------------------------------"
echo ">>> Experiment: ${EXP_NAME}"
echo ">>> Max Step:   ${MAX_STEP}"
echo ">>> HF Source:  ${HF_MODEL_NAME}"
echo "----------------------------------------------------------------"

# 5. 循环处理
for INF in "${BASE_DIR}"/iter_*; do
    if [[ "$INF" == *"_hf" ]]; then continue; fi

    DIR_NAME=$(basename "$INF")
    STEP_STR=${DIR_NAME#iter_}
    
    if (( 10#$STEP_STR <= 10#$MAX_STEP )); then
        OUF="${INF}_hf"
        if [ -d "$OUF" ]; then
            echo ">>> [SKIP] $DIR_NAME: Already exists"
        else
            echo ">>> [CONVERTING] $DIR_NAME"
            python "$CONVERT_SCRIPT" \
                --input-dir "$INF" \
                --output-dir "$OUF" \
                --origin-hf-dir "$HF_SOURCE"
                
            [ $? -eq 0 ] && echo "    Success: $DIR_NAME" || echo "    FAILED: $DIR_NAME"
        fi
    else
        echo ">>> [OUT OF RANGE] $DIR_NAME"
    fi
done