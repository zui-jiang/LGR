cd /cpfs/user/liuyanjiang/research/project/slime
source scripts/models/qwen3-8B.sh

PYTHONPATH=/cpfs/user/liuyanjiang/research/project/Megatron-LM-Slime python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /cpfs/user/liuyanjiang/hf_models/Qwen3-8B \
    --save /cpfs/user/liuyanjiang/hf_models/Qwen3-8B-dist