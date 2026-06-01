cd /cpfs/user/liuyanjiang/Eng/Slime/slime
source scripts/models/qwen3-30B-A3B.sh

PYTHONPATH=/cpfs/user/liuyanjiang/Eng/Slime/Megatron-LM-Slime python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /cpfs/user/liuyanjiang/hf_models/Qwen3-30B-A3B \
    --save /cpfs/user/liuyanjiang/hf_models/Qwen3-30B-A3B-dist