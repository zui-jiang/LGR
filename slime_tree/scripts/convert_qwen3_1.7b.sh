cd /mnt/tidal-alsh-share2/usr/liuyanjiang601/project/slime
source scripts/models/qwen3-1.7B.sh

PYTHONPATH=/mnt/tidal-alsh-share2/usr/liuyanjiang601/project/Megatron-LM-Slime WORLD_SIZE=1 python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /mnt/tidal-alsh-share2/usr/liuyanjiang601/hf_models/Qwen3-1.7B \
    --save /mnt/tidal-alsh-share2/usr/liuyanjiang601/hf_models/Qwen3-1.7B_dist