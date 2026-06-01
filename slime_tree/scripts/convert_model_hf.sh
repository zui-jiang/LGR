cd ${PROJECT_DIR}/slime

export INF=${PROJECT_DIR}/shell/checkpoint/exp31-qwen3-4B-fixt/iter_0000029
export OUF=${PROJECT_DIR}/shell/checkpoint/exp31-qwen3-4B-fixt/iter_0000029_hf
export HF=${HF_MODEL_DIR}/Qwen3-4B

PYTHONPATH=${PROJECT_DIR}/Megatron-LM-Slime python tools/convert_torch_dist_to_hf.py \
  --input-dir $INF \
  --output-dir $OUF \
  --origin-hf-dir $HF

