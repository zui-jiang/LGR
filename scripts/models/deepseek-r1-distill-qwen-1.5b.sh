# Architecture: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/config.json
# Unlike Qwen2.5-1.5B, this checkpoint does not tie input/output embeddings.
MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 1536
   --ffn-hidden-size 8960
   --num-attention-heads 12
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 10000
   --group-query-attention
   --num-query-groups 2
   --vocab-size 151936
   --untie-embeddings-and-output-weights
)
