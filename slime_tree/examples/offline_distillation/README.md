# Offline Distillation Example

Use a teacher model (deployed via SGLang) to generate high-quality responses for your prompts,
then SFT the student model on those teacher-generated responses.

## Architecture

```
Teacher SGLang Server                          Student Training (Megatron)
(e.g. Qwen3-32B on 4× GPU)                    (e.g. Qwen3-4B on 8× GPU)
        │                                               │
        │  concurrent HTTP /generate calls             │
        ├──────────────────────────────────►  offline_distillation_rollout.py
        │  teacher response tokens                     │
        │  + SFT loss mask                             │
        │                                    SFT loss (teacher response only)
        │                                              ▼
        │                                    Updated student weights
```

Key properties:
- **No student SGLang needed** (`--debug-train-only`): the student model is trained with Megatron only.
- **Teacher weights are never updated**: the teacher runs in a standalone SGLang server.
- **Fully concurrent**: all samples in a rollout batch are sent to the teacher server concurrently,
  bounded by `--rm-max-concurrent-requests`.

## Why not use the built-in rollout engine as teacher?

After every training step `train_async.py` calls `actor_model.update_weights()`, which overwrites
the SGLang weights with the latest student checkpoint.  There is no built-in flag to skip this
sync, so it is impossible to keep a frozen teacher model in the rollout engine without patching
core training code.  The external SGLang server approach avoids this problem entirely.

## Prerequisites

1. A prompt-only dataset in `.parquet` or `.jsonl` format.  Each record must have a `messages`
   field that contains **only user/system turns** (no assistant turn); the teacher will fill in
   the assistant response.

   ```json
   {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
   {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain quantum entanglement."}]}
   ```

2. HuggingFace checkpoints for both teacher and student models.

3. Student model converted to `torch_dist` format:
   ```bash
   source scripts/models/qwen3-4B.sh
   PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
       "${MODEL_ARGS[@]}" \
       --hf-checkpoint /root/Qwen3-4B-Base \
       --save /root/Qwen3-4B-Base_torch_dist
   ```

## Running

```bash
# Set required paths
export TEACHER_MODEL_PATH=/root/Qwen3-32B
export TEACHER_PORT=30001
export TEACHER_TP=4                        # GPUs for teacher

export STUDENT_HF_CKPT=/root/Qwen3-4B-Base
export STUDENT_TORCH_DIST=/root/Qwen3-4B-Base_torch_dist
export STUDENT_SAVE=/root/Qwen3-4B-Base_offline_distill_slime

bash examples/offline_distillation/run-offline-distillation.sh
```

## Key Arguments

| Argument | Description |
|---|---|
| `--rollout-function-path examples.offline_distillation.offline_distillation_rollout:generate_rollout` | Use the offline distillation rollout function |
| `--rm-url http://host:port/generate` | Teacher SGLang server endpoint |
| `--rm-max-concurrent-requests N` | Max concurrent requests to teacher (default 32, lower for large teachers) |
| `--debug-train-only` | Skip student SGLang initialisation |
| `--loss-type sft_loss` | Standard SFT cross-entropy loss |
| `--calculate-per-token-loss` | Average loss over all unmasked tokens (standard SFT behaviour) |
| `--disable-compute-advantages-and-returns` | Skip RL advantage computation |

## Teacher Generation Hyperparameters

The rollout file reads generation parameters from `args` via `getattr(args, "teacher_*", default)`:

| `getattr` key | Default | Meaning |
|---|---|---|
| `teacher_temperature` | `0.7` | Sampling temperature |
| `teacher_top_p` | `0.9` | Top-p nucleus sampling |
| `teacher_max_new_tokens` | `2048` | Max tokens to generate |
| `teacher_model_name` | `None` | Model name for multi-model SGLang servers |

These are not registered slime arguments, so the simplest way to override them is to edit
`offline_distillation_rollout.py` directly, or add them via a custom argument parser hook.

## Concurrency Tuning

`--rm-max-concurrent-requests` limits how many in-flight HTTP requests are sent to the teacher
server simultaneously.  A good starting point:

- Small teacher (7-8 B, single GPU): `--rm-max-concurrent-requests 8`
- Medium teacher (32 B, 4 × GPU): `--rm-max-concurrent-requests 32`
- Large teacher (70+ B, 8 × GPU): `--rm-max-concurrent-requests 16`

If you see OOM errors on the teacher server, lower this value.
If the teacher server is idle while the student is training, increase it.

## Differences from On-Policy Distillation (OPD)

| | Offline Distillation (this example) | OPD |
|---|---|---|
| Training signal | SFT cross-entropy on teacher responses | KL divergence from teacher log-probs |
| Student generates? | No | Yes (student rollout) |
| Teacher frozen? | Yes (external server) | Yes (external server or Megatron copy) |
| Student SGLang needed? | No (`--debug-train-only`) | Yes |
| Use case | Imitate teacher outputs directly | Match teacher token distribution |
