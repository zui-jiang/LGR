# Installation and training

This guide walks through the 1.5B math training script, from environment setup and Hugging Face downloads to teacher serving and Ray submission.

## 1. Docker environment

Use the official [Slime Docker image](https://thudm.github.io/slime/get_started/quick_start.html). From the LGR checkout on a Linux GPU host:

```bash
export SLIME_IMAGE=slimerl/slime:latest
docker pull "$SLIME_IMAGE"
docker run --rm -it --gpus all --ipc=host --network host \
  --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/workspace/LGR" \
  -w /workspace/LGR \
  "$SLIME_IMAGE" /bin/bash
```

Inside the container, install the local SGLang and Slime source trees into the image's existing environment:

```bash
export LGR_ROOT=/workspace/LGR
python -m pip install -e "$LGR_ROOT/sg_tree/python" --no-deps
(cd "$LGR_ROOT/slime_tree" && python -m pip install -e . --no-deps)
```

SGLang is installed from **`sg_tree/python`** under the Python package name `sglang`. This is the implementation used for LGR's tree-attention scoring. Install it on the teacher-serving hosts and the student rollout/training hosts. Install `slime_tree` on the training hosts.

Verify the active imports:

```bash
python -c 'import sglang, slime; print("SGLang:", sglang.__file__); print("Slime:", slime.__file__)'
python -c 'import torch, megatron.core; print("Visible GPUs:", torch.cuda.device_count())'
```

The source paths should resolve to `/workspace/LGR/sg_tree/python/sglang` and `/workspace/LGR/slime_tree/slime`. The Slime image supplies the Megatron training environment; set `MEGATRON_PATH` to its checkout location, normally `/root/Megatron-LM`.

## 2. Models and datasets

From the repository root inside the container, initialize your local configuration:

```bash
cp .env.example .env
```

Set these paths in `.env`:

```bash
MEGATRON_PATH=/root/Megatron-LM
HF_MODEL_DIR=/workspace/LGR/models
HF_DATASET_DIR=/workspace/LGR/data
```

Load the configuration into the current shell:

```bash
set -a
source .env
set +a
export LGR_ROOT=/workspace/LGR
```

### Download models

For the 1.5B math experiment:

```bash
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir "$HF_MODEL_DIR/DeepSeek-R1-Distill-Qwen-1.5B"
hf download Skywork/Skywork-OR1-Math-7B \
  --local-dir "$HF_MODEL_DIR/Skywork-OR1-Math-7B"
```

Additional model sources are linked in the [README](../README.md#models-and-datasets).

### Download data

Download the original [Polaris math dataset](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K) and the [AIME-24 JSONL dataset](https://huggingface.co/datasets/zhuzilin/aime-2024):

```bash
hf download --repo-type dataset POLARIS-Project/Polaris-Dataset-53K \
  --local-dir "$HF_DATASET_DIR/Polaris-Dataset-53K"
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir "$HF_DATASET_DIR/aime-2024"
```

The training loader reads `prompt` and `label`. Adapt Polaris's `problem` and `answer` fields to those names:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

folder = Path(os.environ["HF_DATASET_DIR"]) / "Polaris-Dataset-53K"
source = folder / "polaris-data-53K.jsonl"
target = folder / "polaris-data-53K_slime.jsonl"
with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        row["prompt"] = [{"role": "user", "content": row["problem"]}]
        row["label"] = row["answer"]
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
print(target)
PY
```

The AIME-24 JSONL already provides `prompt` and `label`. The resulting paths match the training script's defaults:

```text
$HF_DATASET_DIR/Polaris-Dataset-53K/polaris-data-53K_slime.jsonl
$HF_DATASET_DIR/aime-2024/aime-2024.jsonl
```

### Convert the student checkpoint

Convert the Hugging Face student checkpoint to Megatron's distributed format:

```bash
export PYTHONPATH="${MEGATRON_PATH}:${LGR_ROOT}/slime_tree:${LGR_ROOT}/sg_tree/python:${PYTHONPATH:-}"
source scripts/models/deepseek-r1-distill-qwen-1.5b.sh
python slime_tree/tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_MODEL_DIR/DeepSeek-R1-Distill-Qwen-1.5B" \
  --save "$HF_MODEL_DIR/DeepSeek-R1-Distill-Qwen-1.5B-dist"
```

## 3. Teacher service

Use SGLang's [standard server entry point](https://docs.sglang.io/docs/basic_usage/send_request), with the local `sg_tree` implementation installed in the teacher environment:

```bash
python -m sglang.launch_server \
  --model-path "$HF_MODEL_DIR/Skywork-OR1-Math-7B" \
  --served-model-name math-teacher \
  --host 0.0.0.0 --port 8000 \
  --attention-backend triton \
  --chunked-prefill-size -1
```

The tree-scoring implementation uses the Triton attention backend; the command above selects it and keeps each tree request in a single prefill.

Configure the teacher address and serving name in `.env`:

```bash
TEACHER_IP=127.0.0.1
TEACHER_PORT=8000
TEACHER_NAME=math-teacher
TREE_TEACHER_NAME=${TEACHER_NAME}
```

Set `TEACHER_IP` to the teacher host address when it runs on another machine. Configure GPU parallelism and serving options using SGLang's server arguments for your deployment.

The repository includes a tree-scoring test that compares branch scores with equivalent linear paths:

```bash
python sg_tree/test_tree_attention.py --server http://TEACHER_HOST:8000
```

For deployments with dedicated tree-scoring endpoints, configure `CONFIDENCE_URLS` in `.env`. `CONSUL_URL` can also be set for service discovery.

## 4. Ray and training

Start Ray on the training head node:

```bash
ray start --head --node-ip-address TRAINING_HEAD_IP \
  --port 6379 --dashboard-host 0.0.0.0 --dashboard-port 8265 --num-gpus 8
```

Join additional training workers to the head:

```bash
ray start --address TRAINING_HEAD_IP:6379 --num-gpus 8
```

Use the same container environment and repository/data paths on all workers. In `.env`, `WORLD_SIZE` sets the number of training nodes; the launcher uses 8 GPUs per node. Set `RAY_ADDRESS` to the head's dashboard address if submitting from another host.

Preview the command, then run the training entry point:

```bash
bash scripts/exp258-distill-1.5B-v3.sh --dry-run
bash scripts/exp258-distill-1.5B-v3.sh
```

The script reads `.env`, checks configured paths and teacher availability, and submits the training job to Ray. `USE_WANDB=1` enables W&B; set `WANDB_ENTITY`, `WANDB_PROJECT`, and authentication for your account.

## Outputs

| Output | Location |
| --- | --- |
| Checkpoints | `shell/checkpoint/$EXP_NAME/` |
| Launcher snapshot and repository revision | `shell/logs/$EXP_NAME/<timestamp>/` |
| Training dumps | `dump/$EXP_NAME/` |
| Worker logs | Ray job output and dashboard |

An existing checkpoint under the same experiment name is loaded for continuation. Set a new `EXP_NAME` to start a separate run.
