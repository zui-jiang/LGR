# Slime Tree Experiment

This repository bundles the Slime and SGLang forks used by the tree-attention distillation experiment, plus the launch script for `exp258-distill-1.5B-v3`.

## Repository Layout

```text
.
├── slime_tree/                         # Slime fork, tracked as a git submodule
├── sg_tree/                            # SGLang fork, tracked as a git submodule
├── scripts/
│   └── exp258-distill-1.5B-v3.sh       # Experiment launch script
├── .env.example                        # Local runtime configuration template
└── .gitmodules
```

## Quick Start

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd <repo-name>
```

If you cloned without submodules:

```bash
git submodule update --init --recursive
```

Create local configuration:

```bash
cp .env.example .env
```

Edit `.env` for your cluster paths, teacher server, dataset paths, and W&B settings. Then launch:

```bash
./scripts/exp258-distill-1.5B-v3.sh
```

## Configuration

The launch script reads `.env` automatically when it exists. Important variables:

| Variable | Purpose |
| --- | --- |
| `PROJECT_DIR` | Absolute path to this repository. Defaults to the repository root. |
| `SLIME_PATH` | Slime checkout path. Defaults to `slime_tree`. |
| `SGLANG_PATH` | SGLang checkout path. Defaults to `sg_tree`. |
| `MEGATRON_PATH` | Megatron-LM checkout path. Set this to your local Megatron tree. |
| `TEACHER_IP`, `TEACHER_PORT` | Teacher model server endpoint. |
| `TEACHER_NAME`, `TREE_TEACHER_NAME` | Model names used by the teacher and tree teacher endpoints. |
| `HF_MODEL_DIR`, `HF_DATASET_DIR` | Local model and dataset roots. |
| `WANDB_KEY` | Optional W&B API key. Keep this in `.env`, never in git. |

## Notes

- The root repository contains only experiment orchestration. Each submodule keeps its own license, dependencies, and documentation.
- `.gitmodules` uses HTTPS URLs so public clones can initialize submodules without SSH configuration. If you need private access, override the submodule URLs locally with `git config`.
- Runtime outputs such as logs, checkpoints, dumps, and W&B cache files are ignored by git.

## License

Root-level orchestration files are released under the MIT License. Submodules are governed by their own licenses.
