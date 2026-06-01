# RFT (Rejection Sampling Fine-Tuning)

Generate multiple response candidates per prompt, evaluate with reward model, and train only on correct responses.

## How It Works

```
For each prompt:
  1. Generate N candidates (e.g., 4 responses)
  2. Evaluate each with reward model (deepscaler, remote_rm, etc.)
  3. Keep only responses with reward >= threshold
  4. Result: 0-N correct samples per prompt (variable length)

Framework automatically flattens groups for SFT training.
Over-sampling ensures final batch size = rollout_batch_size.
```

## Key Differences from Offline Distillation

| Feature | Offline Distillation (exp33) | RFT (exp34) |
|---------|------------------------------|-------------|
| Candidates per prompt | 4 (all kept) | 4 (filtered) |
| Reward evaluation | None (reward=0) | Yes (rm_type) |
| Samples per group | Fixed 4 | Variable 0-4 |
| Training samples | All responses | Only correct responses |
| Over-sampling | Not needed | Required |

## Usage

```bash
--rollout-function-path examples.rft.rft_rollout:generate_rollout

# Teacher model (same as offline distillation)
--rm-url http://teacher:8000/generate
--teacher-model-name exp13_ter_0000089_sft

# Reward model
--rm-type deepscaler                        # or remote_rm, dapo, math, etc.
--rft-reward-threshold 0.5                  # Min reward to keep (default 0.5)

# Sampling
--n-samples-per-prompt 4                    # Generate 4 candidates
--rollout-batch-size 128                    # Target: 128 prompt groups
--over-sampling-batch-size 256              # 2x over-sample (expect ~50% filter)

# Concurrency
--rm-max-concurrent-requests 128            # Teacher generation concurrency

# SFT loss (same as offline distillation)
--loss-type sft_loss
--calculate-per-token-loss
--disable-compute-advantages-and-returns
```

## Supported Reward Models

All `rm_type` from `slime.rollout.rm_hub` are supported:

- **deepscaler**: Rule-based math grader (boxed answer + sympy equivalence)
- **remote_rm**: External reward model via HTTP (`--rm-url` for RM endpoint)
- **dapo**: DAPO-style math grading
- **math**: VERL-style math grading
- **f1**: F1 score between response and label
- **gpqa**: GPQA-style evaluation
- **custom**: User-defined via `--custom-rm-path`

## Concurrency Control

- **Teacher generation**: Controlled by single semaphore (`rm_max_concurrent_requests`)
- **Reward evaluation**:
  - `deepscaler`, `dapo`, `math`, `f1`: Local CPU computation (no semaphore needed)
  - `remote_rm`: Uses `rm_max_concurrent_requests` semaphore (shared with generation)
  - `custom`: Depends on implementation

## Over-Sampling Strategy

Set `over-sampling-batch-size` based on expected filter rate:

```
Expected filter rate = % of prompts with all candidates incorrect

Examples:
- 30% filter → over_sampling = rollout_batch_size × 1.5
- 50% filter → over_sampling = rollout_batch_size × 2.0
- 70% filter → over_sampling = rollout_batch_size × 3.0
```

Monitor `rollout/rft/filter_rate` in wandb to tune this parameter.

## Example: exp34 vs exp33

```bash
# exp33 (offline distillation)
--rollout-function-path examples.offline_distillation.offline_distillation_rollout:generate_rollout
--rollout-batch-size 128
--n-samples-per-prompt 4
--over-sampling-batch-size 128              # No filtering
# Result: 128 groups × 4 samples = 512 samples (all kept)

# exp34 (RFT)
--rollout-function-path examples.rft.rft_rollout:generate_rollout
--rollout-batch-size 128
--n-samples-per-prompt 4
--over-sampling-batch-size 256              # 2x over-sample
--rm-type deepscaler
--rft-reward-threshold 0.5
# Result: 128 groups × avg 2 correct samples = ~256 samples (only correct kept)
```

## Data Format

Same as offline distillation: prompts must have `label` field for reward evaluation.

```jsonl
{"prompt": [{"role": "user", "content": "Calculate 2+2"}], "label": "4"}
{"prompt": [{"role": "user", "content": "Solve x^2=4"}], "label": "\\boxed{2}"}
```

## Notes

- Groups can have variable length (1-4 samples) after filtering
- Framework automatically flattens groups before SFT training
- Empty groups (all incorrect) are discarded
- If all over-sampled prompts fail, rollout will retry until target size reached
