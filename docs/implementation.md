# Implementation guide

## Method-to-code map

| Component | Entry point |
| --- | --- |
| Experiment configuration and Ray submission | [`scripts/train_lgr.sh`](../scripts/train_lgr.sh) |
| Student architecture | [`scripts/models/deepseek-r1-distill-qwen-1.5b.sh`](../scripts/models/deepseek-r1-distill-qwen-1.5b.sh) |
| Teacher ordinary scoring | `reward_func` in [`on_policy_distillation_gateway.py`](../slime_tree/examples/on_policy_distillation/on_policy_distillation_gateway.py) |
| Entropy selection, candidate branches, probe scoring | `post_process_rewards` in the same gateway module |
| Union-KL and direct confidence objective | `compute_union_topk_kl` in [`loss.py`](../slime_tree/slime/backends/megatron_utils/loss.py) |
| Alternative sampled-advantage path | `apply_opd_union_topk_confidence_to_advantages` in the same loss module |
| OPD/LGR CLI options | [`arguments.py`](../slime_tree/slime/utils/arguments.py) |
| Tree request schema | [`io_struct.py`](../sg_tree/python/sglang/srt/managers/io_struct.py) |
| Tree mask and depth positions | [`tree_attention_utils.py`](../sg_tree/python/sglang/srt/managers/tree_attention_utils.py) |
| Tree batch construction and logprob parent indices | [`schedule_batch.py`](../sg_tree/python/sglang/srt/managers/schedule_batch.py) |
| Tree request scheduling | [`scheduler.py`](../sg_tree/python/sglang/srt/managers/scheduler.py) |
| Forward metadata and positions | [`forward_batch_info.py`](../sg_tree/python/sglang/srt/model_executor/forward_batch_info.py) |
| Attention mask consumption | [`triton_backend.py`](../sg_tree/python/sglang/srt/layers/attention/triton_backend.py) |
| Parent-aligned log probabilities | [`logits_processor.py`](../sg_tree/python/sglang/srt/layers/logits_processor.py) |
| Tree-versus-linear server tests | [`test_tree_attention.py`](../sg_tree/test_tree_attention.py) |

## Data flow

1. Student rollouts retain sampled tokens and top-K token probabilities.
2. The gateway collects the teacher's ordinary scoring distribution, selects high-entropy response positions, and takes the top-8 student candidates at each selected position by default.
3. For each selected position, it appends a candidate branch and a dummy child probe to the original sequence. The child probe exposes the distribution **after** its candidate parent.
4. SGLang receives `input_ids` and `parent_ids`. A token attends only to itself and its ancestors; position IDs follow tree depth. Returned input log probabilities are indexed by parent rather than by the preceding flattened token.
5. The gateway converts the probe's top-1 teacher log probability to a probability and stores confidence values alongside candidate token IDs.
6. Slime joins confidence values to the student candidate tokens and optimizes a union-top-K reverse-KL objective with an optional confidence term.

The ordinary teacher and tree teacher may have different serving names/endpoints, but must represent the same teacher weights and tokenizer for a consistent objective.

## Training and scoring options

The training script configures candidate selection, teacher scoring, and the loss through the OPD options in `slime/utils/arguments.py`. Confidence values are computed for the student candidates selected during rollout, then matched by token ID in the training loss. The gateway splits candidate groups into tree requests according to `--opd-union-topk-confidence-max-tree-tokens`.

## Minimal tree scoring request

For a main path `0 → 1 → 2` and an alternative `0 → 1 → 3 → 4`, where node 4 is the dummy probe:

```json
{
  "input_ids": [16, 10, 28, 18, 0],
  "parent_ids": [-1, 0, 1, 1, 3],
  "model": "math-teacher",
  "sampling_params": {"max_new_tokens": 0},
  "return_logprob": true,
  "logprob_start_len": 0,
  "top_logprobs_num": 1
}
```

Read `meta_info.input_top_logprobs[4][0][0]` for the top-1 log probability of the distribution after node 3, then exponentiate to obtain the raw confidence reward. The dummy token's own probability is not the reward. `parent_ids[0]` must be -1; every later parent must refer to an earlier flattened node. This fork treats tree requests as scoring-only and forces `max_new_tokens=0`.
