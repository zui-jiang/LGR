"""
TopK On-Policy Distillation (OPD) implementation.

This module implements TopK OPD, which uses weighted KL estimation over topk candidate tokens
for more accurate distillation compared to single-token OPD.

Key differences from standard OPD:
1. Collects topk tokens and their log probs from student during rollout
2. Queries teacher model for log probs of these topk tokens
3. Computes weighted KL: KL ≈ Σ_i p_student(i) * [log p_student(i) - log p_teacher(i)]

Usage:
    1. Start teacher model with SGLang server
    2. Enable in training args:
       --use-opd \\
       --opd-type sglang \\
       --opd-use-topk \\
       --opd-topk-size 10 \\
       --opd-kl-coef 1.0 \\
       --custom-rm-path examples.on_policy_distillation.on_policy_distillation_topk:reward_func \\
       --reward-aggregation-function-path examples.on_policy_distillation.on_policy_distillation_topk:post_process_rewards \\
       --rm-url http://localhost:30000/generate
"""

import torch

from slime.utils.http_utils import post
from slime.utils.types import Sample

async def reward_func(args, sample, **kwargs):
    """Query teacher model to get log probabilities for all tokens in the sequence.

    For TopK OPD, we need the teacher's log probs for the topk tokens selected by student.
    We request full vocabulary logits from teacher so we can index the topk tokens.

    Uses slime's http_utils.post() which includes retry logic (max 60 retries, 1s interval).
    """
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
            "top_k": args.rollout_top_k,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    # Add model parameter if specified
    teacher_model_name = getattr(args, 'teacher_model_name', None)
    if teacher_model_name:
        payload["model"] = teacher_model_name

    # For TopK OPD mode, request full vocabulary logprobs
    if getattr(args, "opd_use_topk", False):
        # Request topk logprobs (large number to get full distribution)
        # SGLang will return top_logprobs_num items per position
        payload["top_logprobs_num"] = min(args.opd_topk_size * 10, 1000)  # Request more than k to ensure coverage
        payload["return_text_in_logprobs"] = False  # Only need token IDs

    # Use slime's post() with built-in retry mechanism (max 60 retries, 1s interval each)
    max_retries = getattr(args, 'teacher_max_retries', 60)
    return await post(args.rm_url, payload, max_retries=max_retries)


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. For standard OPD: stores single-token log-probs in sample.teacher_log_probs
    3. For TopK OPD: extracts topk tokens' log-probs and stores in sample.teacher_topk_log_probs
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    For TopK mode, we index the teacher's logprob distribution using the topk tokens
    selected by the student during rollout.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    # Standard OPD: single token log-probs
    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # TopK OPD: extract topk tokens' log-probs from teacher
    if getattr(args, "opd_use_topk", False):
        for sample, reward, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
            if sample.rollout_topk_tokens is None:
                # No topk tokens collected during rollout, skip
                continue

            # Get top logprobs from teacher (SGLang returns list format)
            if "input_top_logprobs" in reward["meta_info"]:
                # input_top_logprobs: [[[logprob, token_id, null], ...], ...]
                # Format: list of positions, each containing list of [logprob, token_id, null] entries
                top_logprobs_list = reward["meta_info"]["input_top_logprobs"][1:]  # Skip first position
                top_logprobs_list = top_logprobs_list[-response_length:]  # Match response length

                teacher_topk_log_probs = []
                # Statistics: track how many student tokens are in teacher's top-k for different k values
                student_in_teacher_topk_stats = {}  # {k: total_count}
                total_student_tokens = 0

                for pos_idx, (topk_tokens_at_pos, teacher_logprobs_entries) in enumerate(
                    zip(sample.rollout_topk_tokens[-response_length:], top_logprobs_list, strict=False)
                ):
                    # Parse SGLang list format: [[logprob, token_id, null], ...]
                    # Build a dict for efficient lookup: {token_id: logprob}
                    teacher_logprobs_dict = {int(entry[1]): entry[0] for entry in teacher_logprobs_entries}

                    # Get the minimum logprob from teacher as fallback for missing tokens
                    # This is more reasonable than -inf since if student's token is not in teacher's topk,
                    # it means teacher assigns very low probability to it
                    min_teacher_logprob = min(entry[0] for entry in teacher_logprobs_entries) if teacher_logprobs_entries else float('-inf')

                    # Extract teacher's log probs for student's topk tokens at this position
                    topk_log_probs_at_pos = []
                    for token_id in topk_tokens_at_pos:
                        # Get teacher's log prob for this token, use min_teacher_logprob as fallback
                        teacher_logprob = teacher_logprobs_dict.get(int(token_id), min_teacher_logprob)
                        topk_log_probs_at_pos.append(teacher_logprob)

                    teacher_topk_log_probs.append(topk_log_probs_at_pos)

                    # Statistics: check if each student token is in teacher's top-k for various k values
                    # We check k = 1, 5, 10, 20, 50, 100, etc. up to teacher's returned size
                    teacher_token_ids = [int(entry[1]) for entry in teacher_logprobs_entries]
                    for student_token in topk_tokens_at_pos:
                        total_student_tokens += 1
                        # Check coverage for different k values
                        for k in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
                            if k > len(teacher_token_ids):
                                break
                            if int(student_token) in teacher_token_ids[:k]:
                                student_in_teacher_topk_stats[k] = student_in_teacher_topk_stats.get(k, 0) + 1

                sample.teacher_topk_log_probs = teacher_topk_log_probs

                # Store statistics in sample for logging
                if total_student_tokens > 0:
                    sample.student_in_teacher_topk_ratio = {
                        f"top{k}": count / total_student_tokens
                        for k, count in student_in_teacher_topk_stats.items()
                    }
                    sample.student_in_teacher_topk_total = total_student_tokens

            else:
                # Fallback: if teacher doesn't return top_logprobs, log a warning
                # This shouldn't happen if reward_func is configured correctly
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"TopK OPD enabled but teacher response missing 'input_top_logprobs'. "
                    f"Sample {sample.index} will not have teacher_topk_log_probs."
                )

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
