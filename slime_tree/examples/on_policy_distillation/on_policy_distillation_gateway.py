import asyncio
import math
import random

import torch

from slime.utils.async_utils import run as run_async
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Confidence types that require teacher top-k logprob distribution
_TOPK_CONFIDENCE_TYPES = frozenset(
    (
        "max_logp",
        "entropy",
        "action-entropy",
        "action-max_logp",
        "delta-entropy",
        "delta-max_logp",
        "future-action-maxlogp",
    )
)


async def reward_func(args, sample, **kwargs):
    """Call teacher model via SGLang API with model selection support.

    Uses slime's http_utils.post() which includes retry logic (max 60 retries, 1s interval).
    """

    # Get model name from args if specified, otherwise use default
    teacher_model_name = getattr(args, "teacher_model_name", None)

    anneal_start_step = getattr(args, "teacher_temperature_anneal_start_step", None)
    anneal_end_step = getattr(args, "teacher_temperature_anneal_end_step", None)
    anneal_start_value = getattr(args, "teacher_temperature_anneal_start_value", None)
    anneal_end_value = getattr(args, "teacher_temperature_anneal_end_value", None)

    if all(v is not None for v in (anneal_start_step, anneal_end_step, anneal_start_value, anneal_end_value)):
        current_step = getattr(args, "current_rollout_id", 0)
        if current_step <= anneal_start_step:
            teacher_temperature = anneal_start_value
        elif current_step >= anneal_end_step:
            teacher_temperature = anneal_end_value
        else:
            ratio = (current_step - anneal_start_step) / (anneal_end_step - anneal_start_step)
            teacher_temperature = anneal_start_value + ratio * (anneal_end_value - anneal_start_value)
    else:
        teacher_temperature = getattr(args, "teacher_temperature", None)
        if teacher_temperature is None:
            teacher_temperature = args.rollout_temperature

    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": teacher_temperature,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    # Add model parameter if specified
    if teacher_model_name:
        payload["model"] = teacher_model_name

    # Request top-k logprobs when teacher_logit_y collection, RC-OPD, confidence reward (max_logp/entropy),
    # or teacher-topk KL loss is enabled
    need_topk = (
        getattr(args, "opd_dualsample_truncate_by_teacher_logit_y", False)
        or getattr(args, "opd_use_rc", False)
        or (
            getattr(args, "opd_confidence_reward_coef", 0.0) != 0.0
            and getattr(args, "opd_confidence_type", "ppl") in _TOPK_CONFIDENCE_TYPES
        )
        or getattr(args, "opd_teacher_topk_kl", False)
        or getattr(args, "opd_union_topk_kl", False)
    )
    if need_topk:
        teacher_topk_size = getattr(args, "opd_teacher_topk_size", 50)
        payload["top_logprobs_num"] = teacher_topk_size
        payload["return_text_in_logprobs"] = False

    # Use slime's post() with built-in retry mechanism (max 60 retries, 1s interval each)
    max_retries = getattr(args, "teacher_max_retries", 60)
    return await post(args.rm_url, payload, max_retries=max_retries)


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
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

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Collect teacher top-k distribution data when any feature needs it
    need_topk_sample = getattr(args, "opd_dualsample_truncate_by_teacher_logit_y", False) or getattr(
        args, "opd_use_rc", False
    )
    need_dist_topk = (
        getattr(args, "opd_confidence_reward_coef", 0.0) != 0.0
        and getattr(args, "opd_confidence_type", "ppl") in _TOPK_CONFIDENCE_TYPES
    )
    need_teacher_topk_kl = getattr(args, "opd_teacher_topk_kl", False) or getattr(args, "opd_union_topk_kl", False)
    if need_topk_sample or need_dist_topk or need_teacher_topk_kl:
        for sample, reward, response_length in zip(samples, raw_rewards, response_lengths):
            if "input_top_logprobs" not in reward["meta_info"]:
                continue
            teacher_topk_list = reward["meta_info"]["input_top_logprobs"][1:]
            teacher_topk_list = teacher_topk_list[-response_length:]

            if need_topk_sample:
                logit_y = []
                sampled_token_ids = []
                for teacher_topk_entries in teacher_topk_list:
                    teacher_logprobs = torch.tensor([e[0] for e in teacher_topk_entries], dtype=torch.float32)
                    teacher_probs = torch.exp(teacher_logprobs)
                    sampled_idx = torch.multinomial(teacher_probs, num_samples=1).item()
                    logit_y.append(teacher_logprobs[sampled_idx].item())
                    sampled_token_ids.append(teacher_topk_entries[sampled_idx][1])
                sample.teacher_logit_y = torch.tensor(logit_y, dtype=torch.float32)
                sample.teacher_sampled_tokens = sampled_token_ids

            if need_dist_topk:
                # Store teacher's own top-k log probs (sorted by teacher probability) for confidence reward
                sample.teacher_dist_topk_log_probs = [[e[0] for e in pos_entries] for pos_entries in teacher_topk_list]

            if need_teacher_topk_kl:
                # Extract teacher's top-k tokens and log probs for teacher-topk KL loss.
                # The replacement of student sampled token is deferred to loss computation
                # where we have access to training logits.
                # SGLang format: [[logprob, token_id, null], ...]
                sample.teacher_dist_topk_tokens = [
                    [int(e[1]) for e in pos_entries] for pos_entries in teacher_topk_list
                ]
                sample.teacher_dist_topk_log_probs = [[e[0] for e in pos_entries] for pos_entries in teacher_topk_list]

    # --- Teacher next-token confidence collection via tree attention ---
    # For each sample, build tree(s) combining high-entropy positions' candidates.
    # Tree structure per request:
    #   Trunk: original sequence as chain [t_0 -> t_1 -> ... -> t_{n-1}]
    #   Branches: K candidate tokens per high-entropy position, branching from abs_pos-1
    #   Probes: dummy child token per candidate, to extract P(* | prefix, D_i) from input_top_logprobs
    # When tree would exceed max_tree_tokens, splits into multiple requests sharing the same trunk
    # (leverages SGLang RadixAttention prefix caching by pinning all chunks to the same server URL).
    need_confidence = getattr(args, "opd_union_topk_confidence", False)
    if need_confidence:
        conf_k = getattr(args, "opd_union_topk_confidence_k", 5)
        entropy_k = getattr(args, "opd_teacher_topk_size", 50)  # use union's k for entropy estimation
        entropy_threshold = getattr(args, "opd_union_topk_confidence_entropy_threshold", 1.0)
        max_tree_tokens = getattr(args, "opd_union_topk_confidence_max_tree_tokens", 0)
        teacher_model_name = getattr(args, "teacher_model_name", None)
        confidence_model_name = getattr(args, "opd_union_topk_confidence_model_name", None) or teacher_model_name
        confidence_urls = getattr(args, "opd_union_topk_confidence_urls", None)
        max_retries = getattr(args, "teacher_max_retries", 60)
        url_list = confidence_urls if confidence_urls else [args.rm_url]

        # Build tree-attention requests (may be >1 per sample if splitting)
        # Each entry: (sample_idx, chunk_positions, probe_map, payload, fixed_url)
        async_queries = []

        for si, sample in enumerate(samples):
            if sample.rollout_topk_log_probs is None or sample.rollout_topk_tokens is None:
                continue

            resp_len = len(sample.rollout_topk_log_probs)
            response_length = sample.response_length
            prompt_len = len(sample.tokens) - response_length

            # Find all high-entropy positions
            high_entropy_positions = []
            for pos in range(resp_len):
                topk_lps = sample.rollout_topk_log_probs[pos][:entropy_k]
                if not topk_lps:
                    continue
                probs = [math.exp(lp) for lp in topk_lps]
                prob_sum = sum(probs)
                entropy = -sum(p / prob_sum * math.log(p / prob_sum + 1e-10) for p in probs)
                if entropy > entropy_threshold:
                    high_entropy_positions.append((pos, sample.rollout_topk_tokens[pos][:conf_k]))

            if not high_entropy_positions:
                continue

            n = len(sample.tokens)

            # Determine chunk size: how many high-entropy positions per tree request
            if max_tree_tokens > 0 and max_tree_tokens > n:
                positions_per_chunk = max((max_tree_tokens - n) // (2 * conf_k), 1)
            else:
                positions_per_chunk = len(high_entropy_positions)

            # Pin all chunks of the same sample to one URL for prefix cache reuse
            sample_url = random.choice(url_list)

            for chunk_start in range(0, len(high_entropy_positions), positions_per_chunk):
                chunk = high_entropy_positions[chunk_start : chunk_start + positions_per_chunk]

                # Build tree: trunk (original seq chain) + candidate branches + dummy probes
                input_ids = list(sample.tokens)
                parent_ids = list(range(-1, n - 1))

                # probe_map: {(pos, ci): probe_index_in_input_ids}
                probe_map = {}
                for pos, cand_tokens in chunk:
                    abs_pos = prompt_len + pos
                    branch_parent = abs_pos - 1
                    for ci, cand_token in enumerate(cand_tokens):
                        cand_idx = len(input_ids)
                        input_ids.append(cand_token)
                        parent_ids.append(branch_parent)
                        # Dummy probe child to get next-token distribution after candidate
                        probe_idx = len(input_ids)
                        input_ids.append(0)  # dummy token, value irrelevant
                        parent_ids.append(cand_idx)
                        probe_map[(pos, ci)] = probe_idx

                payload = {
                    "input_ids": input_ids,
                    "parent_ids": parent_ids,
                    "sampling_params": {"max_new_tokens": 0},
                    "return_logprob": True,
                    "logprob_start_len": 0,
                    "top_logprobs_num": 1,
                }
                if confidence_model_name:
                    payload["model"] = confidence_model_name

                async_queries.append((si, chunk, probe_map, payload, sample_url))

        # Send all queries concurrently with semaphore-based concurrency control
        if async_queries:
            sem = asyncio.Semaphore(32)

            async def _post_with_fixed_url(payload, fixed_url):
                """First try fixed_url (for prefix cache reuse), retry with random URL on failure."""
                async with sem:
                    # First attempt: use the fixed URL assigned to this sample
                    try:
                        return await post(fixed_url, payload, max_retries=1)
                    except Exception:
                        pass
                    # Subsequent retries: random URL selection
                    for attempt in range(max_retries - 1):
                        url = random.choice(url_list)
                        try:
                            return await post(url, payload, max_retries=1)
                        except Exception:
                            if attempt >= max_retries - 2:
                                raise
                            await asyncio.sleep(1)

            async def _send_all():
                return await asyncio.gather(
                    *[_post_with_fixed_url(q[3], q[4]) for q in async_queries]
                )

            results = run_async(_send_all())

            # Merge results: same sample may have multiple chunks
            sample_results = {}  # si -> (confidence_list, candidates_list)

            for (si, chunk, probe_map, _, _), result in zip(async_queries, results):
                sample = samples[si]
                resp_len = len(sample.rollout_topk_log_probs)

                if si not in sample_results:
                    sample_results[si] = ([None] * resp_len, [None] * resp_len)
                confidence_list, candidates_list = sample_results[si]

                input_top_logprobs = result["meta_info"].get("input_top_logprobs", None)
                if input_top_logprobs is None:
                    continue

                for pos, cand_tokens in chunk:
                    k_actual = len(cand_tokens)
                    conf_values = []
                    for ci in range(k_actual):
                        probe_idx = probe_map[(pos, ci)]
                        if probe_idx < len(input_top_logprobs):
                            top_entry = input_top_logprobs[probe_idx]
                            if top_entry and len(top_entry) > 0:
                                max_prob = math.exp(top_entry[0][0])
                            else:
                                max_prob = 0.0
                        else:
                            max_prob = 0.0
                        conf_values.append(max_prob)
                    confidence_list[pos] = conf_values
                    candidates_list[pos] = list(cand_tokens)

            for si, (conf_list, cand_list) in sample_results.items():
                samples[si].teacher_next_token_confidence = conf_list
                samples[si].teacher_next_token_candidates = cand_list

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
