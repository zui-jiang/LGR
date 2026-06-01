import math
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.types import RolloutBatch

from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    if args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        yield logits_chunk, tokens_chunk


def get_student_topk_post(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    with_topk: bool = False,
) -> dict[str, list[torch.Tensor]]:
    """Compute student top-k tokens/log-probs after optimizer update.

    Only returns topk data (no log_probs) to avoid overwriting the pre-training
    log_probs already stored in rollout_data.
    """
    from slime.utils.ppo_utils import vocab_parallel_topk

    assert non_loss_data
    topk_tokens_list = []
    topk_log_probs_list = []
    dump_k = args.dump_student_topk_size
    tp_group = mpu.get_tensor_model_parallel_group()

    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        topk_lp, topk_ids = vocab_parallel_topk(logits_chunk, dump_k, tp_group)
        topk_tokens_list.append(topk_ids.cpu())
        topk_log_probs_list.append(topk_lp.cpu())

    return torch.empty((0,), device=logits.device), {
        "student_topk_tokens_post": topk_tokens_list,
        "student_topk_log_probs_post": topk_log_probs_list,
    }


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    with_topk: bool = False,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    log_probs_list = []
    entropy_list = []
    topk_tokens_list = []
    topk_log_probs_list = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            mpu.get_tensor_model_parallel_group(),
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
        )

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)

        if with_topk:
            from slime.utils.ppo_utils import vocab_parallel_topk

            topk_lp, topk_ids = vocab_parallel_topk(
                logits_chunk, args.dump_student_topk_size, mpu.get_tensor_model_parallel_group()
            )
            topk_tokens_list.append(topk_ids.cpu())
            topk_log_probs_list.append(topk_lp.cpu())

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    if with_topk:
        res["student_topk_tokens_pre"] = topk_tokens_list
        res["student_topk_log_probs_pre"] = topk_log_probs_list
    return torch.empty((0,), device=logits.device), res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    return torch.empty((0,), device=logits.device), {
        "values": value_list,
    }


def compute_kwise_reverse_kl_stats(
    student_topk_log_probs: list[torch.Tensor],
    teacher_topk_log_probs: list[torch.Tensor],
    k_values: list[int],
) -> dict[str, torch.Tensor]:
    """Compute reverse KL for different k values.

    Args:
        student_topk_log_probs: List of [seq_len, k_max] tensors
        teacher_topk_log_probs: List of [seq_len, k_max] tensors
        k_values: List of k values to compute (e.g., [1, 5, 10, 20])

    Returns:
        Dict mapping f"opd_reverse_kl_k{k}" to mean KL value
    """
    stats = {}

    for k in k_values:
        k_kls = []
        for student_topk, teacher_topk in zip(student_topk_log_probs, teacher_topk_log_probs, strict=False):
            # Slice to first k tokens
            student_k = student_topk[:, :k]  # [seq_len, k]
            teacher_k = teacher_topk[:, :k]  # [seq_len, k]

            # Compute weighted KL for this k
            student_probs_k = torch.exp(student_k)
            kl_per_token_k = student_k - teacher_k
            reverse_kl_k = (student_probs_k * kl_per_token_k).sum(dim=-1)  # [seq_len]

            k_kls.append(reverse_kl_k)

        # Concatenate and compute mean
        all_kl = torch.cat(k_kls)
        stats[f"opd_reverse_kl_k{k}"] = all_kl.mean()

    return stats


def compute_topk_coverage_stats(
    sampled_tokens: list[torch.Tensor],
    topk_tokens: list[torch.Tensor],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute coverage statistics: fraction of sampled tokens that are in topk.

    Args:
        sampled_tokens: Actual sampled response tokens, list of [response_len]
        topk_tokens: TopK candidates at each position, list of [response_len, k]
        k_values: List of k values to compute coverage for (e.g., [1, 5, 10, 20]). None = skip coverage stats.

    Returns:
        Dict containing:
        - "token_in_topk_k{k}_ratio": Coverage ratio for different k values
    """
    stats = {}

    # Compute total count for ratio calculation
    total_count = 0
    for sampled, topk in zip(sampled_tokens, topk_tokens, strict=False):
        if topk is None or sampled is None:
            continue
        total_count += sampled.numel()

    # Coverage for different k values (e.g., k=1, 5, 10)
    # Skip full k coverage (redundant - should always be 100% after replacement)
    if k_values is not None:
        for k in k_values:
            k_in_topk = 0
            for sampled, topk in zip(sampled_tokens, topk_tokens, strict=False):
                if topk is None or sampled is None:
                    continue

                # Only take first k candidates
                topk_k = topk[:, :k]  # [response_len, k]
                matches_k = (sampled.unsqueeze(-1) == topk_k).any(dim=-1)
                k_in_topk += matches_k.sum().item()

            stats[f"token_in_topk_k{k}_ratio"] = k_in_topk / max(1, total_count)

    return stats


def apply_opd_topk_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
    original_loss_masks: list[torch.Tensor] | None = None,
) -> None:
    """Apply TopK OPD KL penalty to advantages using weighted KL estimation.

    Computes KL divergence using topk candidate tokens:
        KL ≈ Σ_i p_student(token_i) * [log p_student(token_i) - log p_teacher(token_i)]

    This provides a better approximation of true KL than single-token sampling.

    Args:
        args: Configuration with opd_topk_size, opd_kl_coef, etc.
        rollout_data: Must contain "rollout_topk_log_probs" and "teacher_topk_log_probs"
        advantages: List of advantage tensors to modify in-place
        student_log_probs: Not used in topk mode (kept for API compatibility)
    """
    # Get topk log probs from rollout data
    student_topk_log_probs = rollout_data.get("rollout_topk_log_probs")
    teacher_topk_log_probs = rollout_data.get("teacher_topk_log_probs")

    if student_topk_log_probs is None or teacher_topk_log_probs is None:
        raise ValueError(
            "TopK OPD mode requires 'rollout_topk_log_probs' and 'teacher_topk_log_probs' in rollout_data, "
            "but one or both are missing. Make sure --opd-use-topk is enabled during rollout."
        )

    device = advantages[0].device

    # Convert to tensors on correct device
    # Each element is a list of lists: [seq_len, k]
    student_topk_log_probs = [
        torch.tensor(lp, dtype=torch.float32, device=device) if not isinstance(lp, torch.Tensor) else lp.to(device)
        for lp in student_topk_log_probs
    ]
    teacher_topk_log_probs = [
        torch.tensor(lp, dtype=torch.float32, device=device) if not isinstance(lp, torch.Tensor) else lp.to(device)
        for lp in teacher_topk_log_probs
    ]

    reverse_kls = []
    gopd_kls = []

    for i, adv in enumerate(advantages):
        # student_topk: [seq_len, k], teacher_topk: [seq_len, k]
        # These are already log probs from full vocabulary softmax
        student_topk = student_topk_log_probs[i]
        teacher_topk = teacher_topk_log_probs[i]

        # Compute probability weights from student's topk log probs
        # These are already normalized over full vocab, just convert to probs
        student_probs = torch.exp(student_topk)  # [seq_len, k]

        # Compute weighted KL: Σ_i p(i) * [log p_student(i) - log p_teacher(i)]
        # No need for log_softmax - already have log probs from full vocab
        kl_per_token = student_topk - teacher_topk  # [seq_len, k]
        reverse_kl = (student_probs * kl_per_token).sum(dim=-1)  # [seq_len]

        reverse_kls.append(reverse_kl)

        # For now, topk mode doesn't support G-OPD (lambda != 1.0)
        # because ref_log_probs would also need topk version
        gopd_kl = reverse_kl
        gopd_kls.append(gopd_kl)

        # Apply KL penalty to advantages
        # Use the same coefficient system as standard OPD
        use_sign_penalty = getattr(args, "opd_use_sign_penalty", False)

        sign_coef = torch.where(
            reverse_kl > 0,
            torch.tensor(args.opd_kl_coef_positive, device=device, dtype=torch.float32),
            torch.where(
                reverse_kl < 0,
                torch.tensor(args.opd_kl_coef_negative, device=device, dtype=torch.float32),
                torch.tensor(args.opd_kl_coef_zero, device=device, dtype=torch.float32),
            ),
        )

        if use_sign_penalty:
            advantages[i] = adv - sign_coef
        else:
            advantages[i] = adv - sign_coef * gopd_kl

    # Store metrics
    rollout_data["opd_reverse_kl"] = reverse_kls
    rollout_data["opd_gopd_kl"] = gopd_kls

    # Prefix-position reverse-KL statistics.
    # Use the original (pre-prefix-mask) loss_masks so that per-position stats
    # still reflect the true KL distribution even when --opd-kl-prefix-mask-len is set.
    _topk_stat_masks = original_loss_masks if original_loss_masks is not None else rollout_data.get("loss_masks")
    rollout_data.update(compute_prefix_reverse_kl_stats(reverse_kls, _topk_stat_masks))

    # Compute k-wise statistics for every k from 1 to k_max
    k_max = student_topk_log_probs[0].size(-1)  # Get max k from data
    k_values = list(range(1, k_max + 1))  # [1, 2, 3, ..., k_max]
    kwise_stats = compute_kwise_reverse_kl_stats(student_topk_log_probs, teacher_topk_log_probs, k_values)
    rollout_data.update(kwise_stats)

    # Compute topk coverage statistics
    if "rollout_topk_tokens" in rollout_data and "tokens" in rollout_data:
        # Extract response tokens from full token sequences
        sampled_response_tokens = []
        for tokens, response_length in zip(rollout_data["tokens"], rollout_data["response_lengths"], strict=False):
            response_tokens = tokens[-response_length:]  # Slice to response part
            sampled_response_tokens.append(response_tokens)

        # Select k values for coverage statistics
        selected_k_values = [1]
        if k_max >= 5:
            selected_k_values.append(5)
        if k_max >= 10:
            selected_k_values.append(10)
        # Add quartiles of k_max
        for ratio in [0.25, 0.5, 0.75]:
            k_val = int(k_max * ratio)
            if k_val > 1 and k_val not in selected_k_values:
                selected_k_values.append(k_val)
        selected_k_values.append(k_max)  # Full k
        selected_k_values = sorted(set(selected_k_values))

        coverage_stats = compute_topk_coverage_stats(
            sampled_response_tokens,
            rollout_data["rollout_topk_tokens"],
            k_values=selected_k_values,
        )
        rollout_data["opd_topk_coverage"] = coverage_stats

    # Compute topk replacement statistics
    if "topk_replacement_counts" in rollout_data:
        replacement_counts = rollout_data["topk_replacement_counts"]
        total_replacements = sum(replacement_counts)
        total_tokens = sum(adv.numel() for adv in advantages)  # Total response tokens

        rollout_data["opd_topk_replacement_stats"] = {
            "replacement_ratio": total_replacements / max(1, total_tokens),
        }

    # Compute sign statistics
    if reverse_kls:
        all_reverse_kl = torch.cat([rk if isinstance(rk, torch.Tensor) else torch.tensor(rk) for rk in reverse_kls])
        positive_count = (all_reverse_kl > 0).sum().item()
        negative_count = (all_reverse_kl < 0).sum().item()
        zero_count = (all_reverse_kl == 0).sum().item()
        total_count = all_reverse_kl.numel()

        rollout_data["opd_sign_stats"] = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "zero_count": zero_count,
            "positive_ratio": positive_count / max(1, total_count),
            "negative_ratio": negative_count / max(1, total_count),
        }

    # Collect student-in-teacher-topk statistics from samples
    # This tracks how many student topk tokens are in teacher's top-k for various k values
    if "student_in_teacher_topk_stats" in rollout_data:
        stats_list = rollout_data["student_in_teacher_topk_stats"]
        # Filter out None values first
        valid_stats = [s for s in stats_list if s]

        if valid_stats:  # Only process if we have valid data
            # Aggregate across all samples
            # Each sample has: {"top1": ratio, "top5": ratio, ...}
            aggregated_stats = {}

            # Collect all unique k values from valid stats
            all_k_values = set()
            for sample_stats in valid_stats:
                all_k_values.update(sample_stats.keys())

            # Average ratios across samples for each k value
            for k_name in sorted(all_k_values):
                ratios = [s.get(k_name, 0.0) for s in valid_stats]
                if ratios:
                    aggregated_stats[f"student_in_teacher_{k_name}_ratio"] = sum(ratios) / len(ratios)

            # Only set the field if we have aggregated data
            if aggregated_stats:
                rollout_data["opd_student_teacher_alignment"] = aggregated_stats


_OPD_REVERSE_KL_PREFIX_LENGTHS = [1000, 2000, 4000, 8000, 16000]


def compute_prefix_reverse_kl_stats(
    reverse_kls: list[torch.Tensor],
    loss_masks: list[torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Compute mean reverse-KL over the first N response tokens for several prefix lengths.

    Args:
        reverse_kls: Per-sample per-token reverse-KL tensors.
        loss_masks: Per-sample loss masks (1 = valid token). If None, all tokens are treated as valid.

    Returns:
        Dict mapping ``opd_reverse_kl_prefix_{n}`` to the mean KL value over
        all valid tokens within the first *n* positions across all samples.
    """
    stats = {}
    for prefix_len in _OPD_REVERSE_KL_PREFIX_LENGTHS:
        kl_values = []
        for idx, rk in enumerate(reverse_kls):
            end = min(prefix_len, rk.shape[0])
            if end <= 0:
                continue
            rk_prefix = rk[:end]
            if loss_masks is not None and idx < len(loss_masks):
                mask = loss_masks[idx][:end].to(dtype=rk_prefix.dtype, device=rk_prefix.device)
                valid = mask.bool()
                if valid.any():
                    kl_values.append(rk_prefix[valid])
            else:
                kl_values.append(rk_prefix)
        if kl_values:
            stats[f"opd_reverse_kl_prefix_{prefix_len}"] = torch.cat(kl_values).mean()
    return stats


def _compute_truncate_mask(
    t_logit_y: torch.Tensor,
    loss_mask_orig: torch.Tensor,
    truncate_window_size: int,
    truncate_fixed_threshold: float | None,
) -> torch.Tensor:
    """Compute a truncation mask based on teacher_logit_y confidence.

    Scans t_logit_y to find the first position where the teacher becomes
    uncertain, then zeros out all positions from that point onward.

    Lookahead mode (truncate_fixed_threshold is not None):
        Find first pos where mean(t_logit_y[pos:pos+window]) < threshold.
    Lookback mode (truncate_fixed_threshold is None):
        Find first pos where t_logit_y[pos] < rolling mean of preceding window.

    Returns a mask of shape [seq_len] where 1 = keep, 0 = truncate.
    """
    seq_len = loss_mask_orig.shape[0]
    truncate_mask = torch.ones(seq_len, dtype=loss_mask_orig.dtype, device=loss_mask_orig.device)
    truncation_point = seq_len  # default: no truncation

    if truncate_fixed_threshold is not None:
        # Lookahead mode: find first pos where mean(t_logit_y[pos:pos+W]) < threshold
        for pos in range(seq_len):
            window_end = min(seq_len, pos + truncate_window_size)
            lookahead_mean = t_logit_y[pos:window_end].mean().item()
            if lookahead_mean < truncate_fixed_threshold:
                truncation_point = pos
                break
    else:
        # Lookback mode: rolling mean of preceding window
        for pos in range(1, seq_len):
            window_start = max(0, pos - truncate_window_size)
            rolling_threshold = t_logit_y[window_start:pos].mean().item()
            if t_logit_y[pos].item() < rolling_threshold:
                truncation_point = pos
                break

    truncate_mask[truncation_point:] = 0
    return truncate_mask


def compute_teacher_confidence(
    confidence_type: str,
    teacher_log_probs_i: torch.Tensor | None,
    teacher_dist_topk_i: torch.Tensor | None,
    clip_min: float | None = None,
) -> torch.Tensor:
    """Compute per-token teacher confidence signal.

    All returned signals satisfy: max value = 0 (achieved when teacher is perfectly confident).
    Values are clipped to [clip_min, 0] when clip_min is specified.

    Args:
        confidence_type: One of "logpt_x", "ppl", "near-ppl", "max_logp", "entropy".
        teacher_log_probs_i: [seq_len] log-prob of student's sampled token under teacher.
            Required for 'logpt_x', 'ppl', and 'near-ppl' modes.
        teacher_dist_topk_i: [seq_len, k] teacher's own top-k log probs sorted by teacher
            probability. Required for 'max_logp' and 'entropy' modes.
        clip_min: If specified, clamp the output to [clip_min, 0].

    Returns:
        [seq_len] confidence tensor. For non-delta types the range is (-inf, 0] (higher = more confident).
        Delta types can be positive — they measure the *change* in teacher confidence caused by x_t.
        - logpt_x:        log p_t(x_t), max=0 when p_t(x_t)=1
        - ppl:            1 - prefix_PPL_t, where prefix_PPL_t = exp(-mean_{i<=t} log p_t(x_i));
                          max=0 when PPL=1, goes to -inf as PPL grows
        - near-ppl:       1 - near_PPL_t, where near_PPL_t = exp(-mean_{i=t-w+1}^{t} log p_t(x_i))
                          with w = min(t+1, 128); uses only the previous 128 tokens
        - max_logp:       log p_t(top1 token), max=0 when top-1 prob=1
        - entropy:        -H(teacher_dist_t), max=0 when distribution is a delta
        - action-entropy:  -H(pi_T|x_{<t+1}): teacher entropy *after* seeing x_t (state reward
                           shifted by one); last token is 0
        - action-max_logp: max_logp[t+1]: teacher top-1 confidence *after* seeing x_t;
                           last token is 0
        - delta-entropy:   H(pi_T|x_{<t}) - H(pi_T|x_{<t},x_t) = entropy[t] - entropy[t+1];
                           positive when x_t sharpens the teacher's next-step prediction;
                           last token is 0 (no next position)
        - delta-max_logp:  max_logp[t+1] - max_logp[t]; positive when x_t increases teacher's
                           top-1 confidence for the next position; last token is 0
    """
    if confidence_type == "logpt_x":
        if teacher_log_probs_i is None:
            raise ValueError("opd_confidence_type='logpt_x' requires teacher_log_probs in rollout_data.")
        conf = teacher_log_probs_i
    elif confidence_type == "ppl":
        if teacher_log_probs_i is None:
            raise ValueError("opd_confidence_type='ppl' requires teacher_log_probs in rollout_data.")
        # prefix_PPL_t = exp(-1/(t+1) * sum_{i<=t} log p_teacher(x_i)), in [1, +inf)
        # confidence = 1 - prefix_PPL_t, in (-inf, 0], max=0 when PPL=1
        positions = torch.arange(
            1, teacher_log_probs_i.size(0) + 1, dtype=torch.float32, device=teacher_log_probs_i.device
        )
        cumsum = torch.cumsum(teacher_log_probs_i, dim=0)
        prefix_ppl = torch.exp(-cumsum / positions)  # [seq_len]
        conf = 1.0 - prefix_ppl
    elif confidence_type == "near-ppl":
        if teacher_log_probs_i is None:
            raise ValueError("opd_confidence_type='near-ppl' requires teacher_log_probs in rollout_data.")
        # near_PPL_t = exp(-1/w * sum_{i=t-w+1}^{t} log p_teacher(x_i)), where w = min(t+1, 128)
        # confidence = 1 - near_PPL_t, in (-inf, 0], max=0 when PPL=1
        seq_len = teacher_log_probs_i.size(0)
        window = 128
        # padded[k] = sum_{i=0}^{k-1} log_probs[i], so padded[0]=0, padded[t+1]=cumsum[t]
        padded = torch.cat([teacher_log_probs_i.new_zeros(1), torch.cumsum(teacher_log_probs_i, dim=0)])  # [seq_len+1]
        idx = torch.arange(seq_len, device=teacher_log_probs_i.device)
        # window sum from max(0, t-window+1) to t
        window_sum = padded[idx + 1] - padded[(idx - window + 1).clamp(min=0)]
        win_sizes = (idx + 1).clamp(max=window).to(dtype=torch.float32)
        near_ppl = torch.exp(-window_sum / win_sizes)  # [seq_len]
        conf = 1.0 - near_ppl
    elif confidence_type == "max_logp":
        if teacher_dist_topk_i is None:
            raise ValueError(
                "opd_confidence_type='max_logp' requires teacher_dist_topk_log_probs in rollout_data. "
                "Set --opd-teacher-topk-size > 0 and use a reward_func that requests top-k logprobs."
            )
        conf = teacher_dist_topk_i[:, 0]
    elif confidence_type == "entropy":
        if teacher_dist_topk_i is None:
            raise ValueError(
                "opd_confidence_type='entropy' requires teacher_dist_topk_log_probs in rollout_data. "
                "Set --opd-teacher-topk-size > 0 and use a reward_func that requests top-k logprobs."
            )
        probs = torch.exp(teacher_dist_topk_i)  # [seq_len, k]
        entropy = -(probs * teacher_dist_topk_i).sum(dim=-1)  # [seq_len], >= 0
        conf = -entropy  # in (-inf, 0]
    elif confidence_type == "action-entropy":
        if teacher_dist_topk_i is None:
            raise ValueError(
                "opd_confidence_type='action-entropy' requires teacher_dist_topk_log_probs in rollout_data. "
                "Set --opd-teacher-topk-size > 0 and use a reward_func that requests top-k logprobs."
            )
        probs = torch.exp(teacher_dist_topk_i)  # [seq_len, k]
        entropy = -(probs * teacher_dist_topk_i).sum(dim=-1)  # [seq_len], >= 0
        # conf[t] = -H(pi_T|x_{<t+1}): teacher's entropy *after* seeing x_t
        # last token gets 0 (no next position)
        conf = torch.cat([-entropy[1:], entropy.new_zeros(1)])
    elif confidence_type == "action-max_logp":
        if teacher_dist_topk_i is None:
            raise ValueError(
                "opd_confidence_type='action-max_logp' requires teacher_dist_topk_log_probs in rollout_data. "
                "Set --opd-teacher-topk-size > 0 and use a reward_func that requests top-k logprobs."
            )
        max_logp = teacher_dist_topk_i[:, 0]  # [seq_len]
        # conf[t] = max_logp[t+1]: teacher's top-1 confidence *after* seeing x_t
        # last token gets 0 (no next position)
        conf = torch.cat([max_logp[1:], max_logp.new_zeros(1)])
    elif confidence_type == "delta-entropy":
        if teacher_dist_topk_i is None:
            raise ValueError(
                "opd_confidence_type='delta-entropy' requires teacher_dist_topk_log_probs in rollout_data. "
                "Set --opd-teacher-topk-size > 0 and use a reward_func that requests top-k logprobs."
            )
        probs = torch.exp(teacher_dist_topk_i)  # [seq_len, k]
        entropy = -(probs * teacher_dist_topk_i).sum(dim=-1)  # [seq_len], >= 0
        # delta[t] = H(pi_T|x_{<t}) - H(pi_T|x_{<t},x_t) = entropy[t] - entropy[t+1]
        # positive means x_t sharpened the teacher's next-step prediction (entropy decreased)
        # last token gets 0 (no next position to evaluate)
        conf = torch.cat([entropy[:-1] - entropy[1:], entropy.new_zeros(1)])
    elif confidence_type == "delta-max_logp":
        if teacher_dist_topk_i is None:
            raise ValueError(
                "opd_confidence_type='delta-max_logp' requires teacher_dist_topk_log_probs in rollout_data. "
                "Set --opd-teacher-topk-size > 0 and use a reward_func that requests top-k logprobs."
            )
        max_logp = teacher_dist_topk_i[:, 0]  # [seq_len]
        # delta[t] = max_logp[t+1] - max_logp[t]
        # positive means x_t increased teacher's top-1 confidence for the next position
        # last token gets 0 (no next position to evaluate)
        conf = torch.cat([max_logp[1:] - max_logp[:-1], max_logp.new_zeros(1)])
    elif confidence_type == "future-ppl":
        if teacher_log_probs_i is None:
            raise ValueError("opd_confidence_type='future-ppl' requires teacher_log_probs in rollout_data.")
        # Raw per-token signal: teacher log prob at each position.
        # The forward-looking PPL window and gamma decay are applied in
        # apply_opd_confidence_reward_to_advantages after this function returns.
        conf = teacher_log_probs_i
    elif confidence_type == "future-action-maxlogp":
        if teacher_dist_topk_i is None:
            raise ValueError(
                "opd_confidence_type='future-action-maxlogp' requires teacher_dist_topk_log_probs in rollout_data. "
                "Set --opd-teacher-topk-size > 0 and use a reward_func that requests top-k logprobs."
            )
        # raw[t] = exp(max_logp[t+1]): teacher's confidence about the next token.
        # Higher value means teacher is more certain. Last token gets 0.
        max_prob = torch.exp(teacher_dist_topk_i[:, 0])  # [seq_len]
        conf = torch.cat([max_prob[1:], max_prob.new_zeros(1)])
    else:
        raise ValueError(
            f"Unknown opd_confidence_type: '{confidence_type}'. Choose from 'logpt_x', 'ppl', 'near-ppl', "
            f"'max_logp', 'entropy', 'action-entropy', 'action-max_logp', 'delta-entropy', 'delta-max_logp', "
            f"'future-ppl', 'future-action-maxlogp'."
        )

    if clip_min is not None:
        conf = conf.clamp(min=clip_min)
    return conf


def _confidence_future_sum(conf: torch.Tensor, gamma: float, window: int | None = None) -> torch.Tensor:
    """Forward-looking gamma-decayed sum for confidence reward.

    For 'future-ppl' type, raw[t] is the forward-looking PPL over [t+1, t+window]:
        raw[t] = exp(-1/W * sum_{s=t+1}^{min(t+W, T)} log_p_teacher(x_s))
    Then gamma decay: result[t] = sum_{s=t}^{T} gamma^(s-t) * raw[s]

    For other future types, raw[t] is the per-token signal and:
        result[t] = sum_{s=t+1}^{T} gamma^(s-t-1) * raw[s]

    Args:
        conf: [seq_len] raw per-token confidence signal.
        gamma: Discount factor for future sum.
        window: For future-ppl, this is the PPL window size N. None means no window limit.

    Returns:
        [seq_len] gamma-decayed future sum.
    """
    T = conf.shape[0]
    result = torch.zeros_like(conf)

    if window is None:
        # Backward scan, O(T): result[t] = sum_{s=t+1}^{T-1} gamma^(s-t-1) * conf[s]
        running = conf.new_zeros(())
        for t in range(T - 1, -1, -1):
            result[t] = running
            running = conf[t] + gamma * running
    else:
        # Window-limited: result[t] = sum_{s=t+1}^{min(t+W, T-1)} gamma^(s-t-1) * conf[s]
        for t in range(T):
            end = min(t + 1 + window, T)
            if t + 1 >= end:
                continue
            offsets = torch.arange(end - t - 1, device=conf.device, dtype=conf.dtype)
            weights = gamma**offsets
            result[t] = (weights * conf[t + 1 : end]).sum()

    # Normalize by (1 - gamma) so the sum stays O(1) instead of O(1/(1-gamma))
    if gamma < 1.0:
        result = result * (1 - gamma)

    return result


def _compute_future_ppl_raw(
    teacher_log_probs_i: torch.Tensor,
    ppl_window: int | None,
) -> torch.Tensor:
    """Compute negative forward-looking PPL at each position.

    raw[t] = -exp(-1/W * sum_{s=t+1}^{min(t+W, T)} log_p_teacher(x_s))

    Higher value (closer to 0) = teacher more confident about the future.
    Lower value (more negative) = teacher less confident.

    Args:
        teacher_log_probs_i: [seq_len] teacher log probs.
        ppl_window: Window size W. None uses all remaining tokens.

    Returns:
        [seq_len] negative PPL values (<= -1, higher = more confident).
    """
    T = teacher_log_probs_i.size(0)
    # Prefix sum for efficient window sum: padded[k] = sum_{i=0}^{k-1} log_probs[i]
    padded = torch.cat([teacher_log_probs_i.new_zeros(1), torch.cumsum(teacher_log_probs_i, dim=0)])  # [T+1]

    raw = -torch.ones(T, dtype=teacher_log_probs_i.dtype, device=teacher_log_probs_i.device)
    for t in range(T):
        start = t + 1
        if ppl_window is not None:
            end = min(start + ppl_window, T)
        else:
            end = T
        if start >= end:
            continue  # No future tokens, -PPL = -1
        window_sum = padded[end] - padded[start]
        n = end - start
        raw[t] = -torch.exp(-window_sum / n)  # -PPL <= -1

    return raw


def apply_opd_confidence_reward_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
) -> None:
    """Add per-token teacher confidence reward to advantages (in-place).

    Modifies advantages as:
        A_t' = A_t + conf_coef * confidence_t

    For 'future-ppl' and 'future-action-maxlogp' types, applies gamma-decayed
    future sum after computing the raw per-token signal.

    Args:
        args: Configuration with opd_confidence_reward_coef and opd_confidence_type.
        rollout_data: Must contain "teacher_log_probs" (for 'ppl') or
            "teacher_dist_topk_log_probs" (for 'max_logp'/'entropy').
        advantages: Per-sample advantage tensors, modified in-place.
    """
    conf_coef = getattr(args, "opd_confidence_reward_coef", 0.0)
    confidence_type = getattr(args, "opd_confidence_type", "logpt_x")
    clip_min = getattr(args, "opd_confidence_clip_min", None)
    future_gamma = getattr(args, "opd_confidence_future_gamma", 1.0)
    ppl_window = getattr(args, "opd_confidence_ppl_window", None)
    device = advantages[0].device

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    teacher_dist_topk = rollout_data.get("teacher_dist_topk_log_probs")

    confidence_list = []
    for i, adv in enumerate(advantages):
        t_log_probs_i = None
        if teacher_log_probs is not None:
            lp = teacher_log_probs[i]
            t_log_probs_i = (
                lp.to(device) if isinstance(lp, torch.Tensor) else torch.tensor(lp, dtype=torch.float32, device=device)
            )

        t_topk_i = None
        if teacher_dist_topk is not None:
            topk = teacher_dist_topk[i]
            t_topk_i = (
                topk.to(device)
                if isinstance(topk, torch.Tensor)
                else torch.tensor(topk, dtype=torch.float32, device=device)
            )

        if confidence_type == "future-ppl":
            # Step 1: Compute forward-looking PPL at each position
            raw_ppl = _compute_future_ppl_raw(t_log_probs_i, ppl_window)
            # Step 2: Apply gamma-decayed future sum on the PPL values
            conf = _confidence_future_sum(raw_ppl, future_gamma)
        else:
            conf = compute_teacher_confidence(confidence_type, t_log_probs_i, t_topk_i, clip_min=clip_min)
            # Apply gamma decay for future-action-maxlogp
            if confidence_type == "future-action-maxlogp":
                conf = _confidence_future_sum(conf, future_gamma)

        if clip_min is not None and confidence_type in ("future-ppl", "future-action-maxlogp"):
            conf = conf.clamp(min=clip_min)

        confidence_list.append(conf)
        advantages[i] = adv + conf_coef * conf

    rollout_data["opd_teacher_confidence"] = confidence_list


def apply_opd_union_topk_confidence_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
) -> None:
    """Apply union-topk confidence signal to advantages via sampled token only.

    Instead of computing confidence bonus inside compute_union_topk_kl (which flows
    gradients through all tokens in the union set), this function adds the confidence
    bonus directly to advantages. Since policy gradient loss only differentiates
    through the sampled token's log probability, this ensures only sampled tokens
    receive gradient — matching standard OPD behavior.

    For each high-entropy position, finds whether the sampled token is among the K
    candidates. If so, uses its (normalized) confidence value as the bonus.

    Args:
        args: Configuration with opd_union_topk_confidence_via_advantage_coef, etc.
        rollout_data: Must contain "teacher_next_token_confidence",
            "teacher_next_token_candidates", "tokens", "response_lengths".
        advantages: Per-sample advantage tensors, modified in-place.
    """
    conf_data = rollout_data.get("teacher_next_token_confidence")
    cand_data = rollout_data.get("teacher_next_token_candidates")
    tokens_list = rollout_data.get("tokens")
    response_lengths = rollout_data.get("response_lengths")

    if conf_data is None or cand_data is None or tokens_list is None:
        return

    conf_coef = getattr(args, "opd_union_topk_confidence_via_advantage_coef", 0.1)
    normalize_std = getattr(args, "opd_union_topk_confidence_normalize_std", False)

    confidence_applied_count = 0
    confidence_total_count = 0
    all_conf_raw_values = []  # raw confidence values at matched positions (for metrics)
    bonus_list = []  # per-sample bonus tensors (for metrics)

    for i, adv in enumerate(advantages):
        device = adv.device
        resp_len = adv.shape[0]

        conf_sample = conf_data[i] if i < len(conf_data) else None
        cand_sample = cand_data[i] if i < len(cand_data) else None

        if conf_sample is None or cand_sample is None:
            bonus_list.append(torch.zeros(resp_len, device=device, dtype=torch.float32))
            continue

        # Get sampled response tokens
        sample_tokens = tokens_list[i][-response_lengths[i]:]

        conf_bonus = torch.zeros(resp_len, device=device, dtype=torch.float32)

        for pos in range(min(resp_len, len(conf_sample))):
            confidence_total_count += 1

            if conf_sample[pos] is None:
                continue

            conf_vals = torch.tensor(conf_sample[pos], device=device, dtype=torch.float32)
            cand_tokens = cand_sample[pos]

            # Find sampled token in candidates
            sampled_token = sample_tokens[pos].item()
            matched_ci = None
            for ci, cand_token in enumerate(cand_tokens):
                if cand_token == sampled_token:
                    matched_ci = ci
                    break

            if matched_ci is None:
                continue

            confidence_applied_count += 1
            all_conf_raw_values.append(conf_vals[matched_ci].item())

            # Normalize: subtract mean across all K candidates
            normalized = conf_vals - conf_vals.mean()
            if normalize_std:
                std = conf_vals.std()
                if std > 1e-8:
                    normalized = normalized / std

            conf_bonus[pos] = normalized[matched_ci]

        advantages[i] = adv + conf_coef * conf_bonus
        bonus_list.append(conf_bonus.detach())

    # Store metrics for wandb reporting
    rollout_data["opd_union_topk_conf_adv_bonus"] = bonus_list
    rollout_data["opd_union_topk_conf_adv_metrics"] = {
        "applied_count": confidence_applied_count,
        "total_count": confidence_total_count,
        "raw_values": all_conf_raw_values,
    }


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply G-OPD (Generalized On-Policy Distillation) KL penalty to advantages.

    Implements the G-OPD advantage function from paper "Learning beyond Teacher:
    Generalized On-Policy Distillation with Reward Extrapolation" (arXiv:2602.12125v1).

    G-OPD Advantage:
        A^{G-OPD}_t = A_t - β * [(log π_θ - log π*) + (λ-1) * (log π_ref - log π*)]

    Where:
        - π_θ: student model
        - π*: teacher RL model
        - π_ref: reference model (student ref or teacher base)
        - λ: reward scaling factor (opd_lambda)
        - β: KL coefficient (opd_kl_coef)

    When λ = 1.0, reduces to standard OPD.
    When λ > 1.0 (ExOPD), enables reward extrapolation to surpass teacher.

    TopK OPD Mode:
        When --opd-use-topk is enabled, uses weighted KL estimation:
        KL ≈ Σ_i p_student(token_i) * [log p_student(token_i) - log p_teacher(token_i)]
        where the sum is over topk candidate tokens.

    Args:
        args: Configuration containing:
            - opd_kl_coef: KL penalty coefficient (β)
            - opd_lambda: reward scaling factor (λ), default 1.0
            - opd_use_reward_correction: use teacher_base as ref (strong-to-weak)
            - opd_use_topk: enable topk mode for better KL estimation
            - opd_topk_size: number of topk tokens (k)
        rollout_data: Dict containing "teacher_log_probs" and optionally
            "teacher_base_log_probs" (if reward correction enabled).
            For topk mode: "rollout_topk_log_probs", "teacher_topk_log_probs"
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of student log-probability tensors.

    References:
        Paper: https://arxiv.org/abs/2602.12125v1
        Standard OPD: https://github.com/thinking-machines-lab/tinker-cookbook
    """

    if student_log_probs is None:
        return

    # Apply prefix mask to loss_masks: zero out the first N response tokens so they
    # are excluded from normalization and policy gradient loss.
    # Save a copy of the original masks *before* modification so that prefix-position
    # KL statistics still reflect the true per-position KL distribution.
    prefix_mask_len = getattr(args, "opd_kl_prefix_mask_len", None)
    original_loss_masks = None
    if prefix_mask_len is not None and prefix_mask_len > 0:
        loss_masks = rollout_data.get("loss_masks")
        if loss_masks is not None:
            original_loss_masks = [m.clone() for m in loss_masks]
            for i, mask in enumerate(loss_masks):
                if prefix_mask_len < mask.shape[0]:
                    mask[:prefix_mask_len] = 0

    # REOPOLD: Relaxed On-Policy Distillation
    # arXiv:2603.11137 — replaces vanilla OPD KL with clipped rewards + two-phase training
    reopold_enabled = getattr(args, "reopold", False)
    if reopold_enabled:
        lam = args.reopold_lambda
        clip_threshold = math.log(lam / (1.0 - lam))  # e.g. log(0.3/0.7) ≈ -0.847
        current_step = getattr(args, "reopold_current_rollout_id", 0)
        switch_step = args.reopold_switch_step
        if switch_step is None:
            switch_step = max(1, getattr(args, "num_rollout", 300) // 3)
        current_phase = 1 if current_step < switch_step else 2
        args.reopold_current_phase = current_phase

        teacher_log_probs = rollout_data.get("teacher_log_probs")
        if teacher_log_probs is None:
            raise ValueError("REOPOLD requires teacher_log_probs in rollout_data.")
        device = student_log_probs[0].device
        teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

        reverse_kls = []
        for i, adv in enumerate(advantages):
            reward = student_log_probs[i] - teacher_log_probs[i]  # R = log(πT/πθ) = -(reverse_kl)
            reward = -reward  # R_i,t = log(πT/πθ)
            reward_clipped = torch.clamp(reward, min=clip_threshold)  # Eq(5): mixture-based clipping
            if current_phase == 1:
                # Phase I (Exploration): only backprop through tokens above clip threshold
                phase_mask = (reward >= clip_threshold).float()  # Eq(9)
                effective_reward = reward_clipped * phase_mask
            else:
                # Phase II (Refinement): entropy mask applied later in policy_loss_function
                effective_reward = reward_clipped
            advantages[i] = adv + args.opd_kl_coef * effective_reward
            reverse_kls.append(student_log_probs[i] - teacher_log_probs[i])

        rollout_data["opd_reverse_kl"] = reverse_kls
        return

    # Check if using topk mode
    use_topk = getattr(args, "opd_use_topk", False)

    if use_topk:
        # TopK OPD mode: use weighted KL estimation
        apply_opd_topk_kl_to_advantages(args, rollout_data, advantages, student_log_probs, original_loss_masks)
        return

    # Standard OPD mode (original implementation)
    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    device = student_log_probs[0].device
    teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

    # G-OPD parameters
    lambda_scale = getattr(args, "opd_lambda", 1.0)
    use_reward_correction = getattr(args, "opd_use_reward_correction", False)

    # 选择参考模型 (π_ref)
    if lambda_scale == 1.0:
        # 标准 OPD: 不需要 ref_log_probs
        ref_log_probs = None
    else:
        # G-OPD (λ ≠ 1): 需要 ref_log_probs
        if use_reward_correction:
            ref_log_probs = rollout_data.get("teacher_base_log_probs")
            if ref_log_probs is None:
                raise ValueError(
                    "G-OPD with reward correction (opd_use_reward_correction=True) requires "
                    "teacher_base_log_probs, but it is missing. Make sure --teacher-base-model-name "
                    "is set and post_process_rewards extracts teacher_base_log_probs."
                )
        else:
            # 默认: 使用 student 的 ref_log_probs
            ref_log_probs = rollout_data.get("ref_log_probs")
            if ref_log_probs is None:
                raise ValueError(
                    f"G-OPD with lambda={lambda_scale} requires ref_log_probs from student's "
                    "reference model, but it is missing. Make sure kl_coef > 0 to enable ref model."
                )

        ref_log_probs = [r.to(device=device) for r in ref_log_probs]

    # 获取符号惩罚模式配置（提到循环外避免重复）
    use_sign_penalty = getattr(args, "opd_use_sign_penalty", False)

    reverse_kls = []
    gopd_kls = []

    # Truncation statistics (collected across all samples when teacher truncation is enabled)
    truncated_positions = 0
    total_active_positions = 0

    for i, adv in enumerate(advantages):
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]
        reverse_kls.append(reverse_kl)

        if lambda_scale == 1.0:
            gopd_kl = reverse_kl
        else:
            # G-OPD: A = A - β * [(log π_θ - log π*) + (λ-1) * (log π_ref - log π*)]
            ref_teacher_diff = ref_log_probs[i] - teacher_log_probs[i]
            gopd_kl = reverse_kl + (lambda_scale - 1.0) * ref_teacher_diff

        gopd_kls.append(gopd_kl)

        # 根据 reverse_kl 符号选择对应的值（系数或固定惩罚）
        sign_coef = torch.where(
            reverse_kl > 0,
            args.opd_kl_coef_positive,
            torch.where(reverse_kl < 0, args.opd_kl_coef_negative, args.opd_kl_coef_zero),
        )

        # Apply teacher_logit_y truncation to KL penalty if enabled.
        truncate_by_teacher = getattr(args, "opd_dualsample_truncate_by_teacher_logit_y", False)
        teacher_logit_y_list = rollout_data.get("teacher_logit_y", None)
        if truncate_by_teacher and teacher_logit_y_list is not None:
            t_logit_y = teacher_logit_y_list[i].to(device=gopd_kl.device)
            loss_mask_i = rollout_data["loss_masks"][i].to(device=gopd_kl.device)
            trunc_window = getattr(args, "opd_dualsample_truncate_window_size", 32)
            trunc_threshold = getattr(args, "opd_dualsample_truncate_threshold", None)
            trunc_mask = _compute_truncate_mask(t_logit_y, loss_mask_i, trunc_window, trunc_threshold)
            gopd_kl = gopd_kl * trunc_mask
            if use_sign_penalty:
                sign_coef = sign_coef * trunc_mask
            truncated_positions += ((1 - trunc_mask) * loss_mask_i).sum().item()
            total_active_positions += loss_mask_i.sum().item()

        # Apply prefix mask: zero out KL for the first N response tokens
        prefix_mask_len = getattr(args, "opd_kl_prefix_mask_len", None)
        if prefix_mask_len is not None and prefix_mask_len > 0:
            seq_len = gopd_kl.shape[0]
            if prefix_mask_len < seq_len:
                prefix_mask = torch.ones(seq_len, dtype=gopd_kl.dtype, device=gopd_kl.device)
                prefix_mask[:prefix_mask_len] = 0.0
                gopd_kl = gopd_kl * prefix_mask
                if use_sign_penalty:
                    sign_coef = sign_coef * prefix_mask

        # Apply per-position exponential decay: weight[t] = decay^t
        kl_decay = getattr(args, "opd_kl_decay", 1.0)
        if kl_decay != 1.0:
            seq_len = gopd_kl.shape[0]
            decay_weights = kl_decay ** torch.arange(seq_len, dtype=gopd_kl.dtype, device=gopd_kl.device)
            gopd_kl = gopd_kl * decay_weights
            if use_sign_penalty:
                sign_coef = sign_coef * decay_weights

        # Apply future-weighted KL: replace kl[t] with gamma-discounted weighted average of kl[t..T].
        # kl_future[t] = sum_{s=t}^{T} gamma^(s-t) * kl[s] / sum_{s=t}^{T} gamma^(s-t)
        # Computed efficiently via backward scan in O(T).
        kl_future_gamma = getattr(args, "opd_kl_future_gamma", 1.0)
        if kl_future_gamma != 1.0:
            seq_len = gopd_kl.shape[0]
            kl_future = torch.zeros_like(gopd_kl)
            running_sum = gopd_kl.new_zeros(())
            running_weight = gopd_kl.new_zeros(())
            for t in range(seq_len - 1, -1, -1):
                running_sum = gopd_kl[t] + kl_future_gamma * running_sum
                running_weight = 1.0 + kl_future_gamma * running_weight
                kl_future[t] = running_sum / running_weight
            gopd_kl = kl_future

        if use_sign_penalty:
            advantages[i] = adv - sign_coef
        else:
            advantages[i] = adv - sign_coef * gopd_kl

    rollout_data["opd_reverse_kl"] = reverse_kls
    rollout_data["opd_gopd_kl"] = gopd_kls

    # Prefix-position reverse-KL statistics.
    # Use the original (pre-prefix-mask) loss_masks so that per-position stats
    # still reflect the true KL distribution even when --opd-kl-prefix-mask-len is set.
    _stat_masks = original_loss_masks if original_loss_masks is not None else rollout_data.get("loss_masks")
    rollout_data.update(compute_prefix_reverse_kl_stats(reverse_kls, _stat_masks))

    # Write truncation ratio for wandb reporting
    truncate_by_teacher = getattr(args, "opd_dualsample_truncate_by_teacher_logit_y", False)
    teacher_logit_y_list = rollout_data.get("teacher_logit_y", None)
    if truncate_by_teacher and teacher_logit_y_list is not None:
        rollout_data["opd_truncation_ratio"] = torch.tensor(
            truncated_positions / max(1, total_active_positions), dtype=torch.float32
        )

    if reverse_kls:
        all_reverse_kl = torch.cat([rk if isinstance(rk, torch.Tensor) else torch.tensor(rk) for rk in reverse_kls])
        positive_count = (all_reverse_kl > 0).sum().item()
        negative_count = (all_reverse_kl < 0).sum().item()
        zero_count = (all_reverse_kl == 0).sum().item()
        total_count = all_reverse_kl.numel()

        rollout_data["opd_sign_stats"] = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "zero_count": zero_count,
            "positive_ratio": positive_count / max(1, total_count),
            "negative_ratio": negative_count / max(1, total_count),
        }


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if not mpu.is_pipeline_last_stage():
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    if args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    if args.use_opd:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )
        if getattr(args, "opd_confidence_reward_coef", 0.0) != 0.0:
            apply_opd_confidence_reward_to_advantages(args, rollout_data, advantages)

    # Union-topk confidence via advantage path (sampled-token-only gradient)
    if getattr(args, "opd_union_topk_confidence_via_advantage", False):
        apply_opd_union_topk_confidence_to_advantages(args, rollout_data, advantages)

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def _sign_with_eps(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Return +1 / -1 / 0 depending on whether x > eps / < -eps / in [-eps, eps].

    Using an epsilon threshold avoids sign flips caused by floating-point noise
    when the two log-prob differences are near zero (e.g. when x == y, or when
    the student and teacher assign nearly identical probabilities to x and y).
    """
    pos = (x > eps).float()
    neg = (x < -eps).float()
    return pos - neg  # 0 for |x| <= eps


def compute_union_topk_kl(
    args: Namespace,
    logits: torch.Tensor,
    batch: RolloutBatch,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Compute KL loss on student top-k ∪ teacher top-k union set, with loss_mask filtering.

    For each response position:
    1. Compute student's top-k tokens from current logits (no_grad for token selection).
    2. Concat with teacher's top-k tokens → [resp_len, 2k], deduplicate via mask.
    3. Gather student log_probs on the union set (WITH gradient).
    4. For teacher side: use real log_prob if token is in teacher top-k, else fallback to
       teacher's k-th (last) log_prob.
    5. Compute backward KL per position: sum_i p_s(i) * (log_p_s(i) - log_p_t(i)).
    6. Optionally filter loss_mask: mask out tokens with KL inside [low, high].

    Returns:
        (loss, filtered_loss_masks, metrics).
    """
    from slime.utils.ppo_utils import vocab_parallel_gather_log_softmax, vocab_parallel_topk

    teacher_dist_topk_tokens = batch.get("teacher_dist_topk_tokens")
    teacher_dist_topk_log_probs = batch.get("teacher_dist_topk_log_probs")

    if teacher_dist_topk_tokens is None or teacher_dist_topk_log_probs is None:
        raise ValueError(
            "compute_union_topk_kl requires 'teacher_dist_topk_tokens' and " "'teacher_dist_topk_log_probs' in batch."
        )

    kl_coef = getattr(args, "opd_teacher_topk_kl_coef", 1.0)
    tp_group = mpu.get_tensor_model_parallel_group()

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    loss_masks = batch["loss_masks"]
    max_seq_lens = batch.get("max_seq_lens", None)

    do_filter = getattr(args, "opd_union_topk_kl_filter", False)
    filter_mode = getattr(args, "opd_union_topk_kl_filter_mode", "threshold")
    threshold_low = getattr(args, "opd_union_topk_kl_filter_threshold_low", None)
    threshold_high = getattr(args, "opd_union_topk_kl_filter_threshold_high", None)

    # Confidence reward settings
    # When confidence-via-advantage is enabled, skip confidence in direct loss path
    do_confidence = getattr(args, "opd_union_topk_confidence", False) and not getattr(
        args, "opd_union_topk_confidence_via_advantage", False
    )
    conf_coef = getattr(args, "opd_union_topk_confidence_coef", 0.1)
    conf_normalize_std = getattr(args, "opd_union_topk_confidence_normalize_std", False)
    conf_data = batch.get("teacher_next_token_confidence") if do_confidence else None
    cand_data = batch.get("teacher_next_token_candidates") if do_confidence else None
    confidence_applied_count = 0
    confidence_total_count = 0
    all_conf_values = []
    kl_pure_per_sample = []

    # Alternating mode: switch between pure KL and pure confidence every N steps
    alternate_steps = getattr(args, "opd_union_topk_confidence_alternate_steps", 0)
    if alternate_steps > 0 and do_confidence:
        global_step = getattr(args, "current_global_step", 0)
        use_confidence_only = (global_step % (2 * alternate_steps)) >= alternate_steps
        use_kl_only = not use_confidence_only
    else:
        use_confidence_only = False
        use_kl_only = False

    # Collect per-sample logits chunks
    sample_chunks = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        sample_chunks.append((logits_chunk, tokens_chunk))

    kl_per_sample_for_loss = []
    kl_per_sample_detached = []
    filtered_loss_masks = []

    for i, ((logits_chunk, tokens_chunk), t_tokens_raw, t_log_probs_raw, loss_mask) in enumerate(
        zip(
            sample_chunks,
            teacher_dist_topk_tokens,
            teacher_dist_topk_log_probs,
            loss_masks,
            strict=False,
        )
    ):
        resp_len = logits_chunk.shape[0]
        device = logits_chunk.device

        if resp_len == 0:
            filtered_loss_masks.append(loss_mask)
            continue

        if t_tokens_raw is None or t_log_probs_raw is None:
            kl_per_sample_for_loss.append(torch.zeros(resp_len, device=device, dtype=torch.float32))
            kl_per_sample_detached.append(torch.zeros(resp_len, device=device, dtype=torch.float32))
            filtered_loss_masks.append(loss_mask)
            continue

        # Teacher top-k: [resp_len, k]
        t_tokens = torch.tensor(t_tokens_raw, dtype=torch.long, device=device)
        t_log_probs = torch.tensor(t_log_probs_raw, dtype=torch.float32, device=device)

        # Align lengths
        resp_len_actual = min(resp_len, t_tokens.shape[0])
        logits_chunk = logits_chunk[:resp_len_actual]
        t_tokens = t_tokens[:resp_len_actual]
        t_log_probs = t_log_probs[:resp_len_actual]
        loss_mask_local = loss_mask[:resp_len_actual].to(device=device, dtype=torch.float32)

        k = t_tokens.shape[1]

        # Step 1: Student top-k (no grad for token selection)
        with torch.no_grad():
            s_topk_log_probs, s_topk_tokens = vocab_parallel_topk(logits_chunk, k, tp_group)

        # Step 2: Concat → [resp_len_actual, 2k]
        union_tokens = torch.cat([s_topk_tokens, t_tokens], dim=-1)

        # Step 3: Dedup mask — mark teacher tokens that already appear in student top-k
        overlap = (t_tokens.unsqueeze(-1) == s_topk_tokens.unsqueeze(-2)).any(dim=-1)  # [resp_len_actual, k]
        dedup_mask = torch.cat([torch.ones(resp_len_actual, k, device=device), (~overlap).float()], dim=-1)

        # Step 4: Student log_probs on union set (WITH gradient)
        s_log_probs_union = vocab_parallel_gather_log_softmax(logits_chunk, union_tokens, tp_group)

        # Step 5: Teacher log_probs on union set
        t_fallback = t_log_probs[:, -1:]  # teacher's k-th (lowest) log_prob

        s_in_teacher = (s_topk_tokens.unsqueeze(-1) == t_tokens.unsqueeze(-2)).any(dim=-1)
        match_matrix = s_topk_tokens.unsqueeze(-1) == t_tokens.unsqueeze(-2)
        match_idx = match_matrix.float().argmax(dim=-1)
        t_log_probs_for_s_tokens = t_log_probs.gather(dim=-1, index=match_idx)
        t_log_probs_for_s_tokens = torch.where(
            s_in_teacher, t_log_probs_for_s_tokens, t_fallback.expand_as(t_log_probs_for_s_tokens)
        )

        t_log_probs_union = torch.cat([t_log_probs_for_s_tokens, t_log_probs], dim=-1)

        # Step 6: Renormalize over union set (only valid/dedup positions), then compute KL
        NEG_INF = -1e9
        invalid_mask = (1 - dedup_mask) * NEG_INF  # dedup_mask=0 → -inf
        s_log_norm = s_log_probs_union + invalid_mask
        t_log_norm = t_log_probs_union + invalid_mask

        s_log_norm = s_log_norm - torch.logsumexp(s_log_norm, dim=-1, keepdim=True)
        t_log_norm = t_log_norm - torch.logsumexp(t_log_norm, dim=-1, keepdim=True)

        # Backward KL: KL(student || teacher) = Σ p_s * (log p_s - log p_t - conf_coef * confidence)
        s_probs_norm = s_log_norm.exp() * dedup_mask

        # --- Confidence reward ---
        confidence_bonus = torch.zeros(resp_len_actual, 2 * k, device=device, dtype=torch.float32)
        if do_confidence and conf_data is not None and cand_data is not None:
            conf_sample = conf_data[i] if i < len(conf_data) else None
            cand_sample = cand_data[i] if i < len(cand_data) else None

            if conf_sample is not None and cand_sample is not None:
                for pos in range(resp_len_actual):
                    confidence_total_count += 1
                    # High-entropy positions were determined at rollout time;
                    # conf_sample[pos] is None for non-high-entropy positions
                    if pos >= len(conf_sample) or conf_sample[pos] is None:
                        continue

                    conf_vals = torch.tensor(conf_sample[pos], device=device, dtype=torch.float32)
                    cand_tokens = torch.tensor(cand_sample[pos], device=device, dtype=torch.long)

                    confidence_applied_count += 1
                    all_conf_values.append(conf_vals.clone())  # record pre-normalization values for metrics

                    # Normalize: (x - mean), optionally / std
                    conf_vals = conf_vals - conf_vals.mean()
                    if conf_normalize_std:
                        std = conf_vals.std()
                        if std > 1e-8:
                            conf_vals = conf_vals / std

                    # Map to union set's student top-k part (first k columns) by token ID matching
                    for ci in range(len(cand_tokens)):
                        match = s_topk_tokens[pos] == cand_tokens[ci]
                        if match.any():
                            idx = match.nonzero(as_tuple=True)[0][0].item()
                            confidence_bonus[pos, idx] = conf_vals[ci]

        if use_kl_only:
            # Pure reverse-KL phase: no confidence
            kl_per_pos = (s_probs_norm * (s_log_norm - t_log_norm) * dedup_mask).sum(dim=-1)
        elif use_confidence_only:
            # Pure confidence phase: no KL
            kl_per_pos = (s_probs_norm * (-conf_coef * confidence_bonus) * dedup_mask).sum(dim=-1)
        else:
            # Combined (default, or when alternate_steps=0)
            kl_per_pos = (s_probs_norm * (s_log_norm - t_log_norm - conf_coef * confidence_bonus) * dedup_mask).sum(
                dim=-1
            )

        # Track pure KL (without confidence) for metrics
        if do_confidence and confidence_bonus.any() and not use_kl_only:
            kl_pure_per_pos = (s_probs_norm * (s_log_norm - t_log_norm) * dedup_mask).sum(dim=-1)
            kl_pure_per_sample.append((kl_pure_per_pos * loss_mask_local).detach())

        kl_masked = kl_per_pos * loss_mask_local
        kl_per_sample_for_loss.append(kl_masked)
        kl_per_sample_detached.append(kl_masked.detach())
        filtered_loss_masks.append(loss_mask)

    # Compute loss
    if kl_per_sample_for_loss:
        total_kl_loss = sum_of_sample_mean(torch.cat(kl_per_sample_for_loss, dim=0))
    else:
        total_kl_loss = logits.sum() * 0.0

    loss = kl_coef * total_kl_loss

    # Apply loss_mask filtering based on per-token KL
    if do_filter and kl_per_sample_detached:
        if filter_mode == "quantile":
            quantile_low = getattr(args, "opd_union_topk_kl_filter_quantile_low", None)
            quantile_high = getattr(args, "opd_union_topk_kl_filter_quantile_high", None)
            all_kl_valid = []
            for idx, lm in enumerate(loss_masks):
                if idx < len(kl_per_sample_detached):
                    kl_vals = kl_per_sample_detached[idx]
                    lm_local = lm[: kl_vals.shape[0]].to(device=kl_vals.device, dtype=torch.bool)
                    if lm_local.any():
                        all_kl_valid.append(kl_vals[lm_local])
            if all_kl_valid:
                all_kl_cat = torch.cat(all_kl_valid)
                if quantile_low is not None:
                    threshold_low = torch.quantile(all_kl_cat, quantile_low).item()
                if quantile_high is not None:
                    threshold_high = torch.quantile(all_kl_cat, quantile_high).item()

        # Keep tokens OUTSIDE [low, high]
        filtered_loss_masks = []
        total_original = 0
        total_kept = 0
        for idx, lm in enumerate(loss_masks):
            if idx < len(kl_per_sample_detached):
                kl_vals = kl_per_sample_detached[idx]
                resp_len = kl_vals.shape[0]
                lm_local = lm[:resp_len].clone().to(device=kl_vals.device)

                original_count = (lm_local == 1).sum().item()

                if threshold_low is not None and threshold_high is not None:
                    keep = (kl_vals < threshold_low) | (kl_vals > threshold_high)
                elif threshold_low is not None:
                    keep = kl_vals < threshold_low
                elif threshold_high is not None:
                    keep = kl_vals > threshold_high
                else:
                    keep = torch.ones(resp_len, device=kl_vals.device, dtype=torch.bool)

                new_mask = lm_local.float() * keep.float()
                if lm.shape[0] > resp_len:
                    new_mask = torch.cat([new_mask, lm[resp_len:].to(device=kl_vals.device).float()])
                filtered_loss_masks.append(new_mask.to(device=lm.device, dtype=lm.dtype))

                total_original += original_count
                total_kept += (new_mask[:resp_len] == 1).sum().item()
            else:
                filtered_loss_masks.append(lm)
    else:
        total_original = 0
        total_kept = 0

    # Metrics
    metrics: dict[str, torch.Tensor] = {
        "union_topk_kl_loss": loss.detach(),
    }
    if kl_per_sample_detached:
        all_kl = torch.cat(kl_per_sample_detached)
        nonzero_kl = all_kl[all_kl != 0]
        metrics["union_topk_kl_mean"] = nonzero_kl.mean() if nonzero_kl.numel() > 0 else all_kl.mean()
    if do_filter and total_original > 0:
        metrics["union_topk_kl_filter_ratio"] = torch.tensor(1.0 - total_kept / total_original, device=logits.device)
        if threshold_low is not None:
            metrics["union_topk_kl_filter_threshold_low"] = torch.tensor(threshold_low, device=logits.device)
        if threshold_high is not None:
            metrics["union_topk_kl_filter_threshold_high"] = torch.tensor(threshold_high, device=logits.device)

    if do_confidence and confidence_total_count > 0:
        # Report raw counts so that the global ratio can be computed correctly
        # across micro-batches (sum of counts / sum of totals, not average of ratios).
        metrics["union_topk_confidence_applied_count__raw"] = torch.tensor(
            float(confidence_applied_count), device=logits.device
        )
        metrics["union_topk_confidence_total_count__raw"] = torch.tensor(
            float(confidence_total_count), device=logits.device
        )
        if all_conf_values:
            all_conf_cat = torch.cat(all_conf_values)
            metrics["union_topk_confidence_mean"] = all_conf_cat.mean()
            metrics["union_topk_confidence_std"] = all_conf_cat.std()
        if kl_pure_per_sample:
            pure_kl_loss = kl_coef * sum_of_sample_mean(torch.cat(kl_pure_per_sample))
            metrics["union_topk_kl_pure_loss"] = pure_kl_loss
            metrics["union_topk_confidence_loss_delta"] = pure_kl_loss - loss.detach()
        if alternate_steps > 0:
            # 1 = pure KL phase, 2 = pure confidence phase
            mode = 1 if use_kl_only else 2
            metrics["union_topk_confidence_alternate_mode"] = torch.tensor(
                mode, device=logits.device, dtype=torch.float32
            )

    return loss, filtered_loss_masks, metrics


def compute_opd_teacher_topk_kl_loss(
    args: Namespace,
    logits: torch.Tensor,
    batch: RolloutBatch,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute KL distillation loss using teacher's top-k token distribution.

    Unlike the advantage-based OPD (which uses rollout-time student log probs without gradients),
    this loss is computed from training-time logits so the student receives actual gradients.

    The token set is defined by the teacher's top-k tokens. If the student's sampled token is not
    in the teacher's top-k, it replaces the last entry (lowest teacher probability token).

    Supports three KL variants controlled by --opd-teacher-topk-kl-type:
      - 'backward': KL(teacher || student) = Σ p_t * (log p_t - log p_s)
      - 'forward':  KL(student || teacher) = Σ p_s * (log p_s - log p_t)
      - 'jsd':      Generalized JSD = (1-beta)*KL(p_t||m) + beta*KL(p_s||m), m = beta*p_s + (1-beta)*p_t,
                    beta controlled by --opd-teacher-topk-jsd-beta (default 0.5 = standard JSD)

    Vocab-parallel awareness: token log probs are gathered via vocab_parallel_gather_log_softmax
    which correctly handles TP-sharded logits using all-reduce across the TP group.

    Args:
        args: Configuration with opd_teacher_topk_kl_type, opd_teacher_topk_kl_coef.
        logits: Full (possibly TP-sharded) logits, shape [1, T, local_vocab_size].
        batch: Must contain 'teacher_dist_topk_tokens' and 'teacher_dist_topk_log_probs',
               plus 'unconcat_tokens', 'total_lengths', 'response_lengths', 'loss_masks'.
        sum_of_sample_mean: Reduction function.

    Returns:
        (loss, metrics) where loss is a scalar and metrics contains detached scalars.
    """
    from slime.utils.ppo_utils import vocab_parallel_gather_log_softmax

    teacher_dist_topk_tokens = batch.get("teacher_dist_topk_tokens")
    teacher_dist_topk_log_probs = batch.get("teacher_dist_topk_log_probs")

    if teacher_dist_topk_tokens is None or teacher_dist_topk_log_probs is None:
        raise ValueError(
            "compute_opd_teacher_topk_kl_loss requires 'teacher_dist_topk_tokens' and "
            "'teacher_dist_topk_log_probs' in batch. Make sure --opd-teacher-topk-kl is enabled "
            "and the reward_func stores teacher top-k data."
        )

    kl_type = getattr(args, "opd_teacher_topk_kl_type", "backward")
    kl_coef = getattr(args, "opd_teacher_topk_kl_coef", 1.0)
    tp_group = mpu.get_tensor_model_parallel_group()

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    loss_masks = batch["loss_masks"]
    max_seq_lens = batch.get("max_seq_lens", None)

    kl_per_sample = []

    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        # logits_chunk: [resp_len, local_vocab_size]
        # tokens_chunk: [resp_len]  (student sampled tokens, global ids)
        kl_per_sample.append((logits_chunk, tokens_chunk))

    kl_values = []
    # Collect per-sample kl tensors (zeros for samples with missing teacher data).
    # sum_of_sample_mean expects the full micro-batch concatenated, so we must call it
    # once on torch.cat(kl_per_sample_for_loss) instead of once per sample.
    kl_per_sample_for_loss = []

    for i, ((logits_chunk, tokens_chunk), t_tokens_raw, t_log_probs_raw, loss_mask) in enumerate(
        zip(
            kl_per_sample,
            teacher_dist_topk_tokens,
            teacher_dist_topk_log_probs,
            loss_masks,
            strict=False,
        )
    ):
        resp_len = logits_chunk.shape[0]
        device = logits_chunk.device

        if resp_len == 0:
            continue

        # If teacher topk data is missing for this sample, contribute zero KL.
        if t_tokens_raw is None or t_log_probs_raw is None:
            kl_per_sample_for_loss.append(torch.zeros(resp_len, device=device, dtype=torch.float32))
            continue

        # Build teacher token id tensor: [resp_len, k]
        t_tokens = torch.tensor(t_tokens_raw, dtype=torch.long, device=device)  # [resp_len, k]
        t_log_probs = torch.tensor(t_log_probs_raw, dtype=torch.float32, device=device)  # [resp_len, k]

        # Align lengths (teacher data may span full resp while logits_chunk may be shorter due to CP)
        resp_len_actual = min(resp_len, t_tokens.shape[0])
        logits_chunk = logits_chunk[:resp_len_actual]
        tokens_chunk = tokens_chunk[:resp_len_actual]
        t_tokens = t_tokens[:resp_len_actual]  # [resp_len_actual, k]
        t_log_probs = t_log_probs[:resp_len_actual]  # [resp_len_actual, k]
        loss_mask_local = loss_mask[:resp_len_actual].to(device=device, dtype=torch.float32)

        k = t_tokens.shape[1]

        # Replace last teacher topk entry with student sampled token if not already present.
        # tokens_chunk: [resp_len_actual] contains global token ids sampled by student.
        sampled = tokens_chunk.unsqueeze(-1)  # [resp_len_actual, 1]
        in_topk = (t_tokens == sampled).any(dim=-1)  # [resp_len_actual]
        # For positions where sampled token is absent, overwrite t_tokens[:, -1]
        replace_mask = ~in_topk  # [resp_len_actual]
        t_tokens_new = t_tokens.clone()
        t_tokens_new[replace_mask, -1] = tokens_chunk[replace_mask]
        # For replaced positions we don't know teacher log prob for the sampled token yet;
        # we'll gather it from teacher_dist_topk_log_probs or set to a placeholder.
        # Since teacher didn't return this token's log prob, use the minimum teacher log prob
        # at that position as a conservative estimate (teacher assigns it low probability).
        t_log_probs_new = t_log_probs.clone()
        if replace_mask.any():
            min_t_log_prob = t_log_probs.min(dim=-1).values  # [resp_len_actual]
            t_log_probs_new[replace_mask, -1] = min_t_log_prob[replace_mask]

        # Gather student log probs for the (possibly updated) teacher top-k token ids.
        # vocab_parallel_gather_log_softmax handles TP-sharded logits correctly.
        # Result: [resp_len_actual, k], with gradients flowing through logits_chunk.
        student_log_probs = vocab_parallel_gather_log_softmax(
            logits_chunk, t_tokens_new, tp_group
        )  # [resp_len_actual, k]

        # Normalize teacher log probs over the k-token subset to form a proper distribution.
        # Note: teacher_dist_topk_log_probs are global softmax log probs (not conditional on topk),
        # so we need to renormalize over the k tokens for KL computation.
        t_log_probs_subset = t_log_probs_new - torch.logsumexp(t_log_probs_new, dim=-1, keepdim=True)
        # Similarly normalize student over the k-token subset
        s_log_probs_subset = student_log_probs - torch.logsumexp(student_log_probs, dim=-1, keepdim=True)

        t_probs = t_log_probs_subset.exp()  # [resp_len_actual, k]
        s_probs = s_log_probs_subset.exp()  # [resp_len_actual, k]

        if kl_type == "forward":
            # Forward KL: KL(teacher || student) = Σ p_t * (log p_t - log p_s)
            kl_per_token = (t_probs * (t_log_probs_subset - s_log_probs_subset)).sum(dim=-1)
        elif kl_type == "backward":
            # Backward KL: KL(student || teacher) = Σ p_s * (log p_s - log p_t)
            kl_per_token = (s_probs * (s_log_probs_subset - t_log_probs_subset)).sum(dim=-1)
        elif kl_type == "jsd":
            # Generalized JSD with beta:
            #   m = beta * p_s + (1-beta) * p_t
            #   JSD = (1-beta) * KL(p_t || m) + beta * KL(p_s || m)
            # beta=0.5 gives standard symmetric JSD.
            jsd_beta = getattr(args, "opd_teacher_topk_jsd_beta", 0.5)
            m_probs =  (1.0 - jsd_beta) * s_probs + jsd_beta * t_probs
            m_log_probs = m_probs.log().clamp(min=-1e9)
            kl_t_m = (t_probs * (t_log_probs_subset - m_log_probs)).sum(dim=-1)
            kl_s_m = (s_probs * (s_log_probs_subset - m_log_probs)).sum(dim=-1)
            kl_per_token = (1.0 - jsd_beta) * kl_t_m + jsd_beta * kl_s_m
        else:
            raise ValueError(f"Unknown opd_teacher_topk_kl_type: '{kl_type}'. Choose from forward/backward/jsd.")

        # Apply loss mask
        kl_masked = kl_per_token * loss_mask_local  # [resp_len_actual]
        kl_values.append(kl_masked.detach())
        kl_per_sample_for_loss.append(kl_masked)

    # sum_of_sample_mean expects a single concatenated tensor covering all samples.
    if kl_per_sample_for_loss:
        total_kl_loss = sum_of_sample_mean(torch.cat(kl_per_sample_for_loss, dim=0))
    else:
        total_kl_loss = logits.sum() * 0.0

    loss = kl_coef * total_kl_loss

    metrics = {
        "opd_teacher_topk_kl_loss": loss.detach(),
        "opd_teacher_topk_kl_type": torch.tensor(0.0, device=logits.device),  # placeholder for logging
    }
    if kl_values:
        all_kl = torch.cat(kl_values)
        metrics["opd_teacher_topk_kl_mean"] = all_kl[all_kl != 0].mean() if (all_kl != 0).any() else all_kl.mean()

    return loss, metrics


def compute_rc_opd_loss(
    args: Namespace,
    logits: torch.Tensor,
    batch: RolloutBatch,
    log_probs_list: list[torch.Tensor],
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Rank-Consistency OPD (RC-OPD) loss.

    For each response token position t:
      x = student's sampled token (actual response token)
      y = teacher's sampled token (from teacher distribution)

    Consistency is defined as: the ordering of x and y is the same in both
    the student distribution and the teacher distribution:
      sign_eps(log_ps(x) - log_ps(y)) * sign_eps(log_pt(x) - log_pt(y)) >= 0

    sign_eps avoids floating-point noise near zero (controlled by --opd-rc-sign-eps).
    Ties (|diff| <= eps) are treated as zero and are always counted as consistent.

    Supported loss types per class (consistent / inconsistent):
      "opd_kl"  — reverse KL contribution: log_ps(x) - log_pt(x)
      "mse"     — (ps(x) - pt(x))^2 + (ps(y) - pt(y))^2
      "none"    — no additional loss for this class

    Args:
        args: Configuration namespace. Relevant attributes:
            opd_rc_consistent_loss, opd_rc_inconsistent_loss,
            opd_rc_consistent_coef, opd_rc_inconsistent_coef, opd_rc_sign_eps.
        logits: Full model logits [1, T, V] from the current training forward pass.
        batch: Mini-batch dict. Must contain "teacher_sampled_tokens" (CP-sliced,
            list of [R_local] long tensors), "teacher_log_probs" (CP-sliced,
            list of [R_local] float tensors = log_pt(x)), "teacher_logit_y"
            (list of full-response-length float tensors = log_pt(y)).
        log_probs_list: Per-sample CP-local log_ps(x) tensors (list of [R_local]).
        sum_of_sample_mean: Reduction function that averages across tokens/samples.

    Returns:
        Tuple of (rc_loss scalar tensor, metrics dict).
    """
    teacher_sampled_tokens = batch.get("teacher_sampled_tokens")
    teacher_log_probs = batch.get("teacher_log_probs")  # log_pt(x), CP-local
    teacher_logit_y = batch.get("teacher_logit_y")  # log_pt(y), full response length

    if teacher_sampled_tokens is None or teacher_log_probs is None or teacher_logit_y is None:
        return torch.tensor(0.0, device=logits.device), {}

    total_lengths = batch["total_lengths"]
    response_lengths = batch["response_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    loss_masks = batch["loss_masks"]

    consistent_loss_type = getattr(args, "opd_rc_consistent_loss", "none")
    inconsistent_loss_type = getattr(args, "opd_rc_inconsistent_loss", "opd_kl")
    consistent_coef = getattr(args, "opd_rc_consistent_coef", 1.0)
    inconsistent_coef = getattr(args, "opd_rc_inconsistent_coef", 1.0)
    sign_eps = getattr(args, "opd_rc_sign_eps", 1e-6)
    mse_y_gamma = getattr(args, "opd_rc_mse_y_gamma", 1.0)

    # Compute log_ps(y): gather from training logits at teacher_sampled_tokens positions.
    # teacher_sampled_tokens are already CP-sliced in _get_rollout_data, so they match
    # the per-CP-rank logit chunks yielded by get_responses.
    # Also collect CP-local x tokens (tokens_chunk) for x == y comparison in metrics.
    log_ps_y_list: list[torch.Tensor] = []
    x_tokens_list: list[torch.Tensor] = []
    for (logits_chunk, tokens_chunk), y_tokens in zip(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        ),
        teacher_sampled_tokens,
    ):
        # logits_chunk: [R_local, V] (temperature-scaled by get_responses)
        # tokens_chunk: [R_local] long — CP-local response tokens (x)
        # y_tokens:     [R_local] long — CP-local teacher-sampled tokens (y)
        log_ps_y, _ = calculate_log_probs_and_entropy(
            logits_chunk,
            y_tokens,
            mpu.get_tensor_model_parallel_group(),
            with_entropy=False,
            chunk_size=args.log_probs_chunk_size,
        )
        log_ps_y_list.append(log_ps_y.squeeze(-1))
        x_tokens_list.append(tokens_chunk)

    cp_size = mpu.get_context_parallel_world_size()

    rc_token_losses: list[torch.Tensor] = []
    consistent_count = 0.0
    total_count = 0.0
    consistent_loss_acc: list[float] = []
    inconsistent_loss_acc: list[float] = []
    consistent_n_acc: list[float] = []
    inconsistent_n_acc: list[float] = []
    x_eq_y_count = 0.0
    neutral_count = 0.0
    student_order_abs_acc = 0.0
    teacher_order_abs_acc = 0.0
    mse_x_acc = 0.0  # sum of (ps(x)-pt(x))^2 over valid tokens
    mse_y_acc = 0.0  # sum of (ps(y)-pt(y))^2 over valid tokens

    for i, (log_ps_x, log_ps_y, log_pt_x, t_logit_y_raw, loss_mask) in enumerate(
        zip(log_probs_list, log_ps_y_list, teacher_log_probs, teacher_logit_y, loss_masks)
    ):
        device = log_ps_x.device
        log_pt_x = log_pt_x.to(device)
        loss_mask_float = loss_mask.float().to(device)

        # CP-slice teacher_logit_y to match the CP-local length of log_ps_x.
        # When cp_size == 1, this is a no-op (slice_log_prob_with_cp returns input).
        if isinstance(t_logit_y_raw, torch.Tensor):
            t_logit_y_list = t_logit_y_raw.tolist()
        else:
            t_logit_y_list = list(t_logit_y_raw)
        max_seq_len_i = max_seq_lens[i] if max_seq_lens is not None else None
        log_pt_y = torch.tensor(
            slice_log_prob_with_cp(
                t_logit_y_list,
                total_lengths[i],
                response_lengths[i],
                args.qkv_format,
                max_seq_len_i,
            ),
            dtype=torch.float32,
            device=device,
        )

        # Rank-consistency mask using epsilon-guarded sign comparison.
        # Avoids floating-point noise when differences are near zero.
        student_diff = log_ps_x - log_ps_y
        teacher_diff = log_pt_x - log_pt_y
        student_sign = _sign_with_eps(student_diff, sign_eps)
        teacher_sign = _sign_with_eps(teacher_diff, sign_eps)
        # product >= 0: same sign (+/+, -/-, or either is 0) → consistent
        consistent_mask = (student_sign * teacher_sign >= 0).float()
        inconsistent_mask = 1.0 - consistent_mask
        # neutral: at least one side's diff is in the eps-zone (sign == 0)
        neutral_mask = ((student_sign == 0) | (teacher_sign == 0)).float()

        # Precompute probability values needed for MSE (shared across both classes)
        ps_x = torch.exp(log_ps_x)
        pt_x = torch.exp(log_pt_x)
        ps_y = torch.exp(log_ps_y)
        pt_y = torch.exp(log_pt_y)
        mse_x_term = (ps_x - pt_x).pow(2)
        mse_y_term = (ps_y - pt_y).pow(2)

        def _loss_for_type(loss_type: str) -> torch.Tensor:
            if loss_type == "opd_kl":
                return log_ps_x - log_pt_x
            elif loss_type == "mse":
                return mse_x_term + mse_y_gamma * mse_y_term
            else:  # "none"
                return torch.zeros_like(log_ps_x)

        loss_c = _loss_for_type(consistent_loss_type)
        loss_i = _loss_for_type(inconsistent_loss_type)

        token_loss = (
            consistent_coef * consistent_mask * loss_c + inconsistent_coef * inconsistent_mask * loss_i
        ) * loss_mask_float

        # Compatibility with opd_dualsample_truncate_by_teacher_logit_y:
        # Apply the same truncation mask as in apply_opd_kl_to_advantages so that
        # RC-OPD loss is also zeroed out at positions where the teacher is uncertain.
        truncate_by_teacher = getattr(args, "opd_dualsample_truncate_by_teacher_logit_y", False)
        if truncate_by_teacher:
            trunc_window = getattr(args, "opd_dualsample_truncate_window_size", 32)
            trunc_threshold = getattr(args, "opd_dualsample_truncate_threshold", None)
            # log_pt_y is already CP-sliced, same length as token_loss
            trunc_mask = _compute_truncate_mask(log_pt_y, loss_mask_float, trunc_window, trunc_threshold)
            token_loss = token_loss * trunc_mask

        rc_token_losses.append(token_loss)

        # Metrics accumulators
        n_valid = loss_mask_float.sum().item()
        consistent_count += (consistent_mask * loss_mask_float).sum().item()
        total_count += n_valid

        # Per-class mean loss (over valid + class-masked tokens)
        n_consistent = (consistent_mask * loss_mask_float).sum().item()
        n_inconsistent = (inconsistent_mask * loss_mask_float).sum().item()
        consistent_loss_acc.append((consistent_mask * loss_mask_float * loss_c.detach()).sum().item())
        inconsistent_loss_acc.append((inconsistent_mask * loss_mask_float * loss_i.detach()).sum().item())
        consistent_n_acc.append(n_consistent)
        inconsistent_n_acc.append(n_inconsistent)

        # x_tokens_list[i] is the CP-local response tokens (x), same length as teacher_sampled_tokens[i]
        x_eq_y_count += ((teacher_sampled_tokens[i] == x_tokens_list[i]).float() * loss_mask_float).sum().item()
        neutral_count += (neutral_mask * loss_mask_float).sum().item()
        student_order_abs_acc += (student_diff.abs() * loss_mask_float).sum().item()
        teacher_order_abs_acc += (teacher_diff.abs() * loss_mask_float).sum().item()
        # MSE per-term stats (always computed regardless of loss_type, for monitoring)
        mse_x_acc += (mse_x_term.detach() * loss_mask_float).sum().item()
        mse_y_acc += (mse_y_term.detach() * loss_mask_float).sum().item()

    if not rc_token_losses:
        return torch.tensor(0.0, device=logits.device), {}

    rc_loss = sum_of_sample_mean(torch.cat(rc_token_losses, dim=0))

    denom = max(1.0, total_count)
    n_consistent_total = sum(consistent_n_acc)
    n_inconsistent_total = sum(inconsistent_n_acc)
    metrics = {
        "rc_opd_loss": rc_loss.detach(),
        "rc_opd_consistency_ratio": torch.tensor(consistent_count / denom, device=logits.device),
        "rc_opd_consistent_loss_mean": torch.tensor(
            sum(consistent_loss_acc) / max(1.0, n_consistent_total), device=logits.device
        ),
        "rc_opd_inconsistent_loss_mean": torch.tensor(
            sum(inconsistent_loss_acc) / max(1.0, n_inconsistent_total), device=logits.device
        ),
        "rc_opd_x_eq_y_ratio": torch.tensor(x_eq_y_count / denom, device=logits.device),
        "rc_opd_neutral_ratio": torch.tensor(neutral_count / denom, device=logits.device),
        "rc_opd_student_order_abs_diff": torch.tensor(student_order_abs_acc / denom, device=logits.device),
        "rc_opd_teacher_order_abs_diff": torch.tensor(teacher_order_abs_acc / denom, device=logits.device),
        "rc_opd_mse_x_mean": torch.tensor(mse_x_acc / denom, device=logits.device),
        "rc_opd_mse_y_mean": torch.tensor(mse_y_acc / denom, device=logits.device),
    }
    return rc_loss, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # REOPOLD Phase II: entropy-guided token sampling (Eq.10 in arXiv:2603.11137)
    # Only backprop through top-β% highest-entropy tokens (critical branching points)
    if getattr(args, "reopold", False) and getattr(args, "reopold_current_phase", 1) == 2:
        entropy_list = log_probs_and_entropy["entropy"]
        entropy_flat = torch.cat(entropy_list, dim=0)
        loss_mask_flat = torch.cat(batch["loss_masks"], dim=0).bool()
        valid_entropy = entropy_flat[loss_mask_flat]
        if valid_entropy.numel() > 0:
            tau_beta = torch.quantile(valid_entropy, 1.0 - args.reopold_entropy_beta)
            entropy_mask = (entropy_flat >= tau_beta).float()
            pg_loss = pg_loss * entropy_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with modified_response_masks for denominator correction
        # modified_response_masks will be sliced with cp in get_sum_of_sample_mean
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Apply advantage-based filtering if configured
    adv_filter_clipfrac = None
    if args.grpo_sample_filter != "both":
        # Create mask based on advantage sign
        if args.grpo_sample_filter == "positive":
            adv_filter_mask = (advantages > 0).float()  # Only keep positive samples
        else:  # "negative"
            adv_filter_mask = (advantages < 0).float()  # Only keep negative samples

        # Count filtered tokens for logging
        adv_filter_clipfrac = (adv_filter_mask == 0).sum() / advantages.numel()

        # Apply mask to policy gradient loss
        pg_loss = pg_loss * adv_filter_mask

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # RC-OPD: rank-consistency based distillation loss
    rc_opd_metrics = {}
    if getattr(args, "opd_use_rc", False) and batch.get("teacher_sampled_tokens") is not None:
        rc_loss, rc_opd_metrics = compute_rc_opd_loss(
            args=args,
            logits=logits,
            batch=batch,
            log_probs_list=log_probs_and_entropy["log_probs"],
            sum_of_sample_mean=sum_of_sample_mean,
        )
        loss = loss + rc_loss

    # Teacher-TopK KL distillation loss: KL computed from training logits with full gradient support.
    # Token set is defined by teacher's top-k; student log probs are gathered via vocab-parallel gather.
    teacher_topk_kl_metrics: dict = {}
    if getattr(args, "opd_teacher_topk_kl", False) and batch.get("teacher_dist_topk_tokens") is not None:
        teacher_topk_kl_loss, teacher_topk_kl_metrics = compute_opd_teacher_topk_kl_loss(
            args=args,
            logits=logits,
            batch=batch,
            sum_of_sample_mean=sum_of_sample_mean,
        )
        loss = loss + teacher_topk_kl_loss

    # Union TopK KL: student top-k ∪ teacher top-k, with optional loss_mask filtering
    union_topk_kl_metrics: dict = {}
    if getattr(args, "opd_union_topk_kl", False) and batch.get("teacher_dist_topk_tokens") is not None:
        union_topk_kl_loss, filtered_loss_masks, union_topk_kl_metrics = compute_union_topk_kl(
            args=args,
            logits=logits,
            batch=batch,
            sum_of_sample_mean=sum_of_sample_mean,
        )
        loss = loss + union_topk_kl_loss
        if filtered_loss_masks:
            batch["loss_masks"] = filtered_loss_masks

    # Dump student top-k tokens and log probs for offline analysis (requires --dump-student-topk-size > 0).
    # Uses vocab-parallel top-k to correctly handle TP-sharded logits.
    dump_k = getattr(args, "dump_student_topk_size", 0)
    if dump_k > 0:
        from slime.utils.ppo_utils import vocab_parallel_topk

        tp_group = mpu.get_tensor_model_parallel_group()
        student_topk_tokens_list = []
        student_topk_log_probs_list = []
        for logits_chunk, _ in get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens", None),
        ):
            with torch.no_grad():
                topk_lp, topk_ids = vocab_parallel_topk(logits_chunk, dump_k, tp_group)
            student_topk_tokens_list.append(topk_ids.cpu())
            student_topk_log_probs_list.append(topk_lp.detach().cpu())
        batch["student_topk_tokens_post"] = student_topk_tokens_list
        batch["student_topk_log_probs_post"] = student_topk_log_probs_list

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Add advantage filter metrics if enabled
    if args.grpo_sample_filter != "both":
        reported_loss["adv_filter_clipfrac"] = adv_filter_clipfrac.clone().detach()

        # Additional statistics for monitoring
        pos_mask = advantages > 0
        neg_mask = advantages < 0

        reported_loss["adv_positive_count"] = pos_mask.sum().float().clone().detach()
        reported_loss["adv_negative_count"] = neg_mask.sum().float().clone().detach()

        # Mean advantage values
        if pos_mask.any():
            reported_loss["adv_mean_positive"] = advantages[pos_mask].mean().clone().detach()
        else:
            reported_loss["adv_mean_positive"] = torch.tensor(0.0, device=advantages.device)

        if neg_mask.any():
            reported_loss["adv_mean_negative"] = advantages[neg_mask].mean().clone().detach()
        else:
            reported_loss["adv_mean_negative"] = torch.tensor(0.0, device=advantages.device)
    # RC-OPD metrics
    for k, v in rc_opd_metrics.items():
        reported_loss[k] = v.clone().detach() if isinstance(v, torch.Tensor) else v

    # Teacher-TopK KL metrics
    for k, v in teacher_topk_kl_metrics.items():
        if k == "opd_teacher_topk_kl_type":
            continue  # placeholder, skip
        reported_loss[k] = v.clone().detach() if isinstance(v, torch.Tensor) else v

    # Union TopK KL metrics
    for k, v in union_topk_kl_metrics.items():
        reported_loss[k] = v.clone().detach() if isinstance(v, torch.Tensor) else v

    # Add OPD confidence reward metrics if available
    if "opd_teacher_confidence" in batch:
        conf = torch.cat(batch["opd_teacher_confidence"], dim=0)
        reported_loss["opd_teacher_confidence_mean"] = sum_of_sample_mean(conf).clone().detach()

    # Add union-topk confidence via advantage metrics if available
    if "opd_union_topk_conf_adv_metrics" in batch:
        conf_adv_metrics = batch["opd_union_topk_conf_adv_metrics"]
        applied_count = conf_adv_metrics["applied_count"]
        total_count = conf_adv_metrics["total_count"]

        # Use __raw suffix for correct cross-microbatch aggregation (sum of counts / sum of totals)
        reported_loss["union_topk_conf_adv_applied_count__raw"] = torch.tensor(
            float(applied_count), device=logits.device
        )
        reported_loss["union_topk_conf_adv_total_count__raw"] = torch.tensor(
            float(total_count), device=logits.device
        )

        raw_values = conf_adv_metrics["raw_values"]
        if raw_values:
            raw_t = torch.tensor(raw_values, dtype=torch.float32, device=logits.device)
            reported_loss["union_topk_conf_adv_mean"] = raw_t.mean()
            reported_loss["union_topk_conf_adv_std"] = raw_t.std()

    if "opd_union_topk_conf_adv_bonus" in batch:
        bonus = torch.cat(batch["opd_union_topk_conf_adv_bonus"], dim=0)
        reported_loss["union_topk_conf_adv_bonus_mean"] = sum_of_sample_mean(bonus).clone().detach()

    # Add OPD metrics if available
    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()
    for _prefix_len in _OPD_REVERSE_KL_PREFIX_LENGTHS:
        _key = f"opd_reverse_kl_prefix_{_prefix_len}"
        if _key in batch:
            reported_loss[_key] = batch[_key].clone().detach()
    if "opd_truncation_ratio" in batch:
        reported_loss["opd/truncation_ratio"] = batch["opd_truncation_ratio"].float()

    # REOPOLD phase logging
    if getattr(args, "reopold", False):
        reported_loss["reopold_phase"] = torch.tensor(
            float(getattr(args, "reopold_current_phase", 1)), device=logits.device
        )

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not args.calculate_per_token_loss:
        loss = (
            loss * num_microbatches / global_batch_size * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else num_tokens,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
