"""
RFT (Rejection Sampling Fine-Tuning) Rollout

Generates N candidates per prompt, evaluates with reward model,
and keeps only correct responses for SFT training.

Usage:
    --rollout-function-path examples.rft.rft_rollout:generate_rollout
    --rm-url http://teacher:8000/generate    # Teacher model endpoint
    --rm-type deepscaler                     # Reward model type
    --n-samples-per-prompt 4                 # Candidates per prompt
    --over-sampling-batch-size 256           # Over-sample for filtering
    --rollout-batch-size 128                 # Target batch size
    --rft-reward-threshold 0.5               # Min reward to keep (default 0.5)
    --rm-max-concurrent-requests 128         # Concurrent teacher requests
"""

import asyncio
import logging

from slime.rollout.rm_hub import batched_async_rm
from slime.utils.async_utils import run
from slime.utils.http_utils import post
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

__all__ = ["generate_rollout"]

# Module-level singletons
_TOKENIZER = None
_MASK_GENERATOR = None
_SEMAPHORE: asyncio.Semaphore | None = None


def _get_semaphore(args) -> asyncio.Semaphore:
    """Get global semaphore for teacher generation concurrency control."""
    global _SEMAPHORE
    if _SEMAPHORE is None:
        concurrency = getattr(args, "rm_max_concurrent_requests", 32)
        _SEMAPHORE = asyncio.Semaphore(concurrency)
    return _SEMAPHORE


async def _generate_one(args, sample: Sample, sampling_params: dict) -> Sample:
    """Generate one response from teacher model (with concurrency control)."""
    teacher_url: str = args.rm_url
    teacher_model_name = getattr(args, "teacher_model_name", None)

    # Apply chat template
    if isinstance(sample.prompt, str):
        prompt_text = sample.prompt
    else:
        prompt_text = _TOKENIZER.apply_chat_template(
            sample.prompt, tokenize=False, add_generation_prompt=True
        )

    payload = {
        "text": prompt_text,
        "sampling_params": sampling_params,
        "return_logprob": False,
    }
    if teacher_model_name:
        payload["model"] = teacher_model_name

    # Semaphore control for teacher generation
    semaphore = _get_semaphore(args)
    async with semaphore:
        try:
            output = await post(teacher_url, payload, max_retries=50)
        except Exception as exc:
            logger.warning(f"[RFT] Teacher generation failed: {exc}")
            sample.remove_sample = True
            return sample

    teacher_response: str = output["text"]

    # Build tokens and loss mask
    if isinstance(sample.prompt, str):
        prompt_ids = _TOKENIZER(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = _TOKENIZER(prompt_text + teacher_response, add_special_tokens=False)["input_ids"]
        response_length = len(full_ids) - len(prompt_ids)
        token_ids = full_ids
        loss_mask = [0] * len(prompt_ids) + [1] * response_length
    else:
        complete_messages = list(sample.prompt) + [{"role": "assistant", "content": teacher_response}]
        token_ids, loss_mask = _MASK_GENERATOR.get_loss_mask(complete_messages)
        response_length = _MASK_GENERATOR.get_response_lengths([loss_mask])[0]

    if response_length == 0:
        sample.remove_sample = True
        return sample

    sample.tokens = token_ids
    sample.response = teacher_response
    sample.response_length = response_length
    sample.loss_mask = loss_mask[-response_length:]

    return sample


async def _generate_and_filter_prompt(args, prompt_sample: Sample, sampling_params: dict) -> list[Sample]:
    """Generate N candidates for one prompt and filter by reward."""
    n_candidates = args.n_samples_per_prompt

    # Create candidate samples
    candidates = []
    for _ in range(n_candidates):
        candidate = Sample(
            prompt=prompt_sample.prompt,
            label=prompt_sample.label,
            metadata=prompt_sample.metadata,
            index=prompt_sample.index,
        )
        candidates.append(candidate)

    # Generate all candidates (semaphore controlled inside _generate_one)
    tasks = [_generate_one(args, candidate, sampling_params) for candidate in candidates]
    await asyncio.gather(*tasks)

    # Filter generation failures
    valid = [s for s in candidates if not getattr(s, 'remove_sample', False)]
    if not valid:
        return []

    # Evaluate with reward model (supports all rm_type via batched_async_rm)
    # For deepscaler: local computation, no network
    # For remote_rm: network request with rm_max_concurrent_requests semaphore
    rewards = await batched_async_rm(args, valid, evaluation=False)

    # Attach rewards
    for sample, reward in zip(valid, rewards):
        sample.reward = float(reward)

    # Filter by reward threshold
    threshold = getattr(args, 'rft_reward_threshold', 0.5)
    correct = [s for s in valid if s.reward > threshold]

    return correct


async def _generate_rollout_async(args, rollout_id, data_buffer):
    """Main async rollout logic."""
    target_size = args.rollout_batch_size
    over_sampling_size = args.over_sampling_batch_size

    sampling_params = {
        "temperature": getattr(args, "rollout_temperature", 0.7),
        "top_p": getattr(args, "rollout_top_p", 0.9),
        "max_new_tokens": getattr(args, "rollout_max_response_len", 2048),
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    result_groups = []
    stats = {"prompts_tried": 0, "candidates_generated": 0, "candidates_correct": 0}

    while len(result_groups) < target_size:
        # Sample prompts
        batch_size = min(over_sampling_size, target_size * 2)
        prompt_groups = data_buffer.get_samples(batch_size)
        prompt_samples = [group[0] for group in prompt_groups]

        # Process all prompts in parallel
        tasks = [_generate_and_filter_prompt(args, p, sampling_params) for p in prompt_samples]
        results = await asyncio.gather(*tasks)

        # Collect results
        for correct_samples in results:
            stats["prompts_tried"] += 1
            stats["candidates_generated"] += args.n_samples_per_prompt

            if correct_samples:
                stats["candidates_correct"] += len(correct_samples)
                result_groups.append(correct_samples)  # Variable-length group

                if len(result_groups) >= target_size:
                    break

        if len(result_groups) >= target_size:
            break

    # Trim to target size
    result_groups = result_groups[:target_size]

    # Log stats
    final_count = sum(len(g) for g in result_groups)
    filter_rate = 1.0 - (stats["candidates_correct"] / max(stats["candidates_generated"], 1))
    logger.info(
        f"[RFT] Rollout {rollout_id}: "
        f"prompts={stats['prompts_tried']}, "
        f"generated={stats['candidates_generated']}, "
        f"correct={stats['candidates_correct']}, "
        f"groups={len(result_groups)}, "
        f"samples={final_count}, "
        f"filter_rate={filter_rate:.1%}"
    )

    return result_groups


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """RFT rollout entry point.

    Returns list of sample groups (variable-length), which will be
    automatically flattened by the framework for SFT training.
    """
    assert not evaluation, "[RFT] Evaluation mode not supported"
    assert args.rollout_global_dataset, "[RFT] --rollout-global-dataset required"

    global _TOKENIZER, _MASK_GENERATOR

    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    if _MASK_GENERATOR is None:
        _MASK_GENERATOR = MultiTurnLossMaskGenerator(_TOKENIZER, tokenizer_type=args.loss_mask_type)

    result = run(_generate_rollout_async(args, rollout_id, data_buffer))

    if not result:
        logger.warning("[RFT] No samples passed filtering")

    return result
