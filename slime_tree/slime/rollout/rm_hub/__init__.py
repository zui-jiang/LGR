import asyncio
import logging
import random
import time

import aiohttp

from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl

# Global semaphore to limit concurrent RM requests
# This prevents overwhelming external RM services with too many simultaneous requests
_RM_SEMAPHORE = None


def _get_rm_semaphore(max_concurrent_requests=32):
    """Get or create the global semaphore for RM requests."""
    global _RM_SEMAPHORE
    if _RM_SEMAPHORE is None:
        _RM_SEMAPHORE = asyncio.Semaphore(max_concurrent_requests)
    return _RM_SEMAPHORE


async def remote_rm(args, sample: Sample):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    # Add timeout for remote RM requests
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def async_rm(args, sample: Sample, **kwargs):
    # Get concurrency control semaphore
    max_concurrent = getattr(args, 'rm_max_concurrent_requests', 32)
    semaphore = _get_rm_semaphore(max_concurrent)

    # Record start time for this RM request
    start_time = time.time()

    # Apply semaphore to all RM calls
    async with semaphore:
        evaluation = kwargs.get("evaluation", False)

        # Select appropriate custom RM path based on evaluation flag
        if evaluation:
            custom_rm_path = getattr(args, "eval_custom_rm_path", None)
        else:
            custom_rm_path = args.custom_rm_path

        if custom_rm_path is not None:
            rm_function = load_function(custom_rm_path)
            result = await rm_function(args, sample, **kwargs)
            # Record RM time in sample for perf metrics
            sample.rm_time = time.time() - start_time
            return result

        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

        # Select appropriate rm_type based on evaluation flag
        if evaluation and hasattr(args, "eval_rm_type") and args.eval_rm_type is not None:
            rm_type = (metadata.get("rm_type") or args.eval_rm_type or "").strip()
        else:
            rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()

        response = sample.response
        label = sample.label
        if rm_type.startswith("boxed_"):
            response = extract_boxed_answer(response) or ""
            rm_type = rm_type[len("boxed_") :]

        # This function is intended for remote or time-consuming reward model evaluation.
        # Implement the actual logic as needed.
        if rm_type == "remote_rm":
            result = await remote_rm(args, sample)
        elif rm_type == "deepscaler":
            result = get_deepscaler_rule_based_reward(response, label)
        elif rm_type == "dapo":
            result = compute_score_dapo(response, label)
        elif rm_type == "math":
            result = 1 if grade_answer_verl(response, label) else 0
        elif rm_type == "f1":
            result = f1_score(response, label)[0]
        elif rm_type == "gpqa":
            result = compute_gpqa_reward(response, label, metadata=metadata)
        elif rm_type == "ifbench":
            from .ifbench import compute_ifbench_reward

            result = compute_ifbench_reward(response, label, metadata=metadata)
        elif rm_type == "livecodebench":
            from .livecodebench import evaluate_livecodebench

            result = evaluate_livecodebench(response, label, metadata=metadata, timeout=getattr(args, 'lcb_timeout', 30))
        elif rm_type == "random":
            result = random.randint(0, 1)
        elif rm_type:
            raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
        else:
            raise NotImplementedError("Rule-based RM type is not specified.")

        # Record RM time in sample for perf metrics
        sample.rm_time = time.time() - start_time
        return result


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    """
    Batch async RM with concurrency control.

    Concurrency control is handled inside async_rm(), so we just call it directly.
    This avoids double-semaphore issues.
    """
    evaluation = kwargs.get("evaluation", False)

    # Select appropriate custom RM path based on evaluation flag
    # Same logic as async_rm: evaluation uses eval_custom_rm_path if set, otherwise falls back to rm_type
    if evaluation:
        custom_rm_path = getattr(args, "eval_custom_rm_path", None)
    else:
        custom_rm_path = args.custom_rm_path

    if custom_rm_path is not None:
        # For custom RM, check if it accepts batch mode
        rm_function = load_function(custom_rm_path)
        # Try to call in batch mode, if not supported will fall back to individual calls
        try:
            return await rm_function(args, samples, **kwargs)
        except TypeError:
            # Fall back to individual calls with concurrency control
            pass

    # Call async_rm directly for each sample
    # Concurrency control is handled inside async_rm() via its internal semaphore
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
