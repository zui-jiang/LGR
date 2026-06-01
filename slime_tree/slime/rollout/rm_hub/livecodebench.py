"""LiveCodeBench reward model for code generation evaluation."""

import json
import logging
import multiprocessing
import os
import sys
from typing import Any

import numpy as np

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Import LiveCodeBench testing utilities
try:
    _eval_utils_path = os.path.join(os.path.dirname(__file__), "eval_utils")
    if _eval_utils_path not in sys.path:
        sys.path.insert(0, _eval_utils_path)
    from livecodebench_utils.compute_code_generation_metrics import _temp_run
    HAS_LIVECODEBENCH_UTILS = True
except ImportError as e:
    HAS_LIVECODEBENCH_UTILS = False
    logger.warning(
        f"{e} LiveCodeBench testing utilities not found. "
        "Code execution evaluation will not work."
    )


def extract_code(text: str) -> str:
    """
    Extract Python code from markdown code blocks.

    Args:
        text: Response text that may contain code in markdown blocks

    Returns:
        Extracted code string, or empty string if no code found
    """
    outputlines = text.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]

    if len(indexlines) >= 2:
        # Extract code between first and last ```
        return "\n".join(outputlines[indexlines[-2] + 1:indexlines[-1]])

    # Try to find code without markdown blocks
    code_lines = []
    in_code = False
    for line in outputlines:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)

    return ""


def evaluate_livecodebench(
    response: str,
    label: str,
    metadata: dict[str, Any] | None = None,
    timeout: int = 300,
) -> float:
    """
    Evaluates LiveCodeBench code generation by executing against test cases.

    Args:
        response: Model's generated code (may be in markdown)
        label: JSON string containing evaluation_sample with input/output test cases
        metadata: Additional metadata (unused currently)
        timeout: Timeout in seconds for code execution (default: 30)

    Returns:
        Reward score: 1.0 if all tests pass, 0.0 otherwise
    """
    if not HAS_LIVECODEBENCH_UTILS:
        logger.error("LiveCodeBench testing utilities not available")
        return 0.0

    if not response:
        logger.warning("Empty response")
        return 0.0

    # Extract code from response
    gen_code = extract_code(response)

    if not gen_code:
        logger.warning("Could not extract code from response")
        return 0.0

    try:
        # Parse label (should be JSON string with evaluation_sample)
        if isinstance(label, str):
            evaluation_data = json.loads(label)
        else:
            evaluation_data = label

        # Get input_output string
        if "evaluation_sample" in evaluation_data:
            input_output_str = evaluation_data["evaluation_sample"].get("input_output", "{}")
        else:
            input_output_str = evaluation_data.get("input_output", "{}")

        if not input_output_str:
            logger.warning("No input_output in evaluation data")
            return 0.0

        # Parse input_output
        input_output = json.loads(input_output_str) if isinstance(input_output_str, str) else input_output_str
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])

        if not inputs or not outputs:
            logger.warning(f"No test cases: {len(inputs)} inputs, {len(outputs)} outputs")
            return 0.0

        # Build sample dict for testing utility
        sample = {
            "input_output": json.dumps(input_output) if isinstance(input_output, dict) else input_output
        }

        # Run tests using LiveCodeBench testing utility with timeout protection
        calculated_timeout = (timeout + 1) * len(inputs) + 5
        max_timeout = 300  # Cap at 5 minutes
        actual_timeout = min(calculated_timeout, max_timeout)

        manager = multiprocessing.Manager()
        result = manager.list()
        metadata_list = manager.list()

        p = multiprocessing.Process(
            target=_temp_run,
            args=(sample, gen_code, False, result, metadata_list, timeout),
        )
        p.start()
        p.join(timeout=actual_timeout)

        # Force kill if still alive (timeout)
        if p.is_alive():
            logger.warning(f"Process timed out after {actual_timeout}s, killing...")
            try:
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)
            except Exception as e:
                logger.error(f"Error killing process: {e}")

            # Timeout = failure
            return 0.0

        if not result:
            logger.warning("No result returned from test execution")
            return 0.0

        res = result[0]

        # Convert results to boolean list
        fixed = []
        for e in res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            if e != True and e != False:
                e = False
            fixed.append(e)
        res = fixed

        # Check if all tests pass
        all_pass = all(res) if res else False

        # Return binary reward: 1.0 for pass, 0.0 for fail
        return 1.0 if all_pass else 0.0

    except Exception as e:
        logger.error(f"LiveCodeBench evaluation failed: {e}", exc_info=True)
        return 0.0


async def compute_livecodebench_reward(args, sample: Sample, **kwargs) -> float:
    """
    Async wrapper for LiveCodeBench evaluation (compatible with slime's async_rm interface).

    This function can be used as a custom RM in slime by setting:
    - custom_rm_path: "slime.rollout.rm_hub.livecodebench:compute_livecodebench_reward"

    Args:
        args: Training/evaluation arguments
        sample: Sample object containing:
            - response: Generated code
            - label: Evaluation data (JSON string or dict)
            - metadata: Additional metadata
        **kwargs: Additional keyword arguments

    Returns:
        Reward score (0.0 or 1.0)
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    timeout = getattr(args, 'lcb_timeout', 30)

    return evaluate_livecodebench(
        response=sample.response,
        label=sample.label,
        metadata=metadata,
        timeout=timeout,
    )


# Batch version for better performance
async def compute_livecodebench_reward_batch(args, samples: list[Sample], **kwargs) -> list[float]:
    """
    Batch evaluation for LiveCodeBench (parallel processing).

    Args:
        args: Training/evaluation arguments
        samples: List of Sample objects
        **kwargs: Additional keyword arguments

    Returns:
        List of reward scores
    """
    import asyncio

    # Run evaluations in parallel using asyncio
    tasks = [compute_livecodebench_reward(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return list(rewards)
