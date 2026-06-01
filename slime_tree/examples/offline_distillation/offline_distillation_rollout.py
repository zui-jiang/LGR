"""Offline distillation rollout: use a teacher model (via external SGLang server) to generate
responses for prompts, then use the generated data to SFT the student model.

This approach:
- Does NOT require the rollout SGLang engine (use --debug-train-only to skip it).
- Calls the teacher server concurrently via asyncio, controlled by --rm-max-concurrent-requests.
- Produces samples with SFT loss mask (only the teacher-generated response tokens are trained on).

Key arguments (add to your launch script):
    --rollout-function-path examples.offline_distillation.offline_distillation_rollout.generate_rollout
    --rm-url http://<teacher-host>:<teacher-port>/generate
    --rm-max-concurrent-requests 32        # tune based on teacher GPU count / VRAM
    --loss-type sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --debug-train-only                     # skip student SGLang initialisation
"""

import asyncio
import logging

from slime.utils.async_utils import run
from slime.utils.http_utils import post
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_tokenizer

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (initialised once per worker process)
# ---------------------------------------------------------------------------
_TOKENIZER = None
_MASK_GENERATOR = None
_SEMAPHORE: asyncio.Semaphore | None = None
_SAMPLE_PRINTED = False


def _get_semaphore(args) -> asyncio.Semaphore:
    """Return the global semaphore, creating it on first call."""
    global _SEMAPHORE
    if _SEMAPHORE is None:
        concurrency = getattr(args, "rm_max_concurrent_requests", 32)
        _SEMAPHORE = asyncio.Semaphore(concurrency)
        logger.info(f"offline_distillation: teacher semaphore concurrency = {concurrency}")
    return _SEMAPHORE


# ---------------------------------------------------------------------------
# Per-sample async generation
# ---------------------------------------------------------------------------
async def _generate_one(args, sample, sampling_params: dict) -> None:
    """Call the teacher server for a single sample and fill sample fields in-place.

    On success: sample.tokens / response_length / loss_mask / response are populated.
    On failure: sample.remove_sample is set to True.
    """
    global _SAMPLE_PRINTED

    teacher_url: str = args.rm_url
    teacher_model_name = getattr(args, "teacher_model_name", None)

    # Apply the chat template to get the prompt text sent to teacher.
    # sample.prompt can be a list of OpenAI-style message dicts (role/content pairs)
    # or already a formatted string (when --apply-chat-template is used at data loading).
    if isinstance(sample.prompt, str):
        prompt_text = sample.prompt
    else:
        prompt_text = _TOKENIZER.apply_chat_template(
            sample.prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

    payload = {
        "text": prompt_text,
        "sampling_params": sampling_params,
        "return_logprob": False,
    }
    if teacher_model_name:
        payload["model"] = teacher_model_name

    semaphore = _get_semaphore(args)
    async with semaphore:
        try:
            output = await post(teacher_url, payload, max_retries=50)
        except Exception as exc:
            logger.warning(f"offline_distillation: teacher call failed: {exc}")
            sample.remove_sample = True
            return

    teacher_response: str = output["text"]

    # Build the full conversation and compute SFT loss mask.
    # When sample.prompt is already a formatted string (--apply-chat-template),
    # bypass MultiTurnLossMaskGenerator and tokenize directly.
    if isinstance(sample.prompt, str):
        prompt_ids = _TOKENIZER(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = _TOKENIZER(prompt_text + teacher_response, add_special_tokens=False)["input_ids"]
        response_length = len(full_ids) - len(prompt_ids)
        token_ids = full_ids
        loss_mask = [0] * len(prompt_ids) + [1] * response_length
    else:
        complete_messages = list(sample.prompt) + [{"role": "assistant", "content": teacher_response}]
        # MultiTurnLossMaskGenerator sets mask=1 only for the last assistant turn tokens.
        token_ids, loss_mask = _MASK_GENERATOR.get_loss_mask(complete_messages)
        response_length = _MASK_GENERATOR.get_response_lengths([loss_mask])[0]

    if response_length == 0:
        logger.warning("offline_distillation: teacher returned empty response, skipping sample")
        sample.remove_sample = True
        return

    sample.tokens = token_ids
    sample.response = teacher_response
    sample.response_length = response_length
    sample.reward = 0  # not used for SFT
    sample.loss_mask = loss_mask[-response_length:]

    if not _SAMPLE_PRINTED:
        logger.info(
            "offline_distillation example sample\n"
            f"  prompt  : {sample.prompt}\n"
            f"  response: {teacher_response!r}\n"
            f"  response_length: {response_length}"
        )
        _SAMPLE_PRINTED = True


# ---------------------------------------------------------------------------
# Batch async generation
# ---------------------------------------------------------------------------
async def _generate_batch(args, samples):
    """Concurrently generate teacher responses for all samples in the batch."""
    sampling_params = {
        "temperature": getattr(args, "rollout_temperature", 0.7),
        "top_p": getattr(args, "rollout_top_p", 0.9),
        "max_new_tokens": getattr(args, "rollout_max_response_len", 2048),
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    tasks = [asyncio.create_task(_generate_one(args, sample, sampling_params)) for sample in samples]
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Public rollout entry point (sync wrapper required by slime)
# ---------------------------------------------------------------------------
def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Custom rollout that uses an external teacher model (SGLang) to generate SFT data.

    Signature matches slime's --rollout-function-path interface:
        generate_rollout(args, rollout_id, data_buffer, evaluation=False)
            -> list[list[Sample]]

    Usage in launch script:
        --rollout-function-path \\
            examples.offline_distillation.offline_distillation_rollout.generate_rollout
        --rm-url http://<teacher-host>:<teacher-port>/generate
        --loss-type sft_loss
        --calculate-per-token-loss
        --disable-compute-advantages-and-returns
        --debug-train-only
    """
    assert not evaluation, "offline_distillation_rollout does not support evaluation mode"
    assert args.rollout_global_dataset, "--rollout-global-dataset must be set"

    global _TOKENIZER, _MASK_GENERATOR

    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    if _MASK_GENERATOR is None:
        _MASK_GENERATOR = MultiTurnLossMaskGenerator(_TOKENIZER, tokenizer_type=args.loss_mask_type)

    assert getattr(args, "rm_url", None), (
        "--rm-url must be set to the teacher model's SGLang generate endpoint, "
        "e.g. http://localhost:30001/generate"
    )

    # data_buffer.get_samples returns list[list[Sample]] (groups of n_samples_per_prompt)
    groups = data_buffer.get_samples(args.rollout_batch_size)

    # Flatten to individual samples for concurrent generation
    flat_samples = [sample for group in groups for sample in group]

    # Run all requests concurrently (bounded by _SEMAPHORE)
    run(_generate_batch(args, flat_samples))

    # Filter out failed samples and repack into groups
    result = []
    for group in groups:
        filtered = [s for s in group if not getattr(s, "remove_sample", False)]
        if filtered:
            result.append(filtered)

    if not result:
        logger.warning("offline_distillation: all samples in this batch failed, returning empty list")

    return result
