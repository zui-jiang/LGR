import torch

from slime.utils.http_utils import post
from slime.utils.types import Sample
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward

async def reward_func(args, sample, **kwargs):
    """Call teacher model via SGLang API with model selection support.

    Uses slime's http_utils.post() which includes retry logic (max 60 retries, 1s interval).
    """

    # Get model name from args if specified, otherwise use default
    teacher_model_name = getattr(args, 'teacher_model_name', None)

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
    if teacher_model_name:
        payload["model"] = teacher_model_name

    # Use slime's post() with built-in retry mechanism (max 60 retries, 1s interval each)
    max_retries = getattr(args, 'teacher_max_retries', 60)
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


    task_rewards = torch.tensor(
        [float(get_deepscaler_rule_based_reward(s.response, s.label)) for s in samples],
        dtype=torch.float,
    )

    rewards = task_rewards.reshape(-1, args.n_samples_per_prompt)
    rewards = rewards - rewards.mean(dim=-1, keepdim=True)
    if args.grpo_std_normalization:
        rewards = rewards / (rewards.std(dim=-1, keepdim=True) + 1e-6)
    scalar_rewards = rewards.flatten().tolist()

    return task_rewards.tolist(), scalar_rewards

