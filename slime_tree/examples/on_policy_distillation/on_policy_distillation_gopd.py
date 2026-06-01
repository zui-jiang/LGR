"""
G-OPD (Generalized On-Policy Distillation) with Reward Correction
论文: Learning beyond Teacher (arXiv:2602.12125v1)

配置参数:
  --opd-lambda: reward scaling factor (推荐 1.25 for ExOPD)
  --opd-use-reward-correction: 使用 teacher base model 作为参考
  --teacher-model-name: teacher RL model 名称
  --teacher-base-model-name: teacher base model 名称 (仅 reward correction)
  --opd-use-orm: 是否使用 ORM 计算任务奖励 (default: False)
  --rm-type: ORM 类型 (deepscaler/math/f1/gpqa 等)

使用场景:
  1. 纯蒸馏: --use-opd
  2. 蒸馏 + ORM: --use-opd --opd-use-orm --rm-type deepscaler
"""

import logging
import torch

from slime.utils.http_utils import post
from slime.utils.types import Sample
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward

logger = logging.getLogger(__name__)


async def reward_func(args, sample, **kwargs):
    teacher_model_name = getattr(args, 'teacher_model_name', None)
    use_reward_correction = getattr(args, 'opd_use_reward_correction', False)
    teacher_base_model_name = getattr(args, 'teacher_base_model_name', None)
    max_retries = getattr(args, 'teacher_max_retries', 60)

    payload_teacher = {
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
    if teacher_model_name:
        payload_teacher["model"] = teacher_model_name

    teacher_response = await post(args.rm_url, payload_teacher, max_retries=max_retries)

    teacher_base_response = None
    if use_reward_correction:
        if teacher_base_model_name is None:
            raise ValueError(
                "--opd-use-reward-correction enabled but --teacher-base-model-name not specified"
            )

        payload_teacher_base = {
            "input_ids": sample.tokens,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
            "model": teacher_base_model_name,
        }

        teacher_base_response = await post(args.rm_url, payload_teacher_base, max_retries=max_retries)

    return {
        "teacher_rl": teacher_response,
        "teacher_base": teacher_base_response,
    }


def post_process_rewards(args, samples: list[Sample], **kwargs):
    use_reward_correction = getattr(args, 'opd_use_reward_correction', False)
    use_orm = getattr(args, 'opd_use_orm', False)
    opd_lambda = getattr(args, 'opd_lambda', 1.0)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs = []
    teacher_base_log_probs = []

    for i, (reward, response_length) in enumerate(zip(raw_rewards, response_lengths, strict=False)):
        teacher_rl_response = reward["teacher_rl"]
        if teacher_rl_response is None:
            raise ValueError(f"Sample {i}: teacher_rl response is None")

        try:
            t_log_prob = torch.tensor(
                [item[0] for item in teacher_rl_response["meta_info"]["input_token_logprobs"][1:]],
                dtype=torch.float32
            )
        except (KeyError, TypeError, IndexError) as e:
            raise ValueError(f"Sample {i}: Failed to extract teacher log-probs: {e}") from e

        if len(t_log_prob) < response_length:
            raise ValueError(
                f"Sample {i}: teacher log-probs length ({len(t_log_prob)}) < response_length ({response_length})"
            )
        t_log_prob = t_log_prob[-response_length:]
        teacher_log_probs.append(t_log_prob)

        if use_reward_correction:
            teacher_base_response = reward["teacher_base"]
            if teacher_base_response is None:
                raise ValueError(f"Sample {i}: teacher_base response is None")

            try:
                tb_log_prob = torch.tensor(
                    [item[0] for item in teacher_base_response["meta_info"]["input_token_logprobs"][1:]],
                    dtype=torch.float32
                )
            except (KeyError, TypeError, IndexError) as e:
                raise ValueError(f"Sample {i}: Failed to extract teacher_base log-probs: {e}") from e

            if len(tb_log_prob) < response_length:
                raise ValueError(
                    f"Sample {i}: teacher_base log-probs length ({len(tb_log_prob)}) < response_length ({response_length})"
                )
            tb_log_prob = tb_log_prob[-response_length:]
            teacher_base_log_probs.append(tb_log_prob)

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    if use_reward_correction:
        for sample, tb_log_probs in zip(samples, teacher_base_log_probs, strict=False):
            sample.teacher_base_log_probs = tb_log_probs

    logger.info(
        f"[G-OPD] Lambda={opd_lambda}, RC={'ON' if use_reward_correction else 'OFF'}, "
        f"ORM={'ON' if use_orm else 'OFF'}"
    )

    if use_orm:
        import asyncio
        task_rewards = torch.tensor(
            [float(get_deepscaler_rule_based_reward(s.response, s.label)) for s in samples],
            dtype=torch.float,
        )
        rewards = task_rewards.reshape(-1, args.n_samples_per_prompt)
        rewards = rewards - rewards.mean(dim=-1, keepdim=True)
        if args.grpo_std_normalization:
            rewards = rewards / (rewards.std(dim=-1, keepdim=True) + 1e-6)
        scalar_rewards = rewards.flatten().tolist()
    else:
        logger.info("[G-OPD] Pure distillation: ORM disabled")
        task_rewards = torch.zeros(len(samples), dtype=torch.float)
        scalar_rewards = [0.0] * len(samples)

    return task_rewards.tolist(), scalar_rewards


