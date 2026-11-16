"""Slime custom reward adapter for the round reward functions."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import torch

from own.reward_rounds.core import RoundRewardConfig, clip, compute_round_reward


def _get_float_arg_or_env(args, attr_name: str, env_name: str, default: float) -> float:
    value = getattr(args, attr_name, None)
    if value is None:
        value = os.environ.get(env_name, default)
    return float(value)


def _config_from_args(args) -> RoundRewardConfig:
    return RoundRewardConfig(
        round_name=getattr(args, "round_reward_name", "round5"),
        gamma=float(getattr(args, "round_reward_gamma", 3.0)),
        alpha=float(getattr(args, "round_reward_alpha", 0.5)),
        wrong_base_penalty=_get_float_arg_or_env(
            args,
            "round_reward_wrong_base_penalty",
            "ROUND_REWARD_WRONG_BASE_PENALTY",
            0.0,
        ),
        anti_hack_mode=getattr(args, "round_reward_anti_hack_mode", "absorbed"),
        format_weight=_get_float_arg_or_env(
            args,
            "round_reward_format_weight",
            "ROUND_REWARD_FORMAT_WEIGHT",
            0.05,
        ),
        anti_hack_weight=_get_float_arg_or_env(
            args,
            "round_reward_anti_hack_weight",
            "ROUND_REWARD_ANTI_HACK_WEIGHT",
            0.15,
        ),
        length_weight=_get_float_arg_or_env(
            args,
            "round_reward_length_weight",
            "ROUND_REWARD_LENGTH_WEIGHT",
            0.1,
        ),
        soft_length_cap=int(
            getattr(args, "round_reward_soft_length_cap", None)
            or os.environ.get("ROUND_REWARD_SOFT_LENGTH_CAP", 768)
        ),
        hard_length_cap=int(
            getattr(args, "round_reward_hard_length_cap", None)
            or os.environ.get("ROUND_REWARD_HARD_LENGTH_CAP", 1024)
        ),
    )


async def reward_func(args, sample, **kwargs) -> float:
    metadata = dict(sample.metadata or {})
    metadata["response_length"] = getattr(sample, "response_length", 0)
    metadata["status"] = str(getattr(sample, "status", ""))

    result = compute_round_reward(
        prompt=sample.prompt,
        response=sample.response,
        label=sample.label,
        metadata=metadata,
        config=_config_from_args(args),
    )
    breakdown = result.to_dict()

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["reward_breakdown"] = breakdown
    sample.metadata["raw_reward"] = breakdown["raw_reward"]
    sample.metadata["round_reward_correct"] = breakdown["correct"]
    sample.metadata["round_reward_severity"] = breakdown["severity"]

    return float(breakdown["raw_reward"])


def post_process_rewards(args, samples: list[Any], **kwargs):
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    final_rewards = _group_normalize(args, samples, raw_rewards)

    min_reward = float(getattr(args, "round_reward_min", -1.0))
    max_reward = float(getattr(args, "round_reward_max", 1.0))
    final_rewards = [clip(value, min_reward, max_reward) for value in final_rewards]

    # Preserve the round5 non-positive constraint:
    # raw_reward <= 0 must not become a positive training reward after normalization.
    final_rewards = [min(value, 0.0) if raw <= 0.0 else value for raw, value in zip(raw_rewards, final_rewards)]

    for sample, raw, final in zip(samples, raw_rewards, final_rewards):
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        breakdown = metadata.get("reward_breakdown")
        if isinstance(breakdown, dict):
            breakdown["normalized_reward"] = final
            breakdown["final_reward"] = final
        metadata["raw_reward"] = raw
        metadata["final_reward"] = final
        sample.metadata = metadata

    return raw_rewards, final_rewards


def _group_normalize(args, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    if not getattr(args, "rewards_normalization", True):
        return list(raw_rewards)

    n_samples = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    rollout_batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
    expected_flat = n_samples * rollout_batch_size

    if len(raw_rewards) == expected_flat:
        rewards = torch.tensor(raw_rewards, dtype=torch.float32).reshape(-1, n_samples)
        normalized = rewards - rewards.mean(dim=-1, keepdim=True)
        if getattr(args, "grpo_std_normalization", False) and n_samples > 1:
            normalized = normalized / (normalized.std(dim=-1, keepdim=True) + 1e-6)
        return normalized.flatten().tolist()

    # Fallback for uneven sample counts: group by Sample.group_index when present.
    by_group: dict[Any, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        group = getattr(sample, "group_index", None)
        if group is None:
            group = idx // max(1, n_samples)
        by_group[group].append(idx)

    output = [0.0] * len(raw_rewards)
    for indices in by_group.values():
        group_values = torch.tensor([raw_rewards[i] for i in indices], dtype=torch.float32)
        group_values = group_values - group_values.mean()
        if getattr(args, "grpo_std_normalization", False) and group_values.numel() > 1:
            group_values = group_values / (group_values.std() + 1e-6)
        for idx, value in zip(indices, group_values.tolist()):
            output[idx] = float(value)
    return output
