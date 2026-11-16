"""Round-style rule rewards for math GRPO experiments.

This module intentionally has no dependency on slime.  It can be used by
offline Reward QA and by the slime adapter.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from own.reward_rounds import answer_utils


@dataclass
class RoundRewardConfig:
    round_name: str = "round5"
    gamma: float = 3.0
    alpha: float = 0.5
    format_weight: float = 0.05
    anti_hack_weight: float = 0.15
    length_weight: float = 0.1
    wrong_base_penalty: float = 0.0
    anti_hack_mode: str = "absorbed"
    soft_length_cap: int = 768
    hard_length_cap: int = 1024
    min_reward: float = -1.0
    max_reward: float = 1.0


@dataclass
class RewardBreakdown:
    round_name: str
    extracted_answer: str
    correct: bool
    severity_raw: float
    severity: float
    gated_accuracy: float
    format: float
    anti_hack: float
    length: float
    raw_reward: float
    final_reward: float
    answer_parse_ok: bool
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_round_reward(
    prompt: Any,
    response: str,
    label: Any,
    metadata: dict[str, Any] | None = None,
    *,
    config: RoundRewardConfig | None = None,
) -> RewardBreakdown:
    """Compute a single-sample round reward breakdown."""

    cfg = config or RoundRewardConfig()
    metadata = metadata or {}
    response = response or ""

    extracted = answer_utils.extract_answer(response)
    correct, parse_ok = answer_utils.is_correct_answer(response=response, extracted=extracted, label=label)
    severity_raw, features = compute_severity(response=response, extracted=extracted, prompt=prompt)
    severity = clip(severity_raw, 0.0, 1.0)

    if correct:
        gated_accuracy = (1.0 - severity) ** cfg.gamma
    else:
        gated_accuracy = cfg.wrong_base_penalty - cfg.alpha * (severity**cfg.gamma)

    format_component = compute_format_component(response=response, correct=correct)
    anti_hack = 0.0 if cfg.anti_hack_mode == "absorbed" else -severity
    length_component = compute_length_component(response=response, metadata=metadata, config=cfg)

    raw_reward = (
        gated_accuracy
        + cfg.format_weight * format_component
        + cfg.anti_hack_weight * anti_hack
        + cfg.length_weight * length_component
    )

    return RewardBreakdown(
        round_name=cfg.round_name,
        extracted_answer=extracted,
        correct=correct,
        severity_raw=severity_raw,
        severity=severity,
        gated_accuracy=gated_accuracy,
        format=format_component,
        anti_hack=anti_hack,
        length=length_component,
        raw_reward=float(raw_reward),
        final_reward=float(raw_reward),
        answer_parse_ok=parse_ok,
        features=features,
    )


def extract_answer(response: str) -> str:
    return answer_utils.extract_answer(response)


def is_correct_answer(response: str, extracted: str, label: Any) -> tuple[bool, bool]:
    return answer_utils.is_correct_answer(response=response, extracted=extracted, label=label)


def compute_severity(response: str, extracted: str, prompt: Any = None) -> tuple[float, dict[str, float]]:
    """Aggregate visible reward-hacking signals into a 0..1 severity score."""

    text = response or ""
    low = text.lower()
    features: dict[str, float] = {}

    hedge_patterns = [
        r"\bmaybe\b",
        r"\bprobably\b",
        r"\bpossibly\b",
        r"\bi guess\b",
        r"\bi think\b",
        r"\bshould be\b",
        r"\bcould be\b",
        r"\bapproximately\b",
        r"\u53ef\u80fd",
        r"\u5927\u6982",
        r"\u4e5f\u8bb8",
        r"\u5e94\u8be5",
        r"\u731c",
    ]
    features["hedge"] = min(1.0, _count_patterns(low, hedge_patterns) * 0.18)

    hack_patterns = [
        r"final answer is correct",
        r"the answer is obviously",
        r"trust me",
        r"without solving",
        r"no need to prove",
        r"just put",
        r"\u76f4\u63a5\u7ed9",
        r"\u4e0d\u9700\u8981\u8bc1\u660e",
        r"\u7b54\u6848\u6b63\u786e",
    ]
    features["hack_phrase"] = min(1.0, _count_patterns(low, hack_patterns) * 0.25)

    candidate_markers = len(re.findall(r"\b(?:or|maybe|could be)\b", low))
    boxed_count = len(re.findall(r"\\boxed\s*\{", text))
    answer_count = answer_utils.count_answer_anchors(text)
    features["multi_candidate"] = 0.0
    if boxed_count > 1 or answer_count > 1 or candidate_markers >= 2:
        features["multi_candidate"] = 0.25

    step_markers = _count_patterns(
        low,
        [
            r"\bstep\s*\d+",
            r"\btherefore\b",
            r"\bbecause\b",
            r"\bsince\b",
            r"\bso\b",
            r"\bthus\b",
            r"\u56e0\u4e3a",
            r"\u6240\u4ee5",
            r"\u56e0\u6b64",
        ],
    )
    math_symbols = len(re.findall(r"[=+\-*/^]|\\frac|\\sqrt", text))
    if extracted and len(text.strip()) < 160:
        features["low_information"] = 0.25
    elif extracted and step_markers + math_symbols < 3:
        features["low_information"] = 0.15
    else:
        features["low_information"] = 0.0

    repetition_score = _repetition_score(low)
    features["repetition"] = repetition_score

    special_value_patterns = [
        r"\btry\s+\w+\s*=",
        r"\bplug in\b",
        r"\btest cases?\b",
        r"\bspecial case\b",
        r"\u4ee3\u5165",
        r"\u7279\u6b8a\u503c",
    ]
    features["special_value_shortcut"] = min(1.0, _count_patterns(low, special_value_patterns) * 0.18)

    severity = sum(features.values())
    return clip(severity, 0.0, 1.0), features


def compute_format_component(response: str, correct: bool) -> float:
    passed = answer_utils.passes_format(response)
    if correct:
        return 1.0 if passed else 0.0
    return -0.25 if passed else 0.0


def compute_length_component(response: str, metadata: dict[str, Any], config: RoundRewardConfig) -> float:
    response_len = metadata.get("response_length") or metadata.get("completion_tokens")
    try:
        response_len = int(response_len)
    except Exception:
        response_len = max(1, len((response or "").split()))

    status = str(metadata.get("status") or "").lower()
    if "truncated" in status:
        return -1.0

    if response_len <= config.soft_length_cap:
        return 0.0
    if response_len >= config.hard_length_cap:
        return -1.0
    span = max(1, config.hard_length_cap - config.soft_length_cap)
    return -float(response_len - config.soft_length_cap) / span


def normalize_answer(text: str) -> str:
    return answer_utils.normalize_answer(text)


def clean_answer_text(text: str) -> str:
    return answer_utils.clean_answer_text(text)


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _extract_last_boxed(text: str) -> str:
    return answer_utils.extract_last_boxed(text)


def _math_verify_correct(response: str, extracted: str, label: str) -> bool | None:
    return answer_utils._math_verify_correct(response=response, extracted=extracted, label=label)


def _count_patterns(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)) for pattern in patterns)


def _repetition_score(text: str) -> float:
    words = re.findall(r"[a-zA-Z]{3,}", text)
    if len(words) < 24:
        return 0.0
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    repeated = sum(count - 3 for count in counts.values() if count > 3)
    return clip(repeated / max(1, len(words)) * 2.0, 0.0, 0.25)


def _to_float_or_none(text: str) -> float | None:
    return answer_utils.to_float_or_none(text)
