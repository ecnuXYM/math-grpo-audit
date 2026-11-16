#!/usr/bin/env python3
"""Lightweight tests for the round reward implementation.

This script is intentionally independent from pytest so it can be run directly
on the training server before launching an expensive slime job.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from own.reward_rounds import answer_utils
from own.reward_rounds.core import RoundRewardConfig, compute_round_reward


DEFAULT_ROLLOUT = Path("/root/own/debug_rollouts/qwen2.5-1.5B-round-smoke-32p2-len1024/rollout_0.pt")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_answer_utils() -> dict:
    cases = [
        {
            "name": "zh_answer_boxed",
            "response": "推理过程\n答案: \\boxed{3}",
            "label": "3",
            "expected_extracted": "3",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 1.0,
        },
        {
            "name": "en_answer_boxed",
            "response": "Reasoning\nAnswer: \\boxed{3}",
            "label": "3",
            "expected_extracted": "3",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 1.0,
        },
        {
            "name": "same_line_answer",
            "response": "I solve normally. Answer: 2325.",
            "label": "2325",
            "expected_extracted": "2325.",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 0.0,
        },
        {
            "name": "fraction_decimal_equiv",
            "response": "Answer: \\boxed{\\frac{1}{2}}",
            "label": "0.5",
            "expected_extracted": "\\frac{1}{2}",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 1.0,
        },
        {
            "name": "sqrt_equiv",
            "response": "Answer: \\boxed{\\sqrt{8}}",
            "label": "2\\sqrt{2}",
            "expected_extracted": "\\sqrt{8}",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 1.0,
        },
        {
            "name": "set_order_equiv",
            "response": "Answer: \\boxed{\\{2,1\\}}",
            "label": "\\{1,2\\}",
            "expected_extracted": "\\{2,1\\}",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 1.0,
        },
        {
            "name": "display_math_unboxed",
            "response": "Reasoning\nAnswer: \\[ 44 \\]",
            "label": "44",
            "expected_extracted": "\\[ 44 \\]",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 0.0,
        },
        {
            "name": "variable_equation_answer",
            "response": "Reasoning\nAnswer: x = 27.",
            "label": "27",
            "expected_extracted": "x = 27.",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 0.0,
        },
        {
            "name": "answer_with_unit_word",
            "response": "Reasoning\nAnswer: 31 stickers",
            "label": "31",
            "expected_extracted": "31 stickers",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 0.0,
        },
        {
            "name": "label_variable_equation_equiv",
            "response": "Reasoning\nAnswer: \\boxed{5}",
            "label": "x=5",
            "expected_extracted": "5",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 1.0,
        },
        {
            "name": "dfrac_frac_equiv",
            "response": "Reasoning\nAnswer: \\boxed{\\dfrac{13}{18}}",
            "label": "\\frac{13}{18}",
            "expected_extracted": "\\dfrac{13}{18}",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 1.0,
        },
        {
            "name": "label_shorthand_frac_equiv",
            "response": "Reasoning\nAnswer: \\boxed{\\frac{9}{19}}",
            "label": "\\frac9{19}",
            "expected_extracted": "\\frac{9}{19}",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 1.0,
        },
        {
            "name": "label_leading_dot_decimal_equiv",
            "response": "Reasoning\nAnswer: \\boxed{0.0000672}",
            "label": ".0000672",
            "expected_extracted": "0.0000672",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 1.0,
        },
        {
            "name": "answer_math_span_with_parenthetical",
            "response": "Reasoning\nAnswer: $3\\sqrt{13}$ (boxed $3\\sqrt{13}$).",
            "label": "3\\sqrt{13}",
            "expected_extracted": "$3\\sqrt{13}$ (boxed $3\\sqrt{13}$).",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 0.0,
        },
        {
            "name": "answer_sentence_unique_fraction_span",
            "response": "Reasoning\nAnswer: Therefore, the value of \\(r\\) is \\(\\frac{5}{9}\\).",
            "label": "\\frac{5}{9}",
            "expected_extracted": "Therefore, the value of \\(r\\) is \\(\\frac{5}{9}\\).",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 0.0,
        },
        {
            "name": "base_subscript_braces_equiv",
            "response": "Reasoning\nAnswer: $4210_5$",
            "label": "4210_{5}",
            "expected_extracted": "$4210_5$",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 0.0,
        },
        {
            "name": "thousands_comma_latex_spacing_equiv",
            "response": "Reasoning\nAnswer: \\boxed{45045}",
            "label": "45,\\!045",
            "expected_extracted": "45045",
            "expected_correct": True,
            "expected_parse_ok": True,
            "expected_format": 1.0,
        },
        {
            "name": "multiple_choice_text_letter_equiv",
            "response": "Reasoning\nAnswer: \\boxed{C}",
            "label": "\\text{(C)},",
            "expected_extracted": "C",
            "expected_correct": True,
            "expected_parse_ok": False,
            "expected_format": 1.0,
        },
        {
            "name": "signed_fraction_not_unsigned",
            "response": "Reasoning\nAnswer: \\boxed{x = -\\frac{3}{2}}",
            "label": "x = \\frac{3}{2}",
            "expected_extracted": "x = -\\frac{3}{2}",
            "expected_correct": False,
            "expected_parse_ok": True,
            "expected_format": -0.25,
        },
        {
            "name": "unicode_interval_union_equiv",
            "response": "Reasoning\nAnswer: \\((-∞, -7) ∪ (1, 2)\\)",
            "label": "(-\\infty,-7) \\cup (1,2)",
            "expected_extracted": "\\((-∞, -7) ∪ (1, 2)\\)",
            "expected_correct": True,
            "expected_parse_ok": False,
            "expected_format": 0.0,
        },
        {
            "name": "interval_endpoint_not_equiv",
            "response": "Reasoning\nAnswer: \\boxed{(4,\\infty)}",
            "label": "[4,\\infty)",
            "expected_extracted": "(4,\\infty)",
            "expected_correct": False,
            "expected_parse_ok": False,
            "expected_format": -0.25,
        },
    ]

    for case in cases:
        result = compute_round_reward("", case["response"], case["label"])
        check(result.extracted_answer == case["expected_extracted"], f"{case['name']}: bad extraction {result.extracted_answer!r}")
        check(result.correct is case["expected_correct"], f"{case['name']}: bad correctness {result.correct}")
        check(result.answer_parse_ok is case["expected_parse_ok"], f"{case['name']}: bad parse_ok {result.answer_parse_ok}")
        check(result.format == case["expected_format"], f"{case['name']}: bad format {result.format}")

    for pred, label in [("1/2", "0.5"), ("\\sqrt{8}", "2\\sqrt{2}"), ("\\{2,1\\}", "\\{1,2\\}")]:
        correct, parse_ok = answer_utils.is_correct_answer("", pred, label)
        check(correct and parse_ok, f"equivalence failed: {pred!r} vs {label!r}")

    wrong_response = "I solve normally. Answer: 2325."
    default_reward = compute_round_reward("", wrong_response, "71")
    penalty_reward = compute_round_reward(
        "",
        wrong_response,
        "71",
        config=RoundRewardConfig(wrong_base_penalty=-0.05),
    )
    check(default_reward.raw_reward <= 0, "wrong answer got positive default reward")
    check(
        math.isclose(penalty_reward.raw_reward, default_reward.raw_reward - 0.05, rel_tol=1e-9, abs_tol=1e-9),
        "wrong_base_penalty did not subtract exactly 0.05",
    )

    return {
        "case_count": len(cases),
        "wrong_default_raw": default_reward.raw_reward,
        "wrong_penalty_raw": penalty_reward.raw_reward,
    }


def load_samples(path: Path):
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["samples"] if isinstance(data, dict) and "samples" in data else data


def sample_get(sample, key, default=None):
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def replay_rollout(path: Path, wrong_base_penalty: float) -> dict:
    samples = load_samples(path)
    results = []
    config = RoundRewardConfig(wrong_base_penalty=wrong_base_penalty)

    for sample in samples:
        metadata = {
            "response_length": sample_get(sample, "response_length"),
            "status": sample_get(sample, "status"),
        }
        results.append(
            compute_round_reward(
                prompt=sample_get(sample, "prompt"),
                response=sample_get(sample, "response"),
                label=sample_get(sample, "label"),
                metadata=metadata,
                config=config,
            )
        )

    raw_rewards = [result.raw_reward for result in results]
    summary = {
        "path": str(path),
        "samples": len(results),
        "correct": sum(result.correct for result in results),
        "correct_rate": sum(result.correct for result in results) / len(results) if results else 0.0,
        "parse_ok": sum(result.answer_parse_ok for result in results),
        "format_nonzero": sum(result.format != 0 for result in results),
        "zero_wrong_raw": sum((not result.correct) and result.raw_reward == 0 for result in results),
        "positive_wrong_raw": sum((not result.correct) and result.raw_reward > 0 for result in results),
        "raw_reward_mean": mean(raw_rewards) if raw_rewards else None,
        "raw_reward_min": min(raw_rewards) if raw_rewards else None,
        "raw_reward_max": max(raw_rewards) if raw_rewards else None,
        "wrong_base_penalty": wrong_base_penalty,
    }

    check(summary["positive_wrong_raw"] == 0, "wrong answers received positive raw rewards")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--wrong-base-penalty", type=float, default=0.0)
    args = parser.parse_args()

    output = {"unit": test_answer_utils()}
    if not args.skip_rollout:
        check(args.rollout.exists(), f"rollout file does not exist: {args.rollout}")
        output["rollout"] = replay_rollout(args.rollout, args.wrong_base_penalty)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("OK")


if __name__ == "__main__":
    main()
