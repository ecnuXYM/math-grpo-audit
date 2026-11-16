#!/usr/bin/env python3
"""Run a small, deterministic Reward QA suite for the math GRPO reward.

The suite is intentionally separate from answer-extraction unit tests.  It
checks whether representative response styles receive the intended relative
ranking before a longer GRPO run is launched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from own.reward_rounds.core import RoundRewardConfig, compute_round_reward
from own.reward_rounds.qa_cases import LABEL, PROMPT, build_cases


def to_row(case: dict, result) -> dict:
    breakdown = result.to_dict()
    row = {
        "name": case["name"],
        "category": case["category"],
        "correct": breakdown["correct"],
        "answer_parse_ok": breakdown["answer_parse_ok"],
        "extracted_answer": breakdown["extracted_answer"],
        "severity": breakdown["severity"],
        "gated_accuracy": breakdown["gated_accuracy"],
        "format": breakdown["format"],
        "anti_hack": breakdown["anti_hack"],
        "length": breakdown["length"],
        "raw_reward": breakdown["raw_reward"],
        "features": json.dumps(breakdown["features"], ensure_ascii=False, sort_keys=True),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wrong-base-penalty", type=float, default=-0.05)
    args = parser.parse_args()

    config = RoundRewardConfig(wrong_base_penalty=args.wrong_base_penalty)
    results: dict[str, dict] = {}
    rows: list[dict] = []

    for case in build_cases():
        metadata = dict(case.get("metadata") or {})
        result = compute_round_reward(
            prompt=PROMPT,
            response=case["response"],
            label=LABEL,
            metadata=metadata,
            config=config,
        )
        results[case["name"]] = result.to_dict()
        rows.append(to_row(case, result))

    raw = {name: item["raw_reward"] for name, item in results.items()}
    checks = {
        "wrong_formatted_nonpositive": raw["wrong_formatted"] <= 0.0,
        "hack_wrong_formatted_nonpositive": raw["hack_wrong_formatted"] <= 0.0,
        "best_honest_beats_plain_correct": raw["best_honest"] > raw["plain_correct_unboxed"],
        "best_honest_beats_suspicious_correct": raw["best_honest"] > raw["suspicious_correct_multi_candidate"],
        "best_honest_beats_short_low_information": raw["best_honest"] > raw["short_low_information_correct"],
        "best_honest_beats_repetitive": raw["best_honest"] > raw["long_repetitive_correct"],
        "suspicious_correct_beats_wrong_formatted": raw["suspicious_correct_multi_candidate"] > raw["wrong_formatted"],
        "truncated_correct_below_best_honest": raw["truncated_correct"] < raw["best_honest"],
    }

    summary = {
        "label": LABEL,
        "config": {
            "round_name": config.round_name,
            "gamma": config.gamma,
            "alpha": config.alpha,
            "wrong_base_penalty": config.wrong_base_penalty,
            "format_weight": config.format_weight,
            "anti_hack_weight": config.anti_hack_weight,
            "length_weight": config.length_weight,
            "anti_hack_mode": config.anti_hack_mode,
        },
        "case_count": len(rows),
        "wrong_answer_positive_count": sum(not row["correct"] and row["raw_reward"] > 0.0 for row in rows),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "reward_qa_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_dir / "reward_qa_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    if not summary["all_checks_passed"]:
        raise SystemExit("Reward QA checks failed")


if __name__ == "__main__":
    main()
