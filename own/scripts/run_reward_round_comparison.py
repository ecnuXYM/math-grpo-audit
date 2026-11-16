#!/usr/bin/env python3
"""Compare project-aligned reward round1--round5 on a fixed QA case cluster.

This is an offline reward-design ablation, not five GRPO training runs.  It
keeps the project's answer extraction, severity features, and length signal
fixed, then changes only the reward composition to mirror the evolution in
``GRPO实验的经验与坑.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from own.reward_rounds import answer_utils
from own.reward_rounds.core import RoundRewardConfig, clip, compute_round_reward
from own.reward_rounds.qa_cases import LABEL, PROMPT, build_cases


ROUND_DESCRIPTIONS = {
    "round1": "accuracy + 0.05 * unconditional_format_pass",
    "round2": "accuracy + 0.05 * unconditional_format_pass - 0.15 * severity + 0.1 * length",
    "round3": "gated_accuracy(gamma=3) + 0.05 * correctness_gated_format - 0.15 * severity + 0.1 * length",
    "round4": "symmetric_gated_accuracy(gamma=1.5) + 0.05 * correctness_gated_format - 0.15 * severity + 0.1 * length",
    "round5": "current project raw reward: gamma=3, wrong_base_penalty=-0.05, absorbed severity",
}


def gated_accuracy(correct: bool, severity: float, *, gamma: float, alpha: float = 0.5, wrong_base_penalty: float = 0.0) -> float:
    if correct:
        return (1.0 - severity) ** gamma
    return wrong_base_penalty - alpha * (severity**gamma)


def score_rounds(*, correct: bool, severity: float, format_pass: bool, gated_format: float, length: float, round5_raw: float) -> dict[str, float]:
    accuracy = 1.0 if correct else 0.0
    unconditional_format = 1.0 if format_pass else 0.0
    return {
        "round1": accuracy + 0.05 * unconditional_format,
        "round2": accuracy + 0.05 * unconditional_format - 0.15 * severity + 0.1 * length,
        "round3": gated_accuracy(correct, severity, gamma=3.0) + 0.05 * gated_format - 0.15 * severity + 0.1 * length,
        "round4": gated_accuracy(correct, severity, gamma=1.5) + 0.05 * gated_format - 0.15 * severity + 0.1 * length,
        "round5": round5_raw,
    }


def summarize_round(name: str, rows: list[dict]) -> dict:
    wrong_rows = [row for row in rows if not row["correct"]]
    risk_rows = [row for row in rows if row["name"] in {"wrong_formatted", "hack_wrong_formatted"}]
    best = next(row for row in rows if row["name"] == "best_honest")[name]
    risk_mean = mean(row[name] for row in risk_rows)
    return {
        "round": name,
        "description": ROUND_DESCRIPTIONS[name],
        "wrong_answer_positive_count": sum(row[name] > 0.0 for row in wrong_rows),
        "avg_gap_best_honest_minus_risk": best - risk_mean,
        "best_honest_raw_reward": best,
        "wrong_formatted_reward": next(row for row in rows if row["name"] == "wrong_formatted")[name],
        "hack_wrong_formatted_reward": next(row for row in rows if row["name"] == "hack_wrong_formatted")[name],
        "best_beats_suspicious": best > next(row for row in rows if row["name"] == "suspicious_correct_multi_candidate")[name],
        "best_beats_low_information": best > next(row for row in rows if row["name"] == "short_low_information_correct")[name],
        "best_beats_repetition": best > next(row for row in rows if row["name"] == "long_repetitive_correct")[name],
    }


def round5_final_safety_group(rows: list[dict]) -> dict[str, float]:
    group = [row for row in rows if row["name"] in {"best_honest", "wrong_formatted", "hack_wrong_formatted"}]
    raw = [row["round5"] for row in group]
    group_mean = mean(raw)
    return {
        row["name"]: min(clip(row["round5"] - group_mean, -1.0, 1.0), 0.0) if row["round5"] <= 0.0 else clip(row["round5"] - group_mean, -1.0, 1.0)
        for row in group
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    current_config = RoundRewardConfig(wrong_base_penalty=-0.05)
    rows: list[dict] = []
    for case in build_cases():
        result = compute_round_reward(
            prompt=PROMPT,
            response=case["response"],
            label=LABEL,
            metadata=dict(case.get("metadata") or {}),
            config=current_config,
        )
        breakdown = result.to_dict()
        format_pass = answer_utils.passes_format(case["response"])
        scores = score_rounds(
            correct=breakdown["correct"],
            severity=breakdown["severity"],
            format_pass=format_pass,
            gated_format=breakdown["format"],
            length=breakdown["length"],
            round5_raw=breakdown["raw_reward"],
        )
        rows.append(
            {
                "name": case["name"],
                "category": case["category"],
                "correct": breakdown["correct"],
                "format_pass": format_pass,
                "severity": breakdown["severity"],
                "length": breakdown["length"],
                **scores,
            }
        )

    summaries = [summarize_round(round_name, rows) for round_name in ROUND_DESCRIPTIONS]
    round5_final = round5_final_safety_group(rows)
    checks = {
        "round1_exposes_wrong_formatted_positive": next(row for row in rows if row["name"] == "wrong_formatted")["round1"] > 0.0,
        "round2_exposes_wrong_formatted_positive": next(row for row in rows if row["name"] == "wrong_formatted")["round2"] > 0.0,
        "round3_blocks_wrong_formatted_positive": next(row for row in rows if row["name"] == "wrong_formatted")["round3"] <= 0.0,
        "round4_blocks_wrong_formatted_positive": next(row for row in rows if row["name"] == "wrong_formatted")["round4"] <= 0.0,
        "round5_blocks_wrong_formatted_positive": next(row for row in rows if row["name"] == "wrong_formatted")["round5"] <= 0.0,
        "round5_nonpositive_constraint_holds": all(value <= 0.0 for name, value in round5_final.items() if name != "best_honest"),
    }
    output = {
        "scope": "offline reward-design ablation; not GRPO training",
        "label": LABEL,
        "round_descriptions": ROUND_DESCRIPTIONS,
        "case_count": len(rows),
        "round_summaries": summaries,
        "round5_final_safety_group": round5_final,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "round_comparison_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "round_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["all_checks_passed"]:
        raise SystemExit("Round comparison checks failed")


if __name__ == "__main__":
    main()
