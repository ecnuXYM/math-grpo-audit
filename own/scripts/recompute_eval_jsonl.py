#!/usr/bin/env python3
"""Recompute reward metadata for eval_dapo_hf.py samples.jsonl outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from own.reward_rounds.core import RoundRewardConfig, compute_round_reward


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], prompts: int, n_samples_per_prompt: int) -> dict[str, Any]:
    rewards = [float(row["raw_reward"]) for row in rows]
    lengths = [int(row["response_length"]) for row in rows]
    correct = [truth(row["correct"]) for row in rows]
    truncated = [row["status"] == "truncated" for row in rows]
    wrong_rows = [row for row in rows if not truth(row["correct"])]

    groups: dict[int, list[bool]] = {}
    for row, is_correct in zip(rows, correct):
        groups.setdefault(int(row["group_index"]), []).append(is_correct)

    all_correct = sum(all(values) for values in groups.values())
    all_wrong = sum(not any(values) for values in groups.values())
    mixed = sum(any(values) and not all(values) for values in groups.values())

    return {
        "num_prompts": prompts,
        "n_samples_per_prompt": n_samples_per_prompt,
        "num_samples": len(rows),
        "correct": sum(correct),
        "correct_rate": sum(correct) / len(rows) if rows else 0.0,
        "truncated": sum(truncated),
        "truncated_rate": sum(truncated) / len(rows) if rows else 0.0,
        "response_len_mean": sum(lengths) / len(lengths) if lengths else 0.0,
        "response_len_max": max(lengths) if lengths else 0,
        "raw_reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "raw_reward_min": min(rewards) if rewards else 0.0,
        "raw_reward_max": max(rewards) if rewards else 0.0,
        "positive_wrong_raw": sum(float(row["raw_reward"]) > 0 for row in wrong_rows),
        "zero_wrong_raw": sum(float(row["raw_reward"]) == 0 for row in wrong_rows),
        "groups": {
            "total": len(groups),
            "all_correct": all_correct,
            "mixed": mixed,
            "all_wrong": all_wrong,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_jsonl", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--wrong-base-penalty", type=float, default=-0.05)
    args = parser.parse_args()

    config = RoundRewardConfig(wrong_base_penalty=args.wrong_base_penalty)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    output_samples = args.output_dir / "samples.jsonl"
    with args.samples_jsonl.open(encoding="utf-8") as src, output_samples.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            item = json.loads(line)
            metadata = dict(item.get("metadata") or {})
            reward_metadata = {
                "response_length": metadata.get("response_length"),
                "status": metadata.get("status"),
            }
            result = compute_round_reward(
                prompt=item.get("prompt"),
                response=item.get("response", ""),
                label=item.get("label"),
                metadata=reward_metadata,
                config=config,
            )
            breakdown = result.to_dict()
            metadata["reward_breakdown"] = breakdown
            item["metadata"] = metadata

            row = {
                "group_index": item.get("group_index"),
                "sample_in_group": item.get("sample_in_group"),
                "label": item.get("label"),
                "status": metadata.get("status"),
                "response_length": metadata.get("response_length"),
                "extracted_answer": breakdown["extracted_answer"],
                "correct": breakdown["correct"],
                "answer_parse_ok": breakdown["answer_parse_ok"],
                "severity": breakdown["severity"],
                "raw_reward": breakdown["raw_reward"],
                "final_reward": breakdown["final_reward"],
                "prompt_head": str(item.get("prompt"))[:240],
                "response_head": item.get("response", "")[:240],
                "response_tail": item.get("response", "")[-240:],
            }
            rows.append(row)
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")

    prompts = len({int(row["group_index"]) for row in rows})
    per_prompt_counts = {}
    for row in rows:
        per_prompt_counts.setdefault(int(row["group_index"]), 0)
        per_prompt_counts[int(row["group_index"])] += 1
    n_samples_per_prompt = max(per_prompt_counts.values()) if per_prompt_counts else 0

    write_csv(rows, args.output_dir / "reward_breakdown.csv")
    summary = summarize(rows, prompts, n_samples_per_prompt)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Samples: {output_samples}")
    print(f"CSV: {args.output_dir / 'reward_breakdown.csv'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
