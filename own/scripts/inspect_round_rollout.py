#!/usr/bin/env python3
"""Inspect saved slime debug rollout data for round reward breakdowns."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch


DEFAULT_INPUT = Path("/root/own/debug_rollouts/qwen2.5-1.5B-round-smoke/rollout_0.pt")
DEFAULT_OUTPUT_ROOT = Path("/root/own/reports")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from own.reward_rounds.core import RoundRewardConfig, compute_round_reward


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "rollout"


def sample_get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def flatten_sample(index: int, sample: Any, recompute_config: RoundRewardConfig | None = None) -> dict[str, Any]:
    metadata = sample_get(sample, "metadata", {}) or {}
    if recompute_config is None:
        breakdown = metadata.get("reward_breakdown", {}) or {}
    else:
        reward_metadata = dict(metadata)
        reward_metadata["response_length"] = sample_get(sample, "response_length")
        reward_metadata["status"] = sample_get(sample, "status")
        breakdown = compute_round_reward(
            prompt=sample_get(sample, "prompt"),
            response=sample_get(sample, "response"),
            label=sample_get(sample, "label"),
            metadata=reward_metadata,
            config=recompute_config,
        ).to_dict()
    features = breakdown.get("features", {}) or {}

    row: dict[str, Any] = {
        "sample_index": index,
        "group_index": sample_get(sample, "group_index"),
        "index": sample_get(sample, "index"),
        "label": sample_get(sample, "label"),
        "status": sample_get(sample, "status"),
        "response_length": sample_get(sample, "response_length"),
        "reward": sample_get(sample, "reward"),
        "round_name": breakdown.get("round_name"),
        "extracted_answer": breakdown.get("extracted_answer"),
        "correct": breakdown.get("correct"),
        "answer_parse_ok": breakdown.get("answer_parse_ok"),
        "severity_raw": breakdown.get("severity_raw"),
        "severity": breakdown.get("severity"),
        "gated_accuracy": breakdown.get("gated_accuracy"),
        "format": breakdown.get("format"),
        "anti_hack": breakdown.get("anti_hack"),
        "length": breakdown.get("length"),
        "raw_reward": breakdown.get("raw_reward", metadata.get("raw_reward")),
        "final_reward": breakdown.get("final_reward", sample_get(sample, "reward")),
        "prompt_head": str(sample_get(sample, "prompt", ""))[:240],
        "response_head": str(sample_get(sample, "response", ""))[:240],
        "response_tail": str(sample_get(sample, "response", ""))[-240:],
    }

    for key, value in sorted(features.items()):
        row[f"feature_{key}"] = value
    return row


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_samples": len(rows)}
    for key in [
        "reward",
        "raw_reward",
        "final_reward",
        "severity",
        "gated_accuracy",
        "format",
        "anti_hack",
        "length",
        "response_length",
    ]:
        values = numeric_values(rows, key)
        if values:
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)

    correct_values = [row.get("correct") for row in rows if row.get("correct") is not None]
    if correct_values:
        summary["correct_rate"] = sum(bool(x) for x in correct_values) / len(correct_values)

    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    summary["statuses"] = statuses
    return summary


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


def maybe_plot(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []

    def save_current(name: str) -> None:
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    raw = numeric_values(rows, "raw_reward")
    final = numeric_values(rows, "final_reward")
    reward = numeric_values(rows, "reward")
    if raw or final or reward:
        plt.figure(figsize=(8, 4.5))
        if raw:
            plt.hist(raw, bins=min(10, max(3, len(raw))), alpha=0.55, label="raw_reward")
        if final:
            plt.hist(final, bins=min(10, max(3, len(final))), alpha=0.55, label="final_reward")
        if reward and reward != final:
            plt.hist(reward, bins=min(10, max(3, len(reward))), alpha=0.45, label="sample.reward")
        plt.xlabel("reward")
        plt.ylabel("count")
        plt.legend()
        plt.title("Reward Distribution")
        save_current("reward_distribution.png")

    severity = numeric_values(rows, "severity")
    if severity:
        plt.figure(figsize=(8, 4.5))
        plt.hist(severity, bins=min(10, max(3, len(severity))), color="#5B8FF9", alpha=0.8)
        plt.xlabel("severity")
        plt.ylabel("count")
        plt.title("Severity Distribution")
        save_current("severity_distribution.png")

    if severity and raw:
        points = [
            (float(row["severity"]), float(row["raw_reward"]))
            for row in rows
            if isinstance(row.get("severity"), (int, float))
            and isinstance(row.get("raw_reward"), (int, float))
        ]
        if points:
            xs, ys = zip(*points)
            plt.figure(figsize=(8, 4.5))
            plt.scatter(xs, ys, color="#E8684A")
            plt.xlabel("severity")
            plt.ylabel("raw_reward")
            plt.title("Severity vs Raw Reward")
            save_current("severity_vs_raw_reward.png")

    component_keys = ["gated_accuracy", "format", "anti_hack", "length"]
    component_means = []
    for key in component_keys:
        values = numeric_values(rows, key)
        component_means.append(mean(values) if values else 0.0)
    if any(value != 0.0 for value in component_means):
        plt.figure(figsize=(8, 4.5))
        plt.bar(component_keys, component_means, color=["#5B8FF9", "#61DDAA", "#F6BD16", "#7262FD"])
        plt.ylabel("mean value")
        plt.title("Reward Component Means")
        save_current("component_means.png")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--recompute", action="store_true", help="recompute rewards with current code instead of saved metadata")
    parser.add_argument("--wrong-base-penalty", type=float, default=0.0)
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    recompute_config = RoundRewardConfig(wrong_base_penalty=args.wrong_base_penalty) if args.recompute else None
    rows = [flatten_sample(i, sample, recompute_config) for i, sample in enumerate(samples)]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / safe_name(args.input.parent.name) / safe_name(args.input.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "reward_breakdown.csv"
    summary_path = output_dir / "summary.json"
    write_csv(rows, csv_path)

    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = maybe_plot(rows, output_dir)

    print(f"Loaded samples: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    for path in plot_paths:
        print(f"Plot: {path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
