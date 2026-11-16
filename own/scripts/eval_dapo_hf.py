#!/usr/bin/env python3
"""Run a lightweight HF generation eval on DAPO math data.

This is intended for model-capability diagnosis before running Slime GRPO:
generate answers, recompute the local round reward, and summarize correctness,
truncation, and prompt-group signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from own.reward_rounds.core import RoundRewardConfig, compute_round_reward


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(rows: list[dict[str, Any]], count: int, offset: int, seed: int | None) -> list[dict[str, Any]]:
    if seed is not None:
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        return [rows[i] for i in indices[:count]]
    return rows[offset : offset + count]


def render_prompt(tokenizer: Any, prompt: Any, enable_thinking: bool) -> str:
    if isinstance(prompt, list):
        try:
            return tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return str(prompt)


def eos_ids(tokenizer: Any, model: Any) -> set[int]:
    ids: set[int] = set()
    for value in [getattr(tokenizer, "eos_token_id", None), getattr(model.config, "eos_token_id", None)]:
        if value is None:
            continue
        if isinstance(value, list):
            ids.update(int(x) for x in value)
        else:
            ids.add(int(value))
    return ids


def generate_one(
    *,
    model: Any,
    tokenizer: Any,
    rendered_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> tuple[str, int, str]:
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)

    input_len = int(inputs["input_ids"].shape[-1])
    generated_ids = output[0, input_len:]
    generated_len = int(generated_ids.numel())
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    last_id = int(generated_ids[-1].item()) if generated_len else None
    status = "truncated" if generated_len >= max_new_tokens and last_id not in eos_ids(tokenizer, model) else "completed"
    return text, generated_len, status


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
        "response_len_mean": mean(lengths) if lengths else 0.0,
        "response_len_max": max(lengths) if lengths else 0,
        "raw_reward_mean": mean(rewards) if rewards else 0.0,
        "raw_reward_min": min(rewards) if rewards else 0.0,
        "raw_reward_max": max(rewards) if rewards else 0.0,
        "positive_wrong_raw": sum(float(row["raw_reward"]) > 0 for row in wrong_rows),
        "zero_wrong_raw": sum(abs(float(row["raw_reward"])) < 1e-12 for row in wrong_rows),
        "groups": {
            "total": len(groups),
            "all_correct": all_correct,
            "mixed": mixed,
            "all_wrong": all_wrong,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", default="/root/dapo-math-17k/dapo-math-17k.jsonl", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--wrong-base-penalty", type=float, default=-0.05)
    parser.add_argument("--torch-seed", type=int, help="seed torch sampling for reproducible fixed eval")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.torch_seed is not None:
        random.seed(args.torch_seed)
        torch.manual_seed(args.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.torch_seed)

    print(f"Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    selected = select_rows(load_jsonl(args.data), args.num_prompts, args.offset, args.seed)
    config = RoundRewardConfig(
        wrong_base_penalty=args.wrong_base_penalty,
        soft_length_cap=min(768, args.max_new_tokens),
        hard_length_cap=args.max_new_tokens,
    )

    rows: list[dict[str, Any]] = []
    jsonl_path = args.output_dir / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for group_index, item in enumerate(selected):
            rendered = render_prompt(tokenizer, item["prompt"], args.enable_thinking)
            for sample_index in range(args.n_samples_per_prompt):
                response, response_len, status = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    rendered_prompt=rendered,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=not args.greedy,
                )
                breakdown = compute_round_reward(
                    prompt=item["prompt"],
                    response=response,
                    label=item["label"],
                    metadata={"response_length": response_len, "status": status},
                    config=config,
                ).to_dict()
                row = {
                    "group_index": group_index,
                    "sample_in_group": sample_index,
                    "label": item["label"],
                    "status": status,
                    "response_length": response_len,
                    "extracted_answer": breakdown["extracted_answer"],
                    "correct": breakdown["correct"],
                    "answer_parse_ok": breakdown["answer_parse_ok"],
                    "severity": breakdown["severity"],
                    "raw_reward": breakdown["raw_reward"],
                    "final_reward": breakdown["final_reward"],
                    "prompt_head": str(item["prompt"])[:240],
                    "response_head": response[:240],
                    "response_tail": response[-240:],
                }
                rows.append(row)
                jsonl.write(
                    json.dumps(
                        {
                            "group_index": group_index,
                            "sample_in_group": sample_index,
                            "prompt": item["prompt"],
                            "label": item["label"],
                            "response": response,
                            "metadata": {
                                "status": status,
                                "response_length": response_len,
                                "reward_breakdown": breakdown,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                print(
                    f"[{len(rows):04d}] group={group_index} sample={sample_index} "
                    f"correct={breakdown['correct']} len={response_len} status={status} "
                    f"extracted={breakdown['extracted_answer']!r} label={item['label']!r}",
                    flush=True,
                )

    write_csv(rows, args.output_dir / "reward_breakdown.csv")
    summary = summarize(rows, len(selected), args.n_samples_per_prompt)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Samples: {jsonl_path}")
    print(f"CSV: {args.output_dir / 'reward_breakdown.csv'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
