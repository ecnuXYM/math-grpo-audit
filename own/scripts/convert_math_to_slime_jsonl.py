#!/usr/bin/env python3
"""Convert filtered Hendrycks MATH parquet data to prompt/label JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:"."""


def parse_csv_arg(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_int_csv_arg(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def chat_prompt(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(problem=problem.strip())}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--levels", help="Comma-separated levels, e.g. 2,3")
    parser.add_argument("--subjects", help="Comma-separated subjects matching dataset values")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    levels = parse_int_csv_arg(args.levels)
    subjects = parse_csv_arg(args.subjects)

    if levels is not None:
        df = df[df["level"].isin(levels)]
    if subjects is not None:
        df = df[df["subject"].isin(subjects)]
    if args.seed is not None:
        df = df.sample(frac=1.0, random_state=args.seed)
    if args.limit is not None:
        df = df.head(args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = df.groupby(["subject", "level"]).size().sort_index()
    with args.output.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = {
                "prompt": chat_prompt(str(row["problem"])),
                "label": str(row["answer"]).strip(),
                "source": "hendrycks_math",
                "subject": str(row["subject"]),
                "level": int(row["level"]),
                "unique_id": str(row["unique_id"]),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"wrote {len(df)} examples to {args.output}")
    print("counts:")
    for (subject, level), count in counts.items():
        print(f"  {subject} level {level}: {count}")


if __name__ == "__main__":
    main()
