#!/usr/bin/env python3
"""Create non-overlapping Hendrycks MATH train/eval JSONL splits."""

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


def write_jsonl(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
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


def print_counts(name: str, df: pd.DataFrame) -> None:
    print(f"{name}: {len(df)} examples")
    counts = df.groupby(["subject", "level"]).size().sort_index()
    for (subject, level), count in counts.items():
        print(f"  {subject} level {level}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--eval-output", required=True, type=Path)
    parser.add_argument("--levels", help="Comma-separated levels, e.g. 3,4")
    parser.add_argument("--subjects", help="Comma-separated subjects matching dataset values")
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--eval-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    levels = parse_int_csv_arg(args.levels)
    subjects = parse_csv_arg(args.subjects)

    if levels is not None:
        df = df[df["level"].isin(levels)]
    if subjects is not None:
        df = df[df["subject"].isin(subjects)]

    required = args.train_size + args.eval_size
    if len(df) < required:
        raise ValueError(f"Need {required} rows after filtering, found {len(df)}")

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    train_df = df.iloc[: args.train_size].copy()
    eval_df = df.iloc[args.train_size : required].copy()

    overlap = set(train_df["unique_id"]).intersection(set(eval_df["unique_id"]))
    if overlap:
        raise RuntimeError(f"Unexpected train/eval overlap: {len(overlap)}")

    write_jsonl(train_df, args.train_output)
    write_jsonl(eval_df, args.eval_output)

    print(f"wrote train to {args.train_output}")
    print_counts("train", train_df)
    print(f"wrote eval to {args.eval_output}")
    print_counts("eval", eval_df)


if __name__ == "__main__":
    main()
