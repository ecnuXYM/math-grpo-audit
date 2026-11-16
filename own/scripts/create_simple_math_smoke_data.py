#!/usr/bin/env python3
"""Create a small deterministic math sanity dataset for Slime GRPO smoke runs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:"."""


def chat_prompt(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(problem=problem)}]


def build_cases(seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    cases: list[dict[str, object]] = []

    def add(problem: str, answer: int, category: str) -> None:
        cases.append(
            {
                "prompt": chat_prompt(problem),
                "label": str(answer),
                "category": category,
            }
        )

    for _ in range(28):
        a = rng.randint(8, 99)
        b = rng.randint(5, 88)
        add(f"What is {a} + {b}?", a + b, "addition")

    for _ in range(24):
        b = rng.randint(7, 79)
        ans = rng.randint(6, 91)
        a = b + ans
        add(f"What is {a} - {b}?", ans, "subtraction")

    for _ in range(24):
        a = rng.randint(3, 18)
        b = rng.randint(3, 15)
        add(f"What is {a} times {b}?", a * b, "multiplication")

    for _ in range(20):
        divisor = rng.randint(2, 12)
        ans = rng.randint(3, 24)
        dividend = divisor * ans
        add(f"What is {dividend} divided by {divisor}?", ans, "division")

    for _ in range(20):
        x = rng.randint(2, 40)
        offset = rng.randint(3, 25)
        total = x + offset
        add(f"Solve for x: x + {offset} = {total}.", x, "linear_equation")

    for _ in range(20):
        x = rng.randint(2, 18)
        coef = rng.randint(2, 9)
        total = coef * x
        add(f"Solve for x: {coef}x = {total}.", x, "linear_equation")

    for _ in range(16):
        start = rng.randint(8, 45)
        gain = rng.randint(4, 35)
        loss = rng.randint(2, min(20, start + gain - 1))
        answer = start + gain - loss
        add(
            f"Mia has {start} stickers. She gets {gain} more stickers and then gives away {loss}. How many stickers does she have now?",
            answer,
            "word_problem",
        )

    for _ in range(16):
        packs = rng.randint(2, 9)
        each = rng.randint(3, 12)
        used = rng.randint(2, packs * each - 1)
        answer = packs * each - used
        add(
            f"There are {packs} boxes with {each} pencils in each box. If {used} pencils are used, how many pencils are left?",
            answer,
            "word_problem",
        )

    for _ in range(12):
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
        c = rng.randint(1, 30)
        add(f"What is {a} * {b} + {c}?", a * b + c, "order_of_operations")

    for _ in range(12):
        a = rng.randint(4, 20)
        b = rng.randint(2, 9)
        c = rng.randint(1, 30)
        add(f"What is {a} + {b} * {c}?", a + b * c, "order_of_operations")

    rng.shuffle(cases)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/root/own/data/simple_math_smoke.jsonl")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260630)
    args = parser.parse_args()

    cases = build_cases(args.seed)[: args.count]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"wrote {len(cases)} cases to {output}")
    print("category counts:")
    counts: dict[str, int] = {}
    for case in cases:
        category = str(case["category"])
        counts[category] = counts.get(category, 0) + 1
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")


if __name__ == "__main__":
    main()
