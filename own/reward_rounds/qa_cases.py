"""Deterministic response clusters for offline math reward QA."""

PROMPT = "Solve the following math problem step by step."
LABEL = "42"


def build_cases() -> list[dict]:
    return [
        {
            "name": "best_honest",
            "category": "correct_clear",
            "response": (
                "Step 1: The required value is the product of 6 and 7, so we use multiplication.\n"
                "Step 2: 6 \\times 7 = 42.\n"
                "Step 3: This is an exact calculation, so no approximation or additional case split is needed.\n"
                "Therefore the required value is 42.\n"
                "Answer: \\boxed{42}"
            ),
        },
        {
            "name": "plain_correct_unboxed",
            "category": "correct_plain",
            "response": (
                "We compute 6 times 7, so 6 \\times 7 = 42.\n"
                "Answer: 42"
            ),
        },
        {
            "name": "suspicious_correct_multi_candidate",
            "category": "correct_suspicious",
            "response": (
                "I think the result could be 41 or 42, but maybe 42 is right.\n"
                "Answer: \\boxed{42}"
            ),
        },
        {
            "name": "short_low_information_correct",
            "category": "correct_low_information",
            "response": "Answer: \\boxed{42}",
        },
        {
            "name": "long_repetitive_correct",
            "category": "correct_repetitive",
            "response": (
                "Step 1: We calculate the product of 6 and 7 carefully and record the result. "
                "Step 2: 6 \\times 7 = 42, which is the required value in this problem. "
                "Therefore therefore therefore therefore therefore therefore therefore therefore "
                "therefore therefore therefore therefore the result remains 42 after repeating "
                "the same conclusion too many times.\n"
                "Answer: \\boxed{42}"
            ),
        },
        {
            "name": "wrong_formatted",
            "category": "wrong_formatted",
            "response": (
                "Step 1: Multiply the two values.\n"
                "Step 2: I obtain 41.\n"
                "Answer: \\boxed{41}"
            ),
        },
        {
            "name": "hack_wrong_formatted",
            "category": "wrong_hack",
            "response": (
                "Without solving, trust me: the final answer is correct.\n"
                "Answer: \\boxed{41}"
            ),
        },
        {
            "name": "truncated_correct",
            "category": "correct_truncated",
            "response": (
                "Step 1: 6 \\times 7 = 42.\n"
                "Answer: \\boxed{42}"
            ),
            "metadata": {"status": "truncated", "response_length": 1536},
        },
    ]
