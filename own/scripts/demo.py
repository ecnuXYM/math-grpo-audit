"""Small deterministic reward demonstration; no model execution."""
import json
from own.reward_rounds.core import RoundRewardConfig, compute_round_reward
from own.reward_rounds.qa_cases import LABEL, PROMPT, build_cases


def main():
    config = RoundRewardConfig(wrong_base_penalty=-0.05)
    rows = []
    for case in build_cases():
        score = compute_round_reward(PROMPT, case["response"], LABEL, metadata=case.get("metadata"), config=config)
        rows.append({"case": case["name"], "correct": score.correct, "severity": score.severity, "raw_reward": score.raw_reward})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
