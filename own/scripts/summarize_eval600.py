"""Aggregate the nine archived split summaries under one fixed protocol."""
import json
from pathlib import Path


def summarize():
    root = Path(__file__).resolve().parents[2] / "evidence/server_eval_summaries"
    groups = {"base": [], "iter49": [], "iter99": []}
    for path in sorted(root.glob("*/summary.json")):
        name = path.parent.name
        if "save10-iter49-len1536" in name:
            group = "iter49"
        elif "iter99-len1536" in name:
            group = "iter99"
        else:
            group = "base"
        item = json.loads(path.read_text())
        groups[group].append({"file": str(path.relative_to(root)), "correct": item["correct"], "total": item["num_samples"]})
    result = {}
    for name, rows in groups.items():
        total = sum(row["total"] for row in rows)
        correct = sum(row["correct"] for row in rows)
        if len(rows) != 3 or total != 600:
            raise ValueError(f"Incomplete group {name}: {len(rows)} splits / {total} samples")
        result[name] = {"correct": correct, "total": total, "accuracy": correct / total, "sources": rows}
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(), ensure_ascii=False, indent=2))
