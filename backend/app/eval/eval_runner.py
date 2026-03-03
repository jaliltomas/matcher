import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.eval.eval_dataset_schema import EvalRow


def load_jsonl(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(EvalRow.model_validate(json.loads(line)))
    return rows


def precision_at_1(grouped: dict[str, list[EvalRow]]) -> float:
    if not grouped:
        return 0.0
    hits = 0
    for rows in grouped.values():
        ranked = sorted(rows, key=lambda row: row.label, reverse=True)
        hits += int(ranked[0].label > 0)
    return hits / len(grouped)


def recall_at_k(grouped: dict[str, list[EvalRow]], k: int) -> float:
    if not grouped:
        return 0.0
    recalls: list[float] = []
    for rows in grouped.values():
        positives = sum(1 for row in rows if row.label > 0)
        if positives == 0:
            continue
        ranked = sorted(rows, key=lambda row: row.label, reverse=True)[:k]
        hit = sum(1 for row in ranked if row.label > 0)
        recalls.append(hit / positives)
    return sum(recalls) / max(1, len(recalls))


def mrr_at_k(grouped: dict[str, list[EvalRow]], k: int) -> float:
    if not grouped:
        return 0.0
    rr_total = 0.0
    for rows in grouped.values():
        ranked = sorted(rows, key=lambda row: row.label, reverse=True)[:k]
        rr = 0.0
        for idx, row in enumerate(ranked, start=1):
            if row.label > 0:
                rr = 1.0 / idx
                break
        rr_total += rr
    return rr_total / len(grouped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval runner skeleton")
    parser.add_argument("--dataset", required=True, help="Path to JSONL evaluation dataset")
    parser.add_argument("--k", type=int, default=5, help="K for Recall@K and MRR@K")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    grouped: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        grouped[row.anchor_id].append(row)

    print(f"Rows: {len(rows)}")
    print(f"Anchors: {len(grouped)}")
    print(f"Precision@1: {precision_at_1(grouped):.4f}")
    print(f"Recall@{args.k}: {recall_at_k(grouped, args.k):.4f}")
    print(f"MRR@{args.k}: {mrr_at_k(grouped, args.k):.4f}")


if __name__ == "__main__":
    main()
