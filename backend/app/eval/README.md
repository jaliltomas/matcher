# Evaluation Harness (Skeleton)

This directory contains a minimal offline evaluation skeleton.

## Dataset schema

JSONL rows must follow:

```json
{"anchor_id":"a_1","product_id":"p_10","label":1,"notes":"same sku"}
```

Fields:
- `anchor_id`: anchor identifier
- `product_id`: candidate product identifier
- `label`: `1` for match, `0` for non-match
- `notes`: optional annotation

## Run

```bash
python -m app.eval.eval_runner --dataset path/to/eval.jsonl --k 5
```

Outputs:
- Precision@1
- Recall@K
- MRR@K

This is intentionally lightweight and can be extended with real pipeline scoring and richer metrics.
