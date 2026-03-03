from pydantic import BaseModel


class EvalRow(BaseModel):
    anchor_id: str
    product_id: str
    label: int
    notes: str | None = None
