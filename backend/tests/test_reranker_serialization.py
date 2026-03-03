from app.services.stages.reranker import build_pair_text


def test_reranker_serialization_brand_category_match_hints() -> None:
    text_a, text_b = build_pair_text(
        anchor_text="Cerveza Heineken Lata 473 ml",
        candidate_text="Heineken Lager lata 473ml",
        anchor_ner={"brand": "Heineken", "category": "cerveza"},
        candidate_ner={"brand": "Heineken", "category": "cerveza"},
    )

    assert "ANCHOR_BRAND: Heineken" in text_a
    assert "CANDIDATE_CATEGORY: cerveza" in text_b
    assert "BRAND_MATCH: yes" in text_b
    assert "CATEGORY_MATCH: yes" in text_b


def test_reranker_serialization_unknown_and_no_hints() -> None:
    _, text_b = build_pair_text(
        anchor_text="Producto X",
        candidate_text="Producto Y",
        anchor_ner={"brand": None, "category": "limpieza"},
        candidate_ner={"brand": "Acme", "category": "bebida"},
    )

    assert "BRAND_MATCH: unknown" in text_b
    assert "CATEGORY_MATCH: no" in text_b
