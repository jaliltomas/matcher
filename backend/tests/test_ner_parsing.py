from app.services.stages.qwen_enricher import QwenNerEnricherStage


def test_ner_strict_json_fallback_on_malformed_output() -> None:
    stage = QwenNerEnricherStage(
        model_id="dummy",
        device="cpu",
        dtype="float32",
        max_new_tokens=64,
        use_vllm=False,
        vllm_base_url="http://127.0.0.1:8008/v1",
        vllm_api_key="EMPTY",
        vllm_timeout_seconds=5,
        vllm_max_retries=0,
        vllm_disable_thinking=True,
        vllm_max_parallel=1,
        strict_json=True,
    )

    parsed = stage._parse_json("Cerveza Heineken 473 ml", "not-json output")

    assert parsed == {
        "brand": None,
        "category": None,
        "evidence": {"brand": None, "category": None},
    }
    assert stage.parse_fail_count == 1
