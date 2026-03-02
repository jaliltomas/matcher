# Backend TurboMatcher (FastAPI + Milvus + LLM pipeline)

API para matching de productos con pipeline modular:

1. BLIP-2 embeddings multimodales (texto + imagen)
2. Almacenamiento y busqueda vectorial en Milvus
3. Enriquecimiento NER con Qwen-7B
4. Reranking cross-encoder con XLM-RoBERTa-large
5. Validacion final con Qwen-7B

Clases reemplazables por etapa:

- `app/services/stages/blip2_embedder.py`
- `app/services/stages/qwen_enricher.py`
- `app/services/stages/reranker.py`
- `app/services/stages/validator.py`

`MatchingPipeline` acepta implementaciones custom de cada interfaz definida en `app/services/stages/base.py`.

## Requisitos

- Python 3.10+
- NVIDIA RTX 3090 (24 GB VRAM sugerida)
- Milvus en `localhost:19530`

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Nota: para `torch` con CUDA 12.1 se recomienda instalar desde el indice oficial de PyTorch.

## Ejecutar

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

## Endpoints

- `GET /prompt-presets`
  - devuelve presets guardados para extraccion y validacion.
- `POST /upload`
  - multipart form-data:
    - `anchor_file`: archivo JSON ancla
    - `price_files`: uno o mas JSON de sitios
- `POST /match`
  - body:
  ```json
  {
    "session_id": "...",
    "top_n": 30,
    "top_k": 5,
    "batch_size": 24,
    "ner_batch_size": 16,
    "validator_batch_size": 8,
    "use_resume": true,
    "extraction_prompt_id": "retail_general_v1",
    "validation_prompt_id": "strict_identity_v1"
  }
  ```

`use_resume=true` reutiliza cache por `session_id` para no recalcular etapas ya completadas.

Optimizaciones adicionales:

- Validator usa reglas rapidas para casos obvios y llama al LLM solo en casos ambiguos.
- Parsing JSON robusto en NER/validator para evitar respuestas en 0 por salida malformada.
- Tokenizacion tipo chat y `padding_side=left` para Qwen.
