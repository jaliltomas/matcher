# TurboMatcher - Matching de productos end-to-end

Proyecto completo con:

- `frontend/`: Vue 3 minimalista para subir JSONs y visualizar top matches.
- `backend/`: FastAPI + Milvus + pipeline modular de matching multimodal.

## Arquitectura

Pipeline del backend:

1. **Embeddings multimodales BLIP-2** para anclas y productos.
2. **Persistencia vectorial en Milvus** y busqueda top-N por ancla.
3. **Enriquecedor NER con Qwen-7B** (marca, modelo, categoria, atributos).
4. **Reranking cross-encoder con XLM-RoBERTa-large**.
5. **Validator con Qwen-7B** para score y flag de revision.

Cada etapa vive en una clase separada (`backend/app/services/stages/`) para reemplazo rapido.

## 1) Levantar Milvus

```bash
cd backend
docker compose -f docker-compose.milvus.yml up -d
```

## Scripts PowerShell (recomendado en Windows)

- Instalacion completa (backend + frontend):

```powershell
.\scripts\install.ps1
```

- Instalacion con PyTorch CUDA 12.1 explicito:

```powershell
.\scripts\install.ps1 -InstallTorchCuda
```

- Arranque diario (Milvus + backend + frontend):

```powershell
.\scripts\start.ps1
```

Opciones utiles:

- `-NoMilvus` para no levantar docker/milvus.
- `-NoBrowser` para no abrir el navegador automaticamente.

## 2) Backend (FastAPI)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Si necesitas CUDA para PyTorch, instala la rueda compatible con tu version de CUDA desde el indice oficial de PyTorch.

## 3) Frontend (Vue 3)

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Abre `http://localhost:5173`.

## 4) Prueba rapida

Usa los archivos de ejemplo dentro de `samples/`:

- `samples/anchors.json`
- `samples/precios_tienda_1.json`
- `samples/precios_tienda_2.json`

En la UI:

1. Sube `anchors.json` como JSON ancla.
2. Sube ambos `precios_*.json` como JSONs de precios.
3. Click en **Procesar matching**.

## Notas de rendimiento para RTX 3090

- El backend usa `float16` por defecto (`DTYPE=float16` en `.env`).
- `BATCH_SIZE` configurable por entorno o request `/match`.
- `IMAGE_DOWNLOAD_WORKERS` acelera carga de imagenes remotas en embeddings.
- `IMAGE_TIMEOUT_SECONDS` corta rapido URLs lentas/caidas.
- `OFFLOAD_BETWEEN_STAGES=true` ayuda a liberar VRAM entre modelos grandes.
- `use_resume=true` reutiliza etapas ya calculadas por `session_id` (embeddings, search, NER, reranker, validator).
- Endpoint `GET /prompt-presets` expone prompts guardados para extraccion y validacion.
- Se devuelven metricas por etapa: tiempo, VRAM pico y volumen de candidatos.
