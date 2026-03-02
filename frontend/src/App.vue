<script setup>
import axios from "axios";
import { computed, onMounted, ref } from "vue";

const apiBase = import.meta.env.VITE_API_URL || "http://localhost:8000";

const anchorFile = ref(null);
const priceFiles = ref([]);
const topN = ref(30);
const batchSize = ref(24);
const nerBatchSize = ref(16);
const validatorBatchSize = ref(8);
const useResume = ref(true);
const reuseSession = ref(true);
const manualSessionId = ref("");

const extractionPromptId = ref("");
const validationPromptId = ref("");
const extractionPromptText = ref("");
const validationPromptText = ref("");
const promptPresets = ref({ extraction: [], validation: [], defaults: { extraction: "", validation: "" } });

const loading = ref(false);
const errorMessage = ref("");
const uploadInfo = ref(null);
const resultData = ref(null);

const hasFiles = computed(() => anchorFile.value && priceFiles.value.length > 0);
const canProcess = computed(() => {
  if (loading.value) return false;
  if (reuseSession.value && (uploadInfo.value?.session_id || manualSessionId.value.trim())) return true;
  return !!hasFiles.value;
});

function onAnchorSelected(event) {
  anchorFile.value = event.target.files?.[0] || null;
}

function onPricesSelected(event) {
  priceFiles.value = Array.from(event.target.files || []);
}

function formatScore(value) {
  if (typeof value !== "number") return "-";
  return value.toFixed(3);
}

function findPrompt(kind, id) {
  const items = promptPresets.value[kind] || [];
  return items.find((item) => item.id === id) || null;
}

const extractionPreview = computed(() => {
  if (extractionPromptText.value.trim()) return extractionPromptText.value.trim();
  return findPrompt("extraction", extractionPromptId.value)?.template || "";
});

const validationPreview = computed(() => {
  if (validationPromptText.value.trim()) return validationPromptText.value.trim();
  return findPrompt("validation", validationPromptId.value)?.template || "";
});

async function loadPromptPresets() {
  try {
    const response = await axios.get(`${apiBase}/prompt-presets`);
    promptPresets.value = response.data;
    extractionPromptId.value = response.data.defaults.extraction;
    validationPromptId.value = response.data.defaults.validation;
  } catch {
    extractionPromptId.value = "";
    validationPromptId.value = "";
  }
}

async function processMatching() {
  if (!canProcess.value) return;

  loading.value = true;
  errorMessage.value = "";
  resultData.value = null;

  try {
    let sessionId = null;
    if (reuseSession.value) {
      sessionId = manualSessionId.value.trim() || uploadInfo.value?.session_id || null;
    }

    if (!sessionId) {
      const formData = new FormData();
      formData.append("anchor_file", anchorFile.value);
      for (const file of priceFiles.value) {
        formData.append("price_files", file);
      }

      const uploadResponse = await axios.post(`${apiBase}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      uploadInfo.value = uploadResponse.data;
      sessionId = uploadResponse.data.session_id;
    }

    const matchResponse = await axios.post(`${apiBase}/match`, {
      session_id: sessionId,
      top_n: Number(topN.value) || 30,
      top_k: 5,
      batch_size: Number(batchSize.value) || 24,
      ner_batch_size: Number(nerBatchSize.value) || 16,
      validator_batch_size: Number(validatorBatchSize.value) || 8,
      use_resume: !!useResume.value,
      extraction_prompt_id: extractionPromptId.value || null,
      validation_prompt_id: validationPromptId.value || null,
      extraction_prompt_text: extractionPromptText.value.trim() || null,
      validation_prompt_text: validationPromptText.value.trim() || null
    });

    resultData.value = matchResponse.data;
  } catch (error) {
    errorMessage.value =
      error?.response?.data?.detail || error?.message || "Error al procesar matching";
  } finally {
    loading.value = false;
  }
}

onMounted(loadPromptPresets);
</script>

<template>
  <main class="page">
    <section class="panel">
      <h1>TurboMatcher</h1>
      <p class="subtitle">
        Matching multimodal de productos (BLIP-2 + Milvus + Qwen + XLM-R)
      </p>

      <div class="upload-grid">
        <label class="field">
          <span>JSON ancla</span>
          <input type="file" accept="application/json" @change="onAnchorSelected" />
        </label>

        <label class="field">
          <span>JSONs de precios (multiples)</span>
          <input type="file" multiple accept="application/json" @change="onPricesSelected" />
        </label>

        <label class="field small">
          <span>Top N candidatos</span>
          <input v-model.number="topN" type="number" min="5" max="200" />
        </label>

        <label class="field small">
          <span>Batch embeddings</span>
          <input v-model.number="batchSize" type="number" min="1" max="256" />
        </label>

        <label class="field small">
          <span>Batch extractor NER</span>
          <input v-model.number="nerBatchSize" type="number" min="1" max="128" />
        </label>

        <label class="field small">
          <span>Batch validator</span>
          <input v-model.number="validatorBatchSize" type="number" min="1" max="128" />
        </label>
      </div>

      <label class="resume-toggle">
        <input v-model="reuseSession" type="checkbox" />
        <span>Reusar session cargada (evita re-upload)</span>
      </label>

      <label v-if="reuseSession" class="field">
        <span>Session ID para resumir (opcional)</span>
        <input
          v-model="manualSessionId"
          type="text"
          placeholder="Pega una session_id existente para reanudar"
        />
      </label>

      <label class="resume-toggle">
        <input v-model="useResume" type="checkbox" />
        <span>Usar resume/cache por session para no recalcular etapas ya procesadas</span>
      </label>

      <div class="prompt-columns">
        <section class="prompt-card">
          <h3>Prompts de extraccion</h3>
          <label class="field">
            <span>Preset</span>
            <select v-model="extractionPromptId">
              <option v-for="item in promptPresets.extraction" :key="item.id" :value="item.id">
                {{ item.name }}
              </option>
            </select>
          </label>
          <label class="field">
            <span>Prompt custom (opcional)</span>
            <textarea v-model="extractionPromptText" rows="5" placeholder="Si completas esto, pisa el preset."></textarea>
          </label>
          <p class="attrs">Preview: {{ extractionPreview }}</p>
        </section>

        <section class="prompt-card">
          <h3>Prompts de validacion</h3>
          <label class="field">
            <span>Preset</span>
            <select v-model="validationPromptId">
              <option v-for="item in promptPresets.validation" :key="item.id" :value="item.id">
                {{ item.name }}
              </option>
            </select>
          </label>
          <label class="field">
            <span>Prompt custom (opcional)</span>
            <textarea v-model="validationPromptText" rows="5" placeholder="Si completas esto, pisa el preset."></textarea>
          </label>
          <p class="attrs">Preview: {{ validationPreview }}</p>
        </section>
      </div>

      <button class="primary" :disabled="!canProcess" @click="processMatching">
        {{ loading ? "Procesando..." : "Procesar matching" }}
      </button>

      <p v-if="errorMessage" class="error">{{ errorMessage }}</p>

      <div v-if="uploadInfo" class="upload-info">
        <strong>Session:</strong> {{ uploadInfo.session_id }}<br />
        <strong>Anclas:</strong> {{ uploadInfo.anchors_count }} | <strong>Productos:</strong>
        {{ uploadInfo.products_count }}
      </div>
    </section>

    <section v-if="resultData" class="panel">
      <h2>Resultados top 5 por ancla</h2>
      <p class="attrs">
        Resume: {{ resultData.use_resume ? "ON" : "OFF" }} | Prompt extraccion:
        {{ resultData.extraction_prompt_id }} | Prompt validacion: {{ resultData.validation_prompt_id }}
      </p>

      <div class="metrics">
        <h3>Metricas de pipeline</h3>
        <table>
          <thead>
            <tr>
              <th>Etapa</th>
              <th>Tiempo (s)</th>
              <th>VRAM MB</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="metric in resultData.metrics" :key="metric.stage">
              <td>
                {{ metric.stage }}
                <span v-if="metric.details?.cache_hit" class="cache-hit">(cache)</span>
              </td>
              <td>{{ metric.seconds.toFixed(2) }}</td>
              <td>{{ metric.vram_mb.toFixed(1) }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <article v-for="anchor in resultData.results" :key="anchor.anchor_id" class="anchor-card">
        <header>
          <h3>{{ anchor.anchor_nombre }}</h3>
          <p class="attrs">Atributos: {{ JSON.stringify(anchor.atributos_anchor) }}</p>
        </header>

        <div class="matches">
          <div v-for="(match, idx) in anchor.matches" :key="`${anchor.anchor_id}-${match.url || idx}`" class="match-row">
            <img v-if="match.img" :src="match.img" alt="img" />
            <div class="info">
              <a :href="match.url" target="_blank" rel="noreferrer">{{ match.nombre }}</a>
              <p>${{ match.precio }} | {{ match.sitio || "sin sitio" }}</p>
              <p>
                Sim: {{ formatScore(match.score_similitud) }} | Rerank:
                {{ formatScore(match.score_reranker) }} | Val:
                {{ formatScore(match.score_validacion) }}
              </p>
              <p class="attrs">Atributos: {{ JSON.stringify(match.atributos) }}</p>
            </div>
            <span :class="['tag', match.revisar ? 'warn' : 'ok']">
              {{ match.revisar ? "Revisar" : "Confiable" }}
            </span>
          </div>
        </div>
      </article>
    </section>
  </main>
</template>
