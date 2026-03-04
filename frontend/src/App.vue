<script setup>
import axios from "axios";
import { computed, onMounted, ref } from "vue";

const apiBase = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

const anchorFile = ref(null);
const priceFiles = ref([]);
const topN = ref(30);
const topK = ref(5);
const batchSize = ref(24);
const nerBatchSize = ref(16);
const validatorBatchSize = ref(8);
const useResume = ref(true);
const useFastRules = ref(true);
const reuseSession = ref(true);
const manualSessionId = ref("");

const extractionPromptId = ref("");
const validationPromptId = ref("");
const extractionPromptText = ref("");
const validationPromptText = ref("");
const promptPresets = ref({ extraction: [], validation: [], defaults: { extraction: "", validation: "" } });
const promptGenVertical = ref("");
const promptGenLanguage = ref("es");
const promptGenBrandNotes = ref("");
const promptGenTaxonomy = ref("");
const promptGenEdgeCases = ref("");
const promptGenLoading = ref(false);
const promptGenError = ref("");
const promptGenMeta = ref(null);

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
const efficiencyStats = computed(() => resultData.value?.efficiency_stats || null);

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

function formatStat(value, digits = 2) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function formatPercent(value, digits = 1) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${value.toFixed(digits)}%`;
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
      top_k: Number(topK.value) || 5,
      batch_size: Number(batchSize.value) || 24,
      ner_batch_size: Number(nerBatchSize.value) || 16,
      validator_batch_size: Number(validatorBatchSize.value) || 8,
      use_resume: !!useResume.value,
      use_fast_rules: !!useFastRules.value,
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

function parseCsvLines(value) {
  return String(value || "")
    .split(/\r?\n|,/)
    .map((item) => item.trim())
    .filter(Boolean);
}

async function generatePromptsForClient() {
  if (!promptGenVertical.value.trim()) {
    promptGenError.value = "Completa la descripcion del rubro para generar prompts.";
    return;
  }

  promptGenLoading.value = true;
  promptGenError.value = "";
  promptGenMeta.value = null;

  try {
    const response = await axios.post(`${apiBase}/prompts/generate`, {
      vertical_description: promptGenVertical.value.trim(),
      language: promptGenLanguage.value,
      brand_notes: promptGenBrandNotes.value.trim() || null,
      category_taxonomy: parseCsvLines(promptGenTaxonomy.value),
      edge_cases: parseCsvLines(promptGenEdgeCases.value),
      output_format: "json_only"
    });

    extractionPromptText.value = response.data.ner_prompt || "";
    validationPromptText.value = response.data.validator_prompt || "";
    promptGenMeta.value = response.data.meta || null;
    await loadPromptPresets();
  } catch (error) {
    promptGenError.value =
      error?.response?.data?.detail || error?.message || "No se pudieron generar prompts";
  } finally {
    promptGenLoading.value = false;
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
          <span>Top K final</span>
          <input v-model.number="topK" type="number" min="1" max="20" />
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

      <label class="resume-toggle">
        <input v-model="useFastRules" type="checkbox" />
        <span>Usar fast rules antes del validator LLM</span>
      </label>

      <section class="prompt-generator">
        <h3>Generador de prompts por rubro</h3>
        <p class="attrs">
          Describe el cliente y genera prompts de NER + Validator. Al generarlos, se cargan automaticamente en
          "Prompt custom".
        </p>
        <div class="upload-grid">
          <label class="field">
            <span>Descripcion del rubro</span>
            <textarea
              v-model="promptGenVertical"
              rows="3"
              placeholder="Ej: farmacia, OTC, suplementos deportivos, marcas con alias..."
            ></textarea>
          </label>

          <label class="field small">
            <span>Idioma</span>
            <select v-model="promptGenLanguage">
              <option value="es">es</option>
              <option value="en">en</option>
            </select>
          </label>

          <label class="field">
            <span>Brand notes (opcional)</span>
            <textarea
              v-model="promptGenBrandNotes"
              rows="2"
              placeholder="Patrones de marcas, alias, abreviaciones..."
            ></textarea>
          </label>

          <label class="field">
            <span>Taxonomia de categorias (opcional)</span>
            <textarea
              v-model="promptGenTaxonomy"
              rows="2"
              placeholder="Una por linea o separadas por coma"
            ></textarea>
          </label>

          <label class="field">
            <span>Edge cases (opcional)</span>
            <textarea
              v-model="promptGenEdgeCases"
              rows="2"
              placeholder="packs, sabores, bundles, variantes, etc."
            ></textarea>
          </label>
        </div>
        <button class="primary" :disabled="promptGenLoading" @click="generatePromptsForClient">
          {{ promptGenLoading ? "Generando prompts..." : "Generar prompts" }}
        </button>
        <p v-if="promptGenError" class="error">{{ promptGenError }}</p>
        <div v-if="promptGenMeta" class="prompt-meta attrs">
          <p>
            <strong>Thresholds sugeridos:</strong>
            accept={{ promptGenMeta.recommended_thresholds?.accept }} |
            reject={{ promptGenMeta.recommended_thresholds?.reject }}
          </p>
          <p v-if="promptGenMeta.assumptions?.length">
            <strong>Assumptions:</strong> {{ promptGenMeta.assumptions.join(" | ") }}
          </p>
          <p v-if="promptGenMeta.notes?.length">
            <strong>Notes:</strong> {{ promptGenMeta.notes.join(" | ") }}
          </p>
        </div>
      </section>

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
      <h2>Resultados top {{ resultData.top_k }} por ancla</h2>
      <p class="attrs">
        Resume: {{ resultData.use_resume ? "ON" : "OFF" }} | Prompt extraccion:
        {{ resultData.extraction_prompt_id }} | Prompt validacion: {{ resultData.validation_prompt_id }} | Fast rules:
        {{ resultData.use_fast_rules === false ? "OFF" : "ON" }}
      </p>

      <div class="metrics">
        <h3>Metricas de pipeline</h3>
        <div v-if="efficiencyStats" class="efficiency-grid">
          <p>
            <strong>Wall time:</strong> {{ formatStat(efficiencyStats.wall_seconds, 2) }}s |
            <strong>Tiempo medido en etapas:</strong> {{ formatStat(efficiencyStats.stage_seconds, 2) }}s
          </p>
          <p>
            <strong>Etapas desde cache:</strong> {{ efficiencyStats.stages_cached }}/{{ efficiencyStats.stages_total }}
            ({{ formatPercent(efficiencyStats.stage_cache_hit_rate_pct) }}) |
            <strong>Etapa dominante:</strong> {{ efficiencyStats.dominant_stage || "-" }}
          </p>
          <p>
            <strong>Poda top_n -> top_k:</strong> {{ efficiencyStats.vector_candidates }} ->
            {{ efficiencyStats.shortlisted_candidates }} ({{ formatPercent(efficiencyStats.rerank_pruning_pct) }})
          </p>
          <p>
            <strong>Validator sin LLM:</strong> {{ efficiencyStats.validator_fast_resolved }}/
            {{ efficiencyStats.validator_candidates }} ({{ formatPercent(efficiencyStats.validator_llm_avoidance_pct) }})
          </p>
        </div>
        <table>
          <thead>
            <tr>
              <th>Etapa</th>
              <th>Tiempo (s)</th>
              <th>VRAM MB</th>
              <th>Carga</th>
              <th>ms/unidad</th>
              <th>unid/s</th>
              <th>% tiempo</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(metric, idx) in resultData.metrics" :key="`${metric.stage}-${idx}`">
              <td>
                {{ metric.stage }}
                <span v-if="metric.details?.cache_hit" class="cache-hit">(cache)</span>
              </td>
              <td>{{ metric.seconds.toFixed(2) }}</td>
              <td>{{ metric.vram_mb.toFixed(1) }}</td>
              <td>
                {{ metric.details?.work_units ?? "-" }}
                <span v-if="metric.details?.work_unit_label">{{ metric.details.work_unit_label }}</span>
              </td>
              <td>{{ formatStat(metric.details?.ms_per_unit, 2) }}</td>
              <td>{{ formatStat(metric.details?.throughput_per_sec, 2) }}</td>
              <td>{{ formatPercent(metric.details?.time_share_pct) }}</td>
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
              <p v-if="match.decision_validacion || match.razon_validacion" class="attrs">
                {{ match.decision_validacion || "decision" }}
                <span v-if="match.cantidad_match"> | cantidad: {{ match.cantidad_match }}</span>
                <span v-if="match.razon_validacion"> | razon: {{ match.razon_validacion }}</span>
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
