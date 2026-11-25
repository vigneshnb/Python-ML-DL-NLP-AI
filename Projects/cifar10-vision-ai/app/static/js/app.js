import {
  addHistory,
  canPredict,
  confidenceLevel,
  createInitialState,
  formatBytes,
  formatType,
  resetState,
  resultView,
  validateDimensions,
  validateFile,
} from "./state.js";

const $ = (id) => document.getElementById(id);

const els = {
  dropzone: $("dropzone"),
  fileInput: $("fileInput"),
  browseBtn: $("browseBtn"),
  dzEmpty: $("dzEmpty"),
  dzPreview: $("dzPreview"),
  previewImg: $("previewImg"),
  removeBtn: $("removeBtn"),
  fileMeta: $("fileMeta"),
  fileName: $("fileName"),
  fileSize: $("fileSize"),
  fileDims: $("fileDims"),
  fileType: $("fileType"),
  alertBox: $("alertBox"),
  alertText: $("alertText"),
  predictBtn: $("predictBtn"),
  clearBtn: $("clearBtn"),
  resultCard: $("resultCard"),
  resultEmpty: $("resultEmpty"),
  resultLoading: $("resultLoading"),
  resultError: $("resultError"),
  resultErrorText: $("resultErrorText"),
  retryBtn: $("retryBtn"),
  resultBody: $("resultBody"),
  resultThumb: $("resultThumb"),
  resultLabel: $("resultLabel"),
  resultClass: $("resultClass"),
  resultTime: $("resultTime"),
  resultConfidence: $("resultConfidence"),
  confidenceLevel: $("confidenceLevel"),
  gauge: $("gauge"),
  gaugeFill: $("gaugeFill"),
  srAnnounce: $("srAnnounce"),
  chartCard: $("chartCard"),
  probTable: $("probTable"),
  historyList: $("historyList"),
  historyEmpty: $("historyEmpty"),
  clearHistoryBtn: $("clearHistoryBtn"),
  modelStatus: $("modelStatus"),
  navToggle: $("navToggle"),
  siteNav: $("siteNav"),
};

const limits = {
  maxBytes: Number(els.dropzone.dataset.maxMb) * 1024 * 1024,
  minDim: Number(els.dropzone.dataset.minDim),
  maxDim: Number(els.dropzone.dataset.maxDim),
};

// Class names come from the server-rendered table (single source: app/constants.py)
const CLASS_NAMES = Array.from(els.probTable.querySelectorAll("tbody tr")).map(
  (tr) => tr.children[1].textContent.trim(),
);
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 50;

let state = createInitialState();
let history = [];
let chart = null;
let requestSeq = 0; // responses from a request made before Clear are ignored

/* ============================================================ chart */
const C = {
  rest: "rgba(99, 102, 241, 0.45)",
  restHover: "rgba(99, 102, 241, 0.7)",
  top: "#10B981",
  grid: "rgba(38, 52, 73, 0.75)",
  muted: "#94A3B8",
  text: "#F8FAFC",
};

// Draws "12.34%" at the end of each bar (Chart.js core has no data labels).
const valueLabels = {
  id: "valueLabels",
  afterDatasetsDraw(c) {
    if (c.$empty) return;
    const { ctx } = c;
    const meta = c.getDatasetMeta(0);
    const data = c.data.datasets[0].data;
    ctx.save();
    ctx.font = "500 11px 'JetBrains Mono', monospace";
    ctx.textBaseline = "middle";
    meta.data.forEach((bar, i) => {
      const v = data[i];
      const label = `${v.toFixed(2)}%`;
      const w = ctx.measureText(label).width;
      const inside = bar.x - c.chartArea.left > w + 16;
      ctx.fillStyle = inside ? "#0B1120" : (i === c.$top ? "#6EE7B7" : C.muted);
      ctx.textAlign = inside ? "right" : "left";
      ctx.fillText(label, inside ? bar.x - 8 : bar.x + 6, bar.y);
    });
    ctx.restore();
  },
};

function initChart() {
  if (!window.Chart) return; // CDN blocked: the table still shows the values
  chart = new window.Chart($("probChart").getContext("2d"), {
    type: "bar",
    data: {
      labels: CLASS_NAMES,
      datasets: [{
        data: CLASS_NAMES.map(() => 0),
        backgroundColor: CLASS_NAMES.map(() => C.rest),
        hoverBackgroundColor: CLASS_NAMES.map(() => C.restHover),
        borderRadius: 7,
        borderSkipped: false,
        barPercentage: 0.7,
        categoryPercentage: 0.9,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600, easing: "easeOutQuart" },
      layout: { padding: { right: 8 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#111C32",
          borderColor: "#263449",
          borderWidth: 1,
          padding: 10,
          cornerRadius: 10,
          titleColor: C.text,
          bodyColor: C.text,
          displayColors: false,
          callbacks: { label: (c) => `${c.parsed.x.toFixed(2)}% confidence` },
        },
      },
      scales: {
        x: {
          min: 0,
          max: 100,
          grid: { color: C.grid, drawTicks: false },
          border: { display: false },
          ticks: { color: C.muted, padding: 8, stepSize: 25, callback: (v) => `${v}%` },
        },
        y: {
          grid: { display: false },
          border: { display: false },
          ticks: { color: C.text, padding: 8, font: { family: "Inter", size: 12.5, weight: 500 } },
        },
      },
    },
    plugins: [valueLabels],
  });
  chart.$empty = true;
}

function updateChart(probs, topIndex) {
  if (!chart) return;
  const ds = chart.data.datasets[0];
  ds.data = CLASS_NAMES.map((n) => probs[n] ?? 0);
  ds.backgroundColor = CLASS_NAMES.map((_, i) => (i === topIndex ? C.top : C.rest));
  ds.hoverBackgroundColor = CLASS_NAMES.map((_, i) => (i === topIndex ? C.top : C.restHover));
  chart.$empty = false;
  chart.$top = topIndex;
  chart.update();
}

function resetChart() {
  if (!chart) return;
  const ds = chart.data.datasets[0];
  ds.data = CLASS_NAMES.map(() => 0);
  ds.backgroundColor = CLASS_NAMES.map(() => C.rest);
  ds.hoverBackgroundColor = CLASS_NAMES.map(() => C.restHover);
  chart.$empty = true;
  chart.$top = -1;
  chart.update();
}

/* ============================================================ render */
function render() {
  const hasFile = state.file !== null;

  // upload area
  if (els.dropzone.dataset.state !== "dragover") {
    els.dropzone.dataset.state = hasFile ? "filled" : "empty";
  }
  els.dzEmpty.hidden = hasFile;
  els.dzPreview.hidden = !hasFile;
  if (hasFile && state.previewUrl) els.previewImg.src = state.previewUrl;
  else els.previewImg.removeAttribute("src");

  els.fileMeta.hidden = !state.meta;
  if (state.meta) {
    els.fileName.textContent = state.meta.name;
    els.fileName.title = state.meta.name;
    els.fileSize.textContent = formatBytes(state.meta.size);
    els.fileDims.textContent = state.meta.width ? `${state.meta.width} × ${state.meta.height}` : "…";
    els.fileType.textContent = state.meta.type;
  }

  // file problems show under the upload; API failures show in the result card
  els.alertBox.hidden = !state.error;
  els.alertText.textContent = state.error || "";

  // buttons
  const loading = state.status === "loading";
  els.predictBtn.disabled = !canPredict(state);
  els.predictBtn.classList.toggle("is-loading", loading);
  els.predictBtn.setAttribute("aria-busy", String(loading));
  els.predictBtn.querySelector(".btn-label").textContent = loading ? "Analyzing…" : "Predict Image";

  // result panel
  const view = resultView(state);
  els.resultCard.dataset.view = view;
  els.chartCard.dataset.view = view === "success" ? "success" : view === "loading" ? "loading" : "empty";
  els.resultEmpty.hidden = view !== "empty";
  els.resultLoading.hidden = view !== "loading";
  els.resultError.hidden = view !== "error";
  els.resultBody.hidden = view !== "success";
  els.resultCard.setAttribute("aria-busy", String(view === "loading"));
  els.resultErrorText.textContent = view === "error" ? state.apiError : "";

  if (view === "success") {
    const r = state.result;
    els.resultThumb.src = state.previewUrl;
    els.resultThumb.alt = `Uploaded image, classified as ${r.predicted_label}`;
    els.resultLabel.textContent = r.predicted_label;
    els.resultClass.textContent = String(r.predicted_class);
    els.resultTime.textContent = `${Math.round(r.execution_time_ms)} ms`;
    els.resultConfidence.textContent = `${r.confidence.toFixed(2)}%`;
    els.confidenceLevel.textContent = `${confidenceLevel(r.confidence)} confidence`;
    els.gauge.setAttribute("aria-label", `Confidence ${r.confidence.toFixed(2)} percent`);
    // start empty, then animate to the value
    els.gaugeFill.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      els.gaugeFill.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE * (1 - r.confidence / 100));
    }));
  } else {
    els.resultThumb.removeAttribute("src");
    els.resultThumb.alt = "";
    els.resultLabel.textContent = "";
    els.resultClass.textContent = "";
    els.resultTime.textContent = "";
    els.resultConfidence.textContent = "";
    els.confidenceLevel.textContent = "Confidence";
    els.gaugeFill.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
  }

  // probability table
  els.probTable.querySelectorAll("tbody tr").forEach((tr, i) => {
    const cell = tr.querySelector(".prob-cell");
    if (view === "success") {
      cell.textContent = `${state.result.probabilities[CLASS_NAMES[i]].toFixed(2)}%`;
      tr.classList.toggle("is-top", i === state.result.predicted_class);
    } else {
      cell.textContent = "—";
      tr.classList.remove("is-top");
    }
  });
}

function announce(text) {
  els.srAnnounce.textContent = "";
  // re-set on the next frame so screen readers announce repeated text too
  requestAnimationFrame(() => { els.srAnnounce.textContent = text; });
}

function renderHistory() {
  els.historyList.replaceChildren(
    ...history.map((h) => {
      const li = document.createElement("li");
      li.className = "history-item";
      const img = document.createElement("img");
      img.src = h.thumb;
      img.alt = `Earlier image classified as ${h.label}`;
      const label = document.createElement("span");
      label.className = "h-label";
      label.textContent = h.label;
      const row = document.createElement("div");
      row.className = "h-row";
      const conf = document.createElement("span");
      conf.className = "h-conf";
      conf.textContent = `${h.confidence.toFixed(2)}%`;
      const time = document.createElement("span");
      time.className = "h-time";
      time.textContent = h.time;
      row.append(conf, time);
      li.append(img, label, row);
      return li;
    }),
  );
  els.historyEmpty.hidden = history.length > 0;
  els.clearHistoryBtn.disabled = history.length === 0;
}

/* ============================================================ actions */
function clearAll({ focus = false } = {}) {
  requestSeq += 1; // invalidate any in-flight prediction
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state = resetState();
  els.fileInput.value = "";
  resetChart();
  render();
  if (focus) els.browseBtn.focus();
}

function handleFile(file) {
  if (!file) return;
  clearAll();
  const err = validateFile(file, limits);
  if (err) {
    state.error = err;
    render();
    return;
  }

  const url = URL.createObjectURL(file);
  state.file = file;
  state.previewUrl = url;
  state.meta = { name: file.name, size: file.size, type: formatType(file), width: 0, height: 0 };
  state.status = "ready";
  render();

  const probe = new Image();
  probe.onload = () => {
    if (state.previewUrl !== url) return; // cleared meanwhile
    state.meta.width = probe.naturalWidth;
    state.meta.height = probe.naturalHeight;
    state.error = validateDimensions(probe.naturalWidth, probe.naturalHeight, limits);
    render();
  };
  probe.onerror = () => {
    if (state.previewUrl !== url) return;
    state.error = "This file could not be read as an image. It may be corrupted.";
    render();
  };
  probe.src = url;
}

async function predict() {
  if (!canPredict(state)) return; // also blocks duplicate submissions
  const seq = ++requestSeq;
  state.status = "loading";
  state.apiError = null;
  state.result = null;
  resetChart();
  render();

  const body = new FormData();
  body.append("image", state.file, state.file.name);

  try {
    const res = await fetch("/api/predict/", { method: "POST", body });
    let payload = null;
    try { payload = await res.json(); } catch { /* non-JSON, e.g. proxy error page */ }
    if (seq !== requestSeq) return;

    if (!res.ok || !payload || !payload.success) {
      failPrediction(payload?.message || `Prediction failed (HTTP ${res.status}). Please try again.`);
      return;
    }

    state.status = "success";
    state.result = payload.data;
    render();
    updateChart(payload.data.probabilities, payload.data.predicted_class);
    announce(`Prediction: ${payload.data.predicted_label}, ${payload.data.confidence.toFixed(2)} percent confidence.`);
    await pushHistory(payload.data);
  } catch {
    if (seq !== requestSeq) return;
    failPrediction("Network error. Check your connection and try again.");
  }
}

function failPrediction(message) {
  state.status = "error";
  state.apiError = message;
  render();
  announce(`Prediction failed. ${message}`);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

async function makeThumb(url, size = 160) {
  const img = await loadImage(url);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const s = Math.min(img.naturalWidth, img.naturalHeight);
  canvas.getContext("2d").drawImage(
    img, (img.naturalWidth - s) / 2, (img.naturalHeight - s) / 2, s, s, 0, 0, size, size,
  );
  return canvas.toDataURL("image/jpeg", 0.8);
}

async function pushHistory(data) {
  try {
    const thumb = await makeThumb(state.previewUrl);
    history = addHistory(history, {
      thumb,
      label: data.predicted_label,
      confidence: data.confidence,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    });
    renderHistory();
  } catch { /* thumbnail is cosmetic */ }
}

/* ============================================================ model status */
async function loadModelInfo() {
  const setStatus = (s, text) => {
    els.modelStatus.dataset.state = s;
    els.modelStatus.querySelector(".status-text").textContent = text;
  };
  try {
    const res = await fetch("/api/model-info/");
    const payload = await res.json();
    if (!res.ok || !payload.success) throw new Error(payload.message);
    setStatus("ready", "Model ready");
    const d = payload.data;
    const shape = (arr) => `(${arr.map((v) => v ?? "None").join(", ")})`;
    const values = {
      model_name: d.model_name,
      input_shape: shape(d.input_shape),
      output_shape: shape(d.output_shape),
      output_type: d.output_type,
      preprocessing: `${d.preprocessing.color_mode}, ${d.preprocessing.resize}, ${d.preprocessing.scaling}`,
      framework: Object.entries(d.framework).map(([k, v]) => `${k} ${v}`).join(" · "),
    };
    document.querySelectorAll("#modelGrid dd").forEach((dd) => {
      dd.textContent = values[dd.dataset.k] ?? "—";
    });
  } catch {
    setStatus("error", "Model unavailable");
  }
}

/* ============================================================ events */
function openPicker() { els.fileInput.click(); }

function setNav(open) {
  els.siteNav.classList.toggle("is-open", open);
  els.navToggle.setAttribute("aria-expanded", String(open));
  els.navToggle.setAttribute("aria-label", open ? "Close navigation menu" : "Open navigation menu");
}

function isTyping(target) {
  return target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
}

function bind() {
  // upload: click anywhere on the empty drop area, or the Browse button
  els.dropzone.addEventListener("click", (e) => {
    if (!state.file && !e.target.closest("button")) openPicker();
  });
  els.browseBtn.addEventListener("click", openPicker);
  els.fileInput.addEventListener("change", () => handleFile(els.fileInput.files[0]));

  let dragDepth = 0;
  els.dropzone.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth += 1;
    els.dropzone.dataset.state = "dragover";
  });
  els.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
  });
  els.dropzone.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) { els.dropzone.dataset.state = ""; render(); }
  });
  els.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    els.dropzone.dataset.state = "";
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file); else render();
  });
  // don't let the browser open a file dropped outside the drop area
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  // paste an image from the clipboard
  window.addEventListener("paste", (e) => {
    if (isTyping(e.target)) return;
    const item = Array.from(e.clipboardData?.items || []).find((i) => i.kind === "file");
    const file = item?.getAsFile();
    if (file) {
      e.preventDefault();
      const named = file.name && file.name !== "image.png" ? file : new File([file], `pasted.${(file.type.split("/")[1] || "png").replace("jpeg", "jpg")}`, { type: file.type });
      handleFile(named);
    }
  });

  els.predictBtn.addEventListener("click", predict);
  els.retryBtn.addEventListener("click", predict);
  els.clearBtn.addEventListener("click", () => clearAll({ focus: true }));
  els.removeBtn.addEventListener("click", () => clearAll({ focus: true }));
  els.clearHistoryBtn.addEventListener("click", () => { history = []; renderHistory(); });

  // keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (els.siteNav.classList.contains("is-open")) { setNav(false); els.navToggle.focus(); return; }
      if (!isTyping(e.target) && (state.file || state.error)) { e.preventDefault(); clearAll({ focus: true }); }
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      predict();
    }
  });

  // mobile navigation
  els.navToggle.addEventListener("click", () => setNav(!els.siteNav.classList.contains("is-open")));
  els.siteNav.querySelectorAll("a").forEach((a) => a.addEventListener("click", () => setNav(false)));
}

initChart();
bind();
render();
renderHistory();
loadModelInfo();
