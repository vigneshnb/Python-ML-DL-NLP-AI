/* MNIST AI — dashboard controller */
(() => {
  "use strict";

  const CONFIG = {
    allowedExt: ["png", "jpg", "jpeg"],
    allowedMime: ["image/png", "image/jpeg", "image/jpg", "image/pjpeg"],
    maxBytes: 5 * 1024 * 1024,
    minDim: 8,
    maxDim: 4096,
    endpoints: { predict: "/api/predict/", health: "/api/health/", modelInfo: "/api/model-info/" },
    healthIntervalMs: 30000,
  };

  const COLORS = {
    bar: "rgba(99, 102, 241, 0.35)",
    barBorder: "rgba(99, 102, 241, 0.7)",
    top: "#10B981",
    grid: "rgba(148, 163, 184, 0.10)",
    text: "#94A3B8",
    textStrong: "#F8FAFC",
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    dropzone: $("dropzone"), fileInput: $("fileInput"), dropEmpty: $("dropEmpty"), dropPreview: $("dropPreview"),
    previewImg: $("previewImg"), fileMeta: $("fileMeta"), fileName: $("fileName"), fileDims: $("fileDims"),
    fileSize: $("fileSize"), removeBtn: $("removeBtn"), alert: $("alert"), predictBtn: $("predictBtn"),
    clearBtn: $("clearBtn"), resultCard: $("resultCard"), resultStatus: $("resultStatus"), ring: $("confidenceRing"),
    digit: $("predDigit"), confidence: $("predConfidence"), state: $("predState"), time: $("predTime"),
    thumbOriginal: $("thumbOriginal"), thumbProcessed: $("thumbProcessed"), modelStatus: $("modelStatus"),
    modelSub: $("modelSub"), modelKv: $("modelKv"), modelLayers: $("modelLayers"), nav: $("nav"), navToggle: $("navToggle"),
  };

  const maxMb = Number(el.dropzone?.dataset.maxMb);
  if (maxMb > 0) CONFIG.maxBytes = maxMb * 1024 * 1024;

  const state = { file: null, objectUrl: null, inFlight: null, modelReady: false };
  let chart = null;

  /* ------------------------------------------------------------ helpers */
  const formatBytes = (b) => (b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(2)} MB`);
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function showAlert(message, kind = "error") {
    el.alert.textContent = message;
    el.alert.dataset.kind = kind;
    el.alert.hidden = false;
  }
  function hideAlert() {
    el.alert.hidden = true;
    el.alert.textContent = "";
  }

  function setPill(text, kind = "") {
    el.state.innerHTML = `<span class="pill ${kind ? "pill--" + kind : ""}">${escapeHtml(text)}</span>`;
  }

  function updatePredictButton() {
    el.predictBtn.disabled = !state.file || !!state.inFlight;
  }

  function loadImageDims(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = () => reject(new Error("decode"));
      img.src = url;
    });
  }

  /* ---------------------------------------------------------- file flow */
  async function handleFile(file) {
    if (!file) return;
    hideAlert();
    resetResult();

    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (!file.name.includes(".") || !CONFIG.allowedExt.includes(ext)) {
      return rejectFile("Unsupported file type. Please upload a PNG, JPG or JPEG image.");
    }
    if (file.type && !CONFIG.allowedMime.includes(file.type.toLowerCase())) {
      return rejectFile(`Unsupported content type (${file.type}). Please upload a PNG or JPEG image.`);
    }
    if (file.size === 0) return rejectFile("The selected file is empty.");
    if (file.size > CONFIG.maxBytes) {
      return rejectFile(`File is too large (${formatBytes(file.size)}). Maximum is ${formatBytes(CONFIG.maxBytes)}.`);
    }

    const url = URL.createObjectURL(file);
    let dims;
    try {
      dims = await loadImageDims(url);
    } catch {
      URL.revokeObjectURL(url);
      return rejectFile("This file could not be read as an image. It may be corrupted.");
    }
    if (dims.width < CONFIG.minDim || dims.height < CONFIG.minDim) {
      URL.revokeObjectURL(url);
      return rejectFile(`Image is too small (${dims.width}×${dims.height}). Minimum is ${CONFIG.minDim}px per side.`);
    }
    if (dims.width > CONFIG.maxDim || dims.height > CONFIG.maxDim) {
      URL.revokeObjectURL(url);
      return rejectFile(`Image is too large (${dims.width}×${dims.height}). Maximum is ${CONFIG.maxDim}px per side.`);
    }

    clearFile();
    state.file = file;
    state.objectUrl = url;

    el.previewImg.src = url;
    el.dropEmpty.hidden = true;
    el.dropPreview.hidden = false;
    el.dropzone.classList.add("has-file");
    el.fileName.textContent = file.name;
    el.fileName.title = file.name;
    el.fileDims.textContent = `${dims.width}×${dims.height}px`;
    el.fileSize.textContent = formatBytes(file.size);
    el.fileMeta.hidden = false;
    el.resultStatus.textContent = "Ready — click Predict Digit";
    setPill("Ready", "warn");
    updatePredictButton();
  }

  function rejectFile(message) {
    clearFile();
    showAlert(message);
    el.fileInput.value = "";
    updatePredictButton();
  }

  function clearFile() {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.file = null;
    state.objectUrl = null;
    el.fileInput.value = "";
    el.previewImg.removeAttribute("src");
    el.dropPreview.hidden = true;
    el.dropEmpty.hidden = false;
    el.dropzone.classList.remove("has-file");
    el.fileMeta.hidden = true;
    el.fileName.textContent = "—";
    el.fileDims.textContent = "—";
    el.fileSize.textContent = "—";
    updatePredictButton();
  }

  /* ---------------------------------------------------------- prediction */
  async function predict() {
    if (!state.file || state.inFlight) return; // prevent duplicate submissions
    hideAlert();

    const controller = new AbortController();
    state.inFlight = controller;
    updatePredictButton();
    el.predictBtn.classList.add("is-loading");
    el.predictBtn.querySelector(".btn__label").textContent = "Analyzing…";
    el.resultCard.dataset.state = "loading";
    el.resultStatus.textContent = "Running inference…";
    setPill("Processing", "loading");

    const body = new FormData();
    body.append("image", state.file, state.file.name);

    try {
      const res = await fetch(CONFIG.endpoints.predict, { method: "POST", body, signal: controller.signal });
      let payload = null;
      try { payload = await res.json(); } catch { /* non-JSON (e.g. proxy error) */ }

      if (!res.ok || !payload || !payload.success) {
        const msg = payload?.message || (res.status === 413 ? "File is too large." : `Prediction failed (HTTP ${res.status}).`);
        throw new Error(msg);
      }
      renderResult(payload.data);
    } catch (err) {
      if (err.name === "AbortError") return;
      const msg = err instanceof TypeError ? "Network error — could not reach the server." : err.message;
      showAlert(msg);
      el.resultCard.dataset.state = "error";
      el.resultStatus.textContent = "Prediction failed";
      setPill("Failed", "error");
    } finally {
      if (state.inFlight === controller) state.inFlight = null;
      el.predictBtn.classList.remove("is-loading");
      el.predictBtn.querySelector(".btn__label").textContent = "Predict Digit";
      updatePredictButton();
    }
  }

  function renderResult(data) {
    const conf = Number(data.confidence);
    el.resultCard.dataset.state = "success";
    el.digit.textContent = data.predicted_digit;
    el.ring.style.setProperty("--p", conf.toFixed(2));
    el.confidence.textContent = `${conf.toFixed(2)}%`;
    el.time.textContent = `${Number(data.execution_time_ms).toFixed(1)} ms`;
    el.resultStatus.textContent = `Predicted Digit: ${data.predicted_digit} · Confidence: ${conf.toFixed(2)}%`;
    if (conf >= 80) setPill("High confidence", "success");
    else if (conf >= 50) setPill("Moderate confidence", "warn");
    else setPill("Low confidence", "error");

    if (state.objectUrl) {
      el.thumbOriginal.src = state.objectUrl;
      el.thumbOriginal.hidden = false;
    }
    if (data.processed_image) {
      el.thumbProcessed.src = data.processed_image;
      el.thumbProcessed.hidden = false;
    }
    updateChart(data.probabilities, data.predicted_digit);
  }

  function resetResult() {
    el.resultCard.dataset.state = "idle";
    el.digit.textContent = "–";
    el.ring.style.setProperty("--p", 0);
    el.confidence.textContent = "—";
    el.time.textContent = "—";
    el.resultStatus.textContent = "Waiting for an image";
    setPill("Idle");
    el.thumbOriginal.hidden = true;
    el.thumbOriginal.removeAttribute("src");
    el.thumbProcessed.hidden = true;
    el.thumbProcessed.removeAttribute("src");
    resetChart();
  }

  function resetAll() {
    if (state.inFlight) {
      state.inFlight.abort();
      state.inFlight = null;
      el.predictBtn.classList.remove("is-loading");
      el.predictBtn.querySelector(".btn__label").textContent = "Predict Digit";
    }
    clearFile();
    resetResult();
    hideAlert();
    updatePredictButton();
  }

  /* --------------------------------------------------------------- chart */
  function initChart() {
    const canvas = $("probChart");
    if (typeof Chart === "undefined" || !canvas) {
      canvas?.parentElement && (canvas.parentElement.innerHTML = '<p class="muted">Chart library failed to load.</p>');
      return;
    }
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.color = COLORS.text;

    const valueLabels = {
      id: "valueLabels",
      afterDatasetsDraw(c) {
        const { ctx } = c;
        const meta = c.getDatasetMeta(0);
        const data = c.data.datasets[0].data;
        if (!c.$hasData) return;
        ctx.save();
        ctx.font = "600 12px " + Chart.defaults.font.family;
        ctx.textBaseline = "middle";
        meta.data.forEach((bar, i) => {
          const v = data[i];
          ctx.fillStyle = i === c.$top ? COLORS.textStrong : COLORS.text;
          ctx.fillText(`${v.toFixed(v >= 0.01 || v === 0 ? 2 : 4)}%`, bar.x + 8, bar.y);
        });
        ctx.restore();
      },
    };

    chart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: [...Array(10).keys()].map(String),
        datasets: [{
          data: Array(10).fill(0),
          backgroundColor: Array(10).fill(COLORS.bar),
          borderColor: Array(10).fill(COLORS.barBorder),
          borderWidth: 1,
          borderRadius: 6,
          borderSkipped: false,
          barPercentage: 0.72,
          categoryPercentage: 0.9,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 700, easing: "easeOutCubic" },
        layout: { padding: { right: 64 } },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0B1120", borderColor: "rgba(148,163,184,.25)", borderWidth: 1, padding: 10,
            callbacks: { title: (items) => `Digit ${items[0].label}`, label: (item) => ` ${item.raw.toFixed(4)}%` },
          },
        },
        scales: {
          x: { min: 0, max: 100, grid: { color: COLORS.grid }, border: { display: false }, ticks: { callback: (v) => `${v}%` } },
          y: { grid: { display: false }, border: { display: false },
               ticks: { font: { size: 14, weight: "600", family: "'JetBrains Mono', monospace" }, color: COLORS.textStrong } },
        },
      },
      plugins: [valueLabels],
    });
    chart.$hasData = false;
  }

  function updateChart(probabilities, top) {
    if (!chart) return;
    const values = [...Array(10).keys()].map((d) => Number(probabilities[String(d)] ?? 0));
    const ds = chart.data.datasets[0];
    ds.data = values;
    ds.backgroundColor = values.map((_, i) => (i === top ? COLORS.top : COLORS.bar));
    ds.borderColor = values.map((_, i) => (i === top ? COLORS.top : COLORS.barBorder));
    chart.$top = top;
    chart.$hasData = true;
    chart.update();
  }

  function resetChart() {
    if (!chart) return;
    const ds = chart.data.datasets[0];
    ds.data = Array(10).fill(0);
    ds.backgroundColor = Array(10).fill(COLORS.bar);
    ds.borderColor = Array(10).fill(COLORS.barBorder);
    chart.$top = null;
    chart.$hasData = false;
    chart.update();
  }

  /* ------------------------------------------------------- model status */
  async function checkHealth() {
    try {
      const res = await fetch(CONFIG.endpoints.health, { cache: "no-store" });
      const payload = await res.json();
      state.modelReady = !!payload?.data?.model_loaded;
    } catch {
      state.modelReady = false;
    }
    el.modelStatus.dataset.state = state.modelReady ? "online" : "offline";
    el.modelStatus.querySelector(".status__text").textContent = state.modelReady ? "Model online" : "Model offline";
  }

  async function loadModelInfo() {
    try {
      const res = await fetch(CONFIG.endpoints.modelInfo);
      const { data } = await res.json();
      if (!data?.loaded) {
        el.modelSub.textContent = data?.error || "Model not loaded";
        return;
      }
      el.modelSub.textContent = `${data.file} · ${data.framework}`;
      const rows = [
        ["Format", data.format],
        ["Input shape", `(${data.input_shape.map((v) => v ?? "batch").join(", ")})`],
        ["Output", `${data.num_classes} classes · softmax`],
        ["Parameters", data.total_params.toLocaleString()],
        ["Preprocessing", data.preprocessing.mode === "mnist" ? "MNIST-normalised" : "Notebook (raw resize)"],
        ["Normalisation", "x / 255 → [0, 1]"],
      ];
      el.modelKv.innerHTML = rows.map(([k, v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd></div>`).join("");
      el.modelLayers.innerHTML = data.layers.map((l) => {
        const detail = [l.units ? `${l.units} units` : null, l.activation, `${l.params.toLocaleString()} params`].filter(Boolean).join(" · ");
        return `<li><strong>${escapeHtml(l.type)}</strong><span>${escapeHtml(detail)}</span></li>`;
      }).join("");
    } catch {
      el.modelSub.textContent = "Could not load model details.";
    }
  }

  /* ------------------------------------------------------------- events */
  function bindEvents() {
    el.dropzone.addEventListener("click", (e) => {
      if (state.file || e.target.closest("button")) return;
      el.fileInput.click();
    });
    el.dropzone.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === " ") && !state.file) { e.preventDefault(); el.fileInput.click(); }
    });
    el.fileInput.addEventListener("change", () => handleFile(el.fileInput.files[0]));

    ["dragenter", "dragover"].forEach((type) => el.dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      el.dropzone.classList.add("is-dragover");
    }));
    ["dragleave", "dragend", "drop"].forEach((type) => el.dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      if (type === "dragleave" && el.dropzone.contains(e.relatedTarget)) return;
      el.dropzone.classList.remove("is-dragover");
    }));
    el.dropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer?.files;
      if (!files || !files.length) return;
      if (files.length > 1) showAlert("Please drop a single image. Using the first file.", "info");
      handleFile(files[0]);
    });
    // Stop the browser from opening files dropped outside the zone.
    ["dragover", "drop"].forEach((t) => window.addEventListener(t, (e) => e.preventDefault()));

    el.removeBtn.addEventListener("click", (e) => { e.stopPropagation(); resetAll(); });
    el.predictBtn.addEventListener("click", predict);
    el.clearBtn.addEventListener("click", resetAll);

    el.navToggle.addEventListener("click", () => {
      const open = el.nav.classList.toggle("is-open");
      el.navToggle.setAttribute("aria-expanded", String(open));
    });
    el.nav.querySelectorAll("a").forEach((a) => a.addEventListener("click", () => {
      el.nav.classList.remove("is-open");
      el.navToggle.setAttribute("aria-expanded", "false");
    }));

    const links = [...el.nav.querySelectorAll(".nav__link")];
    const sections = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          links.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === `#${entry.target.id}`));
        });
      }, { rootMargin: "-45% 0px -50% 0px" });
      sections.forEach((s) => io.observe(s));
    }
  }

  /* --------------------------------------------------------------- init */
  function init() {
    initChart();
    bindEvents();
    resetAll();
    checkHealth();
    loadModelInfo();
    setInterval(checkHealth, CONFIG.healthIntervalMs);
  }

  window.MNISTApp = { resetAll, predict, handleFile, state };
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", init) : init();
})();
