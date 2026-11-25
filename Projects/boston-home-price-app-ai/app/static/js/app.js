/* Home Value Estimator - DOM wiring. Logic lives in form-logic.js (HVELogic). */
(function () {
  "use strict";

  var L = window.HVELogic;
  var schema = JSON.parse(document.getElementById("featureSchema").textContent);
  var FEATURES = schema.features;
  var BY_NAME = {};
  FEATURES.forEach(function (f) { BY_NAME[f.name] = f; });

  var $ = function (id) { return document.getElementById(id); };
  var els = {
    form: $("estimateForm"),
    predictBtn: $("predictBtn"),
    clearBtn: $("clearBtn"),
    fillMediansBtn: $("fillMediansBtn"),
    newEstimateBtn: $("newEstimateBtn"),
    formStatus: $("formStatus"),
    progressFill: $("progressFill"),
    resultCard: $("resultCard"),
    resultEmpty: $("resultEmpty"),
    resultLoading: $("resultLoading"),
    resultError: $("resultError"),
    resultErrorText: $("resultErrorText"),
    resultBody: $("resultBody"),
    resultValue: $("resultValue"),
    resultRaw: $("resultRaw"),
    resultTime: $("resultTime"),
    indicatorFill: $("indicatorFill"),
    indicatorMarker: $("indicatorMarker"),
    indicatorMedian: $("indicatorMedian"),
    indicatorMin: $("indicatorMin"),
    indicatorMax: $("indicatorMax"),
    indicatorMedianLabel: $("indicatorMedianLabel"),
    warnings: $("resultWarnings"),
    zBars: $("zBars"),
    inputSummary: $("inputSummary"),
    modelStatus: $("modelStatus"),
    navToggle: $("navToggle"),
    siteNav: $("siteNav"),
  };

  var state = L.initialState();
  var requestSeq = 0;
  var controller = null;

  // ------------------------------------------------------------- form access
  function rawValue(name) {
    var f = BY_NAME[name];
    if (f.control === "toggle") {
      var checked = els.form.querySelector('input[name="' + name + '"]:checked');
      return checked ? checked.value : "";
    }
    var input = els.form.elements[name];
    // Number inputs report "" for unparseable text; surface it as invalid, not missing.
    if (input.validity && input.validity.badInput) return "invalid";
    return input.value;
  }

  function rawValues() {
    var out = {};
    FEATURES.forEach(function (f) { out[f.name] = rawValue(f.name); });
    return out;
  }

  function setValue(name, value) {
    var f = BY_NAME[name];
    if (f.control === "toggle") {
      els.form.querySelectorAll('input[name="' + name + '"]').forEach(function (r) {
        r.checked = String(value) === r.value;
      });
    } else {
      els.form.elements[name].value = value === "" ? "" : String(value);
    }
  }

  function fieldEl(name) { return els.form.querySelector('[data-field="' + name + '"]'); }

  function showFieldError(name, message) {
    var wrap = fieldEl(name);
    var err = $("err-" + name);
    err.textContent = message || "";
    wrap.classList.toggle("has-error", !!message);
    wrap.querySelectorAll("input").forEach(function (i) {
      if (message) i.setAttribute("aria-invalid", "true");
      else i.removeAttribute("aria-invalid");
    });
  }

  function showFieldNote(name, outside) {
    fieldEl(name).classList.toggle("is-outside", !!outside);
  }

  // ------------------------------------------------------------- validation
  function validate(opts) {
    opts = opts || {};
    var res = L.validateAll(FEATURES, rawValues());
    FEATURES.forEach(function (f) {
      var show = opts.showAll || state.touched[f.name];
      showFieldError(f.name, show ? res.errors[f.name] : "");
      showFieldNote(f.name, f.name in res.values && L.isOutsideTraining(f, res.values[f.name]));
    });
    var remaining = Object.keys(res.errors).length;
    var ready = FEATURES.length - remaining;
    els.progressFill.style.width = (ready / FEATURES.length) * 100 + "%";
    els.progressFill.classList.toggle("is-complete", res.valid);
    els.predictBtn.disabled = !res.valid || state.submitting;
    if (state.submitting) {
      els.formStatus.textContent = "Estimating…";
    } else if (res.valid) {
      els.formStatus.textContent = "All values look good. Ready to estimate.";
    } else {
      els.formStatus.textContent = remaining === FEATURES.length
        ? "Complete all fields to enable the estimate."
        : ready + " of " + FEATURES.length + " values ready · " + remaining + " to go.";
    }
    els.formStatus.classList.toggle("is-ready", res.valid && !state.submitting);
    return res;
  }

  // ------------------------------------------------------------- rendering
  function setPhase(phase) {
    state.phase = phase;
    els.resultCard.dataset.state = phase;
    els.resultEmpty.hidden = phase !== "empty";
    els.resultLoading.hidden = phase !== "loading";
    els.resultError.hidden = phase !== "error";
    els.resultBody.hidden = phase !== "result";
  }

  function setSubmitting(on) {
    state.submitting = on;
    els.predictBtn.classList.toggle("is-loading", on);
    els.predictBtn.setAttribute("aria-busy", on ? "true" : "false");
    els.predictBtn.querySelector(".btn-label").textContent = on ? "Estimating…" : "Estimate Home Value";
    validate();
  }

  function clearChildren(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function renderResult(data) {
    var t = data.target_context;
    els.resultValue.textContent = data.display_value || "Not meaningful";
    els.resultValue.classList.toggle("is-invalid", !data.display_value);
    els.resultRaw.textContent = data.predicted_value.toFixed(4);
    els.resultTime.textContent = data.execution_time_ms.toFixed(1) + " ms";

    var pct = L.percentOnScale(data.predicted_value, t.train_min, t.train_max);
    els.indicatorFill.style.width = pct + "%";
    els.indicatorMarker.style.left = pct + "%";
    els.indicatorMedian.style.left = L.percentOnScale(t.train_median, t.train_min, t.train_max) + "%";
    els.indicatorMin.textContent = L.formatUSD(t.train_min);
    els.indicatorMax.textContent = L.formatUSD(t.train_max);
    els.indicatorMedianLabel.textContent = "median " + L.formatUSD(t.train_median);

    clearChildren(els.warnings);
    (data.warnings || []).forEach(function (w) {
      var li = document.createElement("li");
      li.textContent = w;
      els.warnings.appendChild(li);
    });
    els.warnings.hidden = !(data.warnings && data.warnings.length);

    clearChildren(els.zBars);
    FEATURES.forEach(function (f) {
      var z = data.standardized_inputs[f.name];
      var g = L.zBar(z, 3);
      var row = document.createElement("div");
      row.className = "zrow";
      var name = document.createElement("span");
      name.className = "zname";
      name.textContent = f.name;
      name.title = f.label;
      var track = document.createElement("span");
      track.className = "ztrack";
      var bar = document.createElement("span");
      bar.className = "zbar " + g.direction + (g.clipped ? " clipped" : "");
      bar.style.left = g.left + "%";
      bar.style.width = Math.max(g.width, 0.8) + "%";
      track.appendChild(bar);
      var val = document.createElement("span");
      val.className = "zval";
      val.textContent = (z >= 0 ? "+" : "") + z.toFixed(2);
      row.append(name, track, val);
      els.zBars.appendChild(row);
    });

    clearChildren(els.inputSummary);
    FEATURES.forEach(function (f) {
      var div = document.createElement("div");
      var dt = document.createElement("dt");
      dt.textContent = f.label;
      var code = document.createElement("code");
      code.textContent = f.name;
      dt.appendChild(document.createTextNode(" "));
      dt.appendChild(code);
      var dd = document.createElement("dd");
      var v = data.feature_summary[f.name];
      dd.textContent = f.control === "toggle"
        ? (f.options.find(function (o) { return o.value === v; }) || {}).label || String(v)
        : v.toLocaleString("en-US", { maximumFractionDigits: 6 }) + (f.unit ? " " + f.unit : "");
      div.append(dt, dd);
      els.inputSummary.appendChild(div);
    });
  }

  function resetResultView() {
    clearChildren(els.warnings);
    clearChildren(els.zBars);
    clearChildren(els.inputSummary);
    els.warnings.hidden = true;
    els.resultValue.textContent = "—";
    els.resultValue.classList.remove("is-invalid");
    els.resultRaw.textContent = "—";
    els.resultTime.textContent = "—";
    els.indicatorFill.style.width = "0%";
    els.indicatorMarker.style.left = "0%";
    els.resultErrorText.textContent = "";
  }

  // On stacked (single-column) layouts the result sits below the form.
  function revealResult() {
    els.resultCard.scrollTop = 0; // desktop: card scrolls internally; show the value first
    if (window.matchMedia && window.matchMedia("(max-width: 1080px)").matches) {
      els.resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  // ------------------------------------------------------------- actions
  function predict() {
    var res = validate({ showAll: true });
    FEATURES.forEach(function (f) { state.touched[f.name] = true; });
    if (!res.valid || state.submitting) {
      if (!res.valid) {
        var firstBad = FEATURES.find(function (f) { return res.errors[f.name]; });
        var input = fieldEl(firstBad.name).querySelector("input");
        if (input) input.focus();
      }
      return;
    }

    var seq = ++requestSeq;
    if (controller) controller.abort();
    controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    setSubmitting(true);
    setPhase("loading");

    fetch("/api/predict/", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(L.buildPayload(FEATURES, res.values)),
      signal: controller ? controller.signal : undefined,
    })
      .then(function (r) {
        return r.json().catch(function () { return { success: false, message: "Unexpected server response (" + r.status + ")." }; });
      })
      .then(function (body) {
        if (seq !== requestSeq) return; // superseded or cleared
        if (body && body.success) {
          state.result = body.data;
          renderResult(body.data);
          setPhase("result");
          revealResult();
        } else {
          state.error = (body && body.message) || "Prediction failed.";
          if (body && body.errors) {
            Object.keys(body.errors).forEach(function (k) { if (BY_NAME[k]) showFieldError(k, body.errors[k]); });
          }
          els.resultErrorText.textContent = state.error;
          setPhase("error");
          revealResult();
        }
      })
      .catch(function (err) {
        if (seq !== requestSeq || (err && err.name === "AbortError")) return;
        state.error = "Could not reach the server. Check your connection and try again.";
        els.resultErrorText.textContent = state.error;
        setPhase("error");
      })
      .finally(function () {
        if (seq !== requestSeq) return;
        controller = null;
        setSubmitting(false);
      });
  }

  function clearAll() {
    requestSeq++;                       // ignore any in-flight response
    if (controller) { controller.abort(); controller = null; }
    els.form.reset();
    FEATURES.forEach(function (f) { setValue(f.name, ""); showFieldError(f.name, ""); showFieldNote(f.name, false); });
    state = L.initialState();
    els.predictBtn.classList.remove("is-loading");
    els.predictBtn.removeAttribute("aria-busy");
    els.predictBtn.querySelector(".btn-label").textContent = "Estimate Home Value";
    resetResultView();
    setPhase("empty");
    validate();
    els.form.scrollIntoView({ behavior: "smooth", block: "start" });
    var first = els.form.querySelector("input.input");
    if (first) first.focus({ preventScroll: true });
  }

  function fillMedians() {
    FEATURES.forEach(function (f) { setValue(f.name, f.default); state.touched[f.name] = true; });
    validate({ showAll: true });
  }

  // ------------------------------------------------------------- events
  els.form.addEventListener("input", function (e) {
    var name = e.target && e.target.dataset ? e.target.dataset.feature : null;
    if (name && BY_NAME[name].control === "toggle") state.touched[name] = true;
    validate();
  });
  els.form.addEventListener("focusout", function (e) {
    var name = e.target && e.target.dataset ? e.target.dataset.feature : null;
    if (name) { state.touched[name] = true; validate(); }
  });
  els.form.addEventListener("submit", function (e) { e.preventDefault(); predict(); });
  els.clearBtn.addEventListener("click", clearAll);
  els.newEstimateBtn.addEventListener("click", clearAll);
  els.fillMediansBtn.addEventListener("click", fillMedians);

  els.navToggle.addEventListener("click", function () {
    var open = els.siteNav.classList.toggle("is-open");
    els.navToggle.setAttribute("aria-expanded", open ? "true" : "false");
  });
  els.siteNav.addEventListener("click", function (e) {
    if (e.target.tagName === "A") {
      els.siteNav.classList.remove("is-open");
      els.navToggle.setAttribute("aria-expanded", "false");
    }
  });

  // ------------------------------------------------------------- model status
  function setModelStatus(kind, text) {
    els.modelStatus.dataset.state = kind;
    els.modelStatus.querySelector(".status-text").textContent = text;
  }
  fetch("/api/health/", { headers: { Accept: "application/json" } })
    .then(function (r) { return r.json(); })
    .then(function (b) {
      if (b && b.success && b.data.model_loaded) setModelStatus("ready", "Model ready");
      else if (b && b.success && b.data.model_status === "not_loaded") setModelStatus("idle", "Model loads on first use");
      else setModelStatus("down", "Model unavailable");
    })
    .catch(function () { setModelStatus("down", "Status unavailable"); });

  // expose for debugging / tests
  window.HVEApp = { clearAll: clearAll, predict: predict, getState: function () { return state; } };

  setPhase("empty");
  validate();
})();
