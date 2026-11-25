/* Pure, DOM-free logic for the estimator form.
 * Loaded in the browser as a classic script (exposes globalThis.HVELogic) and
 * executed by tests/js/form-logic.test.mjs under Node.
 * Rules mirror app/services/validation.py; the schema comes from app/features.py.
 */
(function (root) {
  "use strict";

  function isBlank(raw) {
    return raw === null || raw === undefined || String(raw).trim() === "";
  }

  function fmtNum(n) {
    return Number(n).toLocaleString("en-US", { maximumFractionDigits: 6 });
  }

  /** Parse one raw string value -> { value, error } using a feature definition. */
  function parseField(feature, raw) {
    if (isBlank(raw)) return { value: null, error: "This value is required." };
    var text = String(raw).trim();
    if (!/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(text)) {
      return { value: null, error: "Must be a number." };
    }
    var value = Number(text);
    if (!Number.isFinite(value)) return { value: null, error: "Must be a finite number." };
    if ((feature.dtype === "int" || feature.dtype === "binary") && !Number.isInteger(value)) {
      return { value: null, error: "Must be a whole number." };
    }
    if (feature.dtype === "binary" && value !== 0 && value !== 1) {
      return { value: null, error: "Must be 0 or 1." };
    }
    if (value < feature.min || value > feature.max) {
      return { value: null, error: "Must be between " + fmtNum(feature.min) + " and " + fmtNum(feature.max) + "." };
    }
    return { value: value, error: null };
  }

  /** Validate every feature. rawValues: { NAME: string }. */
  function validateAll(features, rawValues) {
    var values = {};
    var errors = {};
    features.forEach(function (f) {
      var r = parseField(f, rawValues[f.name]);
      if (r.error) errors[f.name] = r.error;
      else values[f.name] = r.value;
    });
    return { values: values, errors: errors, valid: Object.keys(errors).length === 0 };
  }

  /** Payload in model feature order (object key order follows the schema). */
  function buildPayload(features, values) {
    var out = {};
    features.forEach(function (f) { out[f.name] = values[f.name]; });
    return { features: out };
  }

  function isOutsideTraining(feature, value) {
    if (typeof feature.train_min !== "number" || typeof feature.train_max !== "number") return false;
    return value < feature.train_min || value > feature.train_max;
  }

  function formatUSD(thousands) {
    return "$" + Math.round(thousands * 1000).toLocaleString("en-US");
  }

  /** Position (0-100 %) of value on a [min, max] scale, clamped. */
  function percentOnScale(value, min, max) {
    if (!(max > min)) return 0;
    var p = ((value - min) / (max - min)) * 100;
    return Math.max(0, Math.min(100, p));
  }

  /** Bar geometry for a z-score on a symmetric [-limit, +limit] axis. */
  function zBar(z, limit) {
    limit = limit || 3;
    var clamped = Math.max(-limit, Math.min(limit, z));
    var width = (Math.abs(clamped) / limit) * 50;
    return {
      left: clamped >= 0 ? 50 : 50 - width,
      width: width,
      direction: z >= 0 ? "pos" : "neg",
      clipped: Math.abs(z) > limit,
    };
  }

  /** Fresh UI state. Clear Form returns to exactly this. */
  function initialState() {
    return {
      phase: "empty",       // empty | loading | result | error
      result: null,
      error: null,
      touched: {},
      submitting: false,
    };
  }

  root.HVELogic = {
    parseField: parseField,
    validateAll: validateAll,
    buildPayload: buildPayload,
    isOutsideTraining: isOutsideTraining,
    formatUSD: formatUSD,
    percentOnScale: percentOnScale,
    zBar: zBar,
    initialState: initialState,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
