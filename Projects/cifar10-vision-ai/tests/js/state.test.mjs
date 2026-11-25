// Run: node tests/js/state.test.mjs   (also executed by pytest)
import assert from "node:assert/strict";
import { test } from "node:test";

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
} from "../../app/static/js/state.js";

const limits = { maxBytes: 5 * 1024 * 1024, minDim: 8, maxDim: 6000 };
const ready = () => ({ ...createInitialState(), file: { name: "a.png" }, status: "ready" });

test("initial UI: empty view, predict disabled", () => {
  const s = createInitialState();
  assert.equal(canPredict(s), false);
  assert.equal(resultView(s), "empty");
});

test("predict enabled after valid upload, disabled while loading or invalid", () => {
  const s = ready();
  assert.equal(canPredict(s), true);
  assert.equal(canPredict({ ...s, status: "loading" }), false);
  assert.equal(canPredict({ ...s, error: "bad" }), false);
});

test("result views: loading, success, error", () => {
  assert.equal(resultView({ ...ready(), status: "loading" }), "loading");
  assert.equal(resultView({ ...ready(), status: "success", result: { x: 1 } }), "success");
  const failed = { ...ready(), status: "error", apiError: "boom" };
  assert.equal(resultView(failed), "error");
  assert.equal(canPredict(failed), true, "retry allowed after an API error");
});

test("clear restores the initial UI completely", () => {
  const dirty = {
    file: { name: "dog.jpg" },
    previewUrl: "blob:x",
    meta: { name: "dog.jpg", size: 1, type: "JPG", width: 1, height: 1 },
    error: "some error",
    apiError: "api error",
    status: "success",
    result: { predicted_label: "Dog", confidence: 90, probabilities: {}, execution_time_ms: 5 },
  };
  const s = resetState(dirty);
  assert.deepEqual(s, createInitialState());
  assert.equal(s.file, null);
  assert.equal(s.previewUrl, null);
  assert.equal(s.meta, null);
  assert.equal(s.result, null);
  assert.equal(s.error, null);
  assert.equal(s.apiError, null);
  assert.equal(s.status, "idle");
  assert.equal(resultView(s), "empty");
  assert.equal(canPredict(s), false);
});

test("file validation", () => {
  assert.equal(validateFile({ name: "a.png", type: "image/png", size: 10 }, limits), null);
  assert.equal(validateFile({ name: "a.WEBP", type: "image/webp", size: 10 }, limits), null);
  assert.match(validateFile({ name: "a.gif", type: "image/gif", size: 10 }, limits), /Unsupported/);
  assert.match(validateFile({ name: "a.png", type: "text/plain", size: 10 }, limits), /Unsupported/);
  assert.match(validateFile({ name: "a.png", type: "image/png", size: 0 }, limits), /empty/);
  assert.match(validateFile({ name: "a.png", type: "image/png", size: limits.maxBytes + 1 }, limits), /too large/);
  assert.match(validateFile(null, limits), /choose/);
});

test("dimension validation", () => {
  assert.equal(validateDimensions(32, 32, limits), null);
  assert.match(validateDimensions(4, 100, limits), /too small/);
  assert.match(validateDimensions(7000, 100, limits), /too large/);
});

test("history keeps newest first and is capped", () => {
  let h = [];
  for (let i = 0; i < 12; i++) h = addHistory(h, { i }, 8);
  assert.equal(h.length, 8);
  assert.equal(h[0].i, 11);
});

test("formatting helpers", () => {
  assert.equal(formatBytes(500), "500 B");
  assert.equal(formatBytes(2048), "2.0 KB");
  assert.equal(formatBytes(5 * 1024 * 1024), "5 MB");
  assert.equal(formatType({ name: "x.jpeg", type: "image/jpeg" }), "JPG");
  assert.equal(formatType({ name: "x.webp", type: "" }), "WEBP");
  assert.equal(confidenceLevel(95), "High");
  assert.equal(confidenceLevel(60), "Moderate");
  assert.equal(confidenceLevel(20), "Low");
});
