// Node test for app/static/js/form-logic.js (no dependencies).
// Run: node tests/js/form-logic.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import vm from "node:vm";

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, "..", "..", "app", "static", "js", "form-logic.js"), "utf8");
const ctx = {};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(src, ctx);
const L = ctx.HVELogic;

const F = {
  RM: { name: "RM", dtype: "float", min: 1, max: 15, train_min: 3.561, train_max: 8.725 },
  RAD: { name: "RAD", dtype: "int", min: 1, max: 24 },
  CHAS: { name: "CHAS", dtype: "binary", min: 0, max: 1 },
};
const FEATURES = [F.RM, F.RAD, F.CHAS];

let n = 0;
const t = (name, fn) => { fn(); n++; };

t("required", () => {
  for (const blank of ["", "  ", null, undefined]) assert.equal(L.parseField(F.RM, blank).error, "This value is required.");
});
t("numbers", () => {
  assert.deepEqual({ ...L.parseField(F.RM, "6.5") }, { value: 6.5, error: null });
  assert.equal(L.parseField(F.RM, " 7 ").value, 7);
  assert.equal(L.parseField(F.RM, "1e0").value, 1);
  for (const bad of ["abc", "6,5", "6.5.1", "invalid", "Infinity", "0x10"]) {
    assert.equal(L.parseField(F.RM, bad).error, "Must be a number.", bad);
  }
});
t("range", () => {
  assert.match(L.parseField(F.RM, "0.5").error, /between 1 and 15/);
  assert.match(L.parseField(F.RM, "16").error, /between 1 and 15/);
  assert.equal(L.parseField(F.RM, "15").error, null);
});
t("integers and binary", () => {
  assert.equal(L.parseField(F.RAD, "2.5").error, "Must be a whole number.");
  assert.equal(L.parseField(F.RAD, "24").value, 24);
  assert.equal(L.parseField(F.CHAS, "1").value, 1);
  assert.equal(L.parseField(F.CHAS, "0.5").error, "Must be a whole number.");
  assert.match(L.parseField(F.CHAS, "2").error, /0 or 1|between 0 and 1/);
});
t("validateAll + payload order", () => {
  const bad = L.validateAll(FEATURES, { RM: "6", RAD: "", CHAS: "1" });
  assert.equal(bad.valid, false);
  assert.deepEqual(Object.keys(bad.errors), ["RAD"]);
  const ok = L.validateAll(FEATURES, { CHAS: "0", RAD: "4", RM: "6.2" });
  assert.equal(ok.valid, true);
  const payload = L.buildPayload(FEATURES, ok.values);
  assert.deepEqual(Object.keys(payload.features), ["RM", "RAD", "CHAS"]);
  assert.equal(JSON.stringify(payload), '{"features":{"RM":6.2,"RAD":4,"CHAS":0}}');
});
t("training range flag", () => {
  assert.equal(L.isOutsideTraining(F.RM, 9), true);
  assert.equal(L.isOutsideTraining(F.RM, 6), false);
  assert.equal(L.isOutsideTraining(F.RAD, 99), false); // no training range info
});
t("formatting + scale", () => {
  assert.equal(L.formatUSD(28.0736), "$28,074");
  assert.equal(L.formatUSD(5), "$5,000");
  assert.equal(L.percentOnScale(27.5, 5, 50), 50);
  assert.equal(L.percentOnScale(-3, 5, 50), 0);
  assert.equal(L.percentOnScale(80, 5, 50), 100);
});
t("z bars", () => {
  assert.deepEqual({ ...L.zBar(1.5, 3) }, { left: 50, width: 25, direction: "pos", clipped: false });
  assert.deepEqual({ ...L.zBar(-3, 3) }, { left: 0, width: 50, direction: "neg", clipped: false });
  assert.equal(L.zBar(7, 3).clipped, true);
  assert.equal(L.zBar(7, 3).width, 50);
});
t("initial state (Clear Form target)", () => {
  const s = L.initialState();
  assert.equal(s.phase, "empty");
  assert.equal(s.result, null);
  assert.equal(s.error, null);
  assert.equal(s.submitting, false);
  assert.deepEqual({ ...s.touched }, {});
  assert.notEqual(L.initialState(), s); // fresh object each time
});

console.log(`all form-logic tests passed (${n} groups)`);
