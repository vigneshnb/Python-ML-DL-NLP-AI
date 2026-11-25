// Pure, DOM-free UI state logic. Imported by app.js and unit-tested with Node
// (tests/js/state.test.mjs) so Clear/Reset behaviour is verified automatically.
//
// status: idle -> ready -> loading -> success | error
//   idle    : nothing selected (initial UI / after Clear)
//   ready   : a valid image is selected, not yet predicted
//   loading : request in flight (Predict disabled)
//   success : result available
//   error   : the prediction request failed (Predict re-enabled to retry)

export const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/webp"];
export const ALLOWED_EXT = ["png", "jpg", "jpeg", "webp"];
export const HISTORY_LIMIT = 8;

export function createInitialState() {
  return {
    file: null,          // File currently selected
    previewUrl: null,    // object URL for preview (must be revoked on reset)
    meta: null,          // { name, size, type, width, height }
    error: null,         // file validation error (blocks prediction)
    apiError: null,      // prediction request error (retry allowed)
    status: "idle",
    result: null,        // API data payload
  };
}

/** Clear everything about the current image + prediction. History is kept. */
export function resetState() {
  return createInitialState();
}

export function canPredict(state) {
  return state.file !== null && state.error === null && state.status !== "loading";
}

/** Which view the result panel should show. */
export function resultView(state) {
  if (state.status === "loading") return "loading";
  if (state.status === "success" && state.result) return "success";
  if (state.status === "error") return "error";
  return "empty";
}

/** Validate a File-like {name, type, size}. Returns an error string or null. */
export function validateFile(file, { maxBytes }) {
  if (!file) return "Please choose an image to upload.";
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  const typeOk = !file.type || ALLOWED_TYPES.includes(file.type);
  if (!ALLOWED_EXT.includes(ext) || !typeOk) {
    return "Unsupported file type. Please upload a PNG, JPG, JPEG or WEBP image.";
  }
  if (file.size === 0) return "The selected file is empty.";
  if (file.size > maxBytes) {
    return `File is too large. Maximum size is ${formatBytes(maxBytes)}.`;
  }
  return null;
}

export function validateDimensions(width, height, { minDim, maxDim }) {
  if (width < minDim || height < minDim) {
    return `Image is too small. Minimum size is ${minDim}x${minDim} pixels.`;
  }
  if (width > maxDim || height > maxDim) {
    return `Image is too large. Maximum size is ${maxDim}x${maxDim} pixels.`;
  }
  return null;
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(bytes % (1024 * 1024) === 0 ? 0 : 1)} MB`;
}

/** Short label for a MIME type / filename, e.g. "PNG". */
export function formatType(file) {
  const fromMime = (file.type || "").split("/")[1];
  const fromName = (file.name.split(".").pop() || "");
  return (fromMime || fromName).replace("jpeg", "jpg").toUpperCase();
}

/** Verbal confidence band used next to the gauge. */
export function confidenceLevel(pct) {
  if (pct >= 80) return "High";
  if (pct >= 50) return "Moderate";
  return "Low";
}

export function addHistory(history, entry, limit = HISTORY_LIMIT) {
  return [entry, ...history].slice(0, limit);
}
