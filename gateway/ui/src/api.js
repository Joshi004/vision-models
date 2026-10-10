// Thin wrapper around fetch()/XMLHttpRequest for talking to this same
// origin's /v1/... API (see vite.config.js's dev proxy for why this also
// works with `npm run dev`). No validation lives here -- the API is the
// source of truth; this module only turns FastAPI's error bodies into a
// plain message string every caller can just display.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

// FastAPI's error body is {"detail": "..."} for most errors, or
// {"detail": [{"msg": "...", ...}, ...]} for a 422 validation error --
// handle both instead of just showing "[object Object]". Shared by both
// the fetch() error path below and the XMLHttpRequest upload path (which
// can't use fetch -- see uploadFile).
function messageFromBody(body, status, statusText) {
  const detail = body && body.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join('; ');
  }
  return `${status} ${statusText}`;
}

async function parseErrorDetail(response) {
  let body = null;
  try {
    body = await response.json();
  } catch {
    return `${response.status} ${response.statusText}`;
  }
  return messageFromBody(body, response.status, response.statusText);
}

function parseXhrError(xhr) {
  let body = null;
  try {
    body = JSON.parse(xhr.responseText);
  } catch {
    return `${xhr.status} ${xhr.statusText || 'Request failed'}`;
  }
  return messageFromBody(body, xhr.status, xhr.statusText);
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response;
}

export async function getJSON(path) {
  const response = await request(path);
  return response.json();
}

export async function postJSON(path, body) {
  const response = await request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return response.json();
}

export async function deleteJSON(path) {
  const response = await request(path, { method: 'DELETE' });
  return response.json();
}

// XMLHttpRequest instead of fetch() -- fetch has no way to report upload
// progress, and uploads here can be large videos (see common/config.py's
// MAX_UPLOAD_BYTES, 2 GB by default), where a bare spinner says very
// little. `onProgress(sentBytes, totalBytes)` is called as the browser
// reports it; omit it to upload without progress reporting.
export function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/v1/uploads');
    if (onProgress) {
      xhr.upload.onprogress = (event) => {
        onProgress(event.loaded, event.lengthComputable ? event.total : file.size);
      };
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new ApiError(parseXhrError(xhr), xhr.status));
      }
    };
    xhr.onerror = () => reject(new ApiError('Network error during upload.', 0));
    const form = new FormData();
    form.append('file', file);
    xhr.send(form);
  });
}

export function getHealth() {
  return getJSON('/v1/health');
}

// Live per-partition GPU availability and queue length -- see
// PartitionField.jsx, the only caller. Polled on its own interval while a
// form is open, independent of getHealth()/listJobs()'s own polling.
export function getPartitions() {
  return getJSON('/v1/partitions');
}

// The same document /docs renders from -- component schemas (field
// descriptions, required lists) come from here, see schema.js.
export function getOpenApi() {
  return getJSON('/openapi.json');
}

export function listJobs(limit = 50) {
  return getJSON(`/v1/jobs?limit=${limit}`);
}

// Names of every currently-installed RVC voice model -- see
// services/rvc/router.py. Fetched once by RvcConvertForm.jsx to populate
// its voice dropdown, same "ask the API, don't hardcode a list that can
// drift" reasoning as getOpenApi() above.
export function listVoices() {
  return getJSON('/v1/rvc/voices');
}

export function cancelJob(jobId) {
  return deleteJSON(`/v1/jobs/${jobId}`);
}

// Permanent, unlike cancelJob above -- see JobHistory.jsx, the only caller.
export function purgeJob(jobId) {
  return deleteJSON(`/v1/jobs/${jobId}/purge`);
}

// Fetched as a blob (not a plain <video src=...>) so download progress can
// be reported and the result cached locally -- see
// components/ResultPreview.jsx. Streamed chunk-by-chunk (rather than a
// plain response.blob()) so `onProgress(receivedBytes, totalBytes)` can
// report real download progress; omit it to just await the whole blob.
export async function fetchResultBlob(jobId, onProgress) {
  const response = await request(`/v1/jobs/${jobId}/result`);
  if (!onProgress || !response.body) {
    return response.blob();
  }
  const contentType = response.headers.get('Content-Type') || undefined;
  const totalBytes = Number(response.headers.get('Content-Length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let receivedBytes = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedBytes += value.length;
    onProgress(receivedBytes, totalBytes);
  }
  return new Blob(chunks, contentType ? { type: contentType } : undefined);
}
