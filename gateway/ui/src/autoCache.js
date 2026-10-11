// autoCache.js
//
// Downloads a finished job's result into the local result cache
// (resultCache.js) as soon as the job succeeds, so expanding its row shows
// the preview instantly -- but only for jobs submitted from this browser.
// The jobs list also shows jobs submitted by curl, scripts or agents, and
// downloading an incognito job's result deletes it from the server, so
// auto-downloading those would take them away from whoever submitted them.
// Every other job still loads only when its
// "Load preview" button is clicked (components/ResultPreview.jsx).
//
// Every download -- automatic, or a "Load preview" click -- goes through
// loadResult below, so the same job is never fetched twice at once (a
// second fetch of a one-time result would only get a 410).

import { useEffect } from 'react';
import { fetchResultBlob } from './api.js';
import { hasCachedResult, isCacheAvailable, putCachedResult } from './resultCache.js';

// --- jobs submitted from this browser ------------------------------------
//
// localStorage, so a job that finishes while this tab is closed is still
// downloaded the next time the UI is opened. Bounded: an id is removed once
// its result has been auto-downloaded, and only the most recent
// MAX_SUBMITTED_IDS are kept regardless.
const SUBMITTED_KEY = 'model-api:v1:submittedJobIds';
const MAX_SUBMITTED_IDS = 200;

function readSubmittedIds() {
  try {
    const ids = JSON.parse(window.localStorage.getItem(SUBMITTED_KEY) ?? '[]');
    return Array.isArray(ids) ? ids : [];
  } catch {
    return [];
  }
}

function writeSubmittedIds(ids) {
  try {
    window.localStorage.setItem(SUBMITTED_KEY, JSON.stringify(ids.slice(-MAX_SUBMITTED_IDS)));
  } catch {
    // Disabled storage or quota -- the job just won't be auto-downloaded.
  }
}

// Called by utils.js's useSubmitJob after every successful submit.
export function rememberSubmittedJob(jobId) {
  writeSubmittedIds([...readSubmittedIds().filter((id) => id !== jobId), jobId]);
}

export function forgetSubmittedJob(jobId) {
  writeSubmittedIds(readSubmittedIds().filter((id) => id !== jobId));
}

// --- shared download -----------------------------------------------------

// jobId -> { promise, progressListeners } for each download in flight.
const inflight = new Map();
// Results downloaded but not stored in the cache (IndexedDB write failed,
// or the file is bigger than the cache's whole cap) -- kept for this page's
// lifetime so a one-time download is never simply thrown away.
const uncachedBlobs = new Map();
const loadedListeners = new Set();

export function isLoadingResult(jobId) {
  return inflight.has(jobId);
}

export function getUncachedResult(jobId) {
  return uncachedBlobs.get(jobId) ?? null;
}

// `listener(jobId, blob)` runs after every finished download, automatic or
// not -- lets a mounted ResultPreview show a result the auto-download just
// fetched. Returns an unsubscribe function.
export function onResultLoaded(listener) {
  loadedListeners.add(listener);
  return () => loadedListeners.delete(listener);
}

// Downloads jobId's result and stores it in the cache, or joins the
// download already in flight for it. `onProgress(receivedBytes,
// totalBytes)` is optional.
export function loadResult(jobId, onProgress) {
  let entry = inflight.get(jobId);
  if (!entry) {
    const progressListeners = new Set();
    const promise = fetchResultBlob(jobId, (receivedBytes, totalBytes) => {
      for (const listener of progressListeners) listener(receivedBytes, totalBytes);
    })
      .then(async (blob) => {
        const stored = await putCachedResult(jobId, blob);
        if (!stored) uncachedBlobs.set(jobId, blob);
        for (const listener of loadedListeners) listener(jobId, blob);
        return blob;
      })
      .finally(() => inflight.delete(jobId));
    entry = { promise, progressListeners };
    inflight.set(jobId, entry);
  }
  if (onProgress) entry.progressListeners.add(onProgress);
  return entry.promise;
}

// --- automatic download --------------------------------------------------

// Jobs already queued for auto-download on this page, so the 10 s job-list
// poll doesn't queue them again. A failed auto-download is not retried
// automatically; its "Load preview" button still works and shows the error.
const autoAttempted = new Set();
// One download at a time, so several jobs finishing together don't compete
// for the same (often tunneled) connection.
let autoQueue = Promise.resolve();

async function autoLoad(jobId) {
  // Without a working cache the downloaded file would have nowhere to go,
  // and on a one-time-download server the server copy would be gone too.
  if (!(await isCacheAvailable())) return;
  if (!(await hasCachedResult(jobId))) await loadResult(jobId);
  forgetSubmittedJob(jobId);
}

// Called with the polled job list by JobsPanel.jsx and JobHistory.jsx (one
// of them is always on screen); autoAttempted keeps the two from queueing
// the same job twice.
export function useAutoCacheResults(jobs) {
  useEffect(() => {
    const submitted = new Set(readSubmittedIds());
    for (const job of jobs) {
      if (!submitted.has(job.job_id) || autoAttempted.has(job.job_id)) continue;
      if (job.status !== 'succeeded' || !job.result_ready || job.downloaded_at) continue;
      autoAttempted.add(job.job_id);
      autoQueue = autoQueue
        .then(() => autoLoad(job.job_id))
        .catch(() => {
          // See autoAttempted above.
        });
    }
  }, [jobs]);
}
