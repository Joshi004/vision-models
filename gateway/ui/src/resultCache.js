// resultCache.js
//
// Persists downloaded job results (video/audio blobs) in IndexedDB so a
// result fetched once stays instantly available -- across collapsing and
// re-expanding a job row, switching between the Jobs panel and Job
// History, and even a full page reload -- instead of re-downloading the
// same file from the cluster every time "Load preview" is clicked. See
// components/ResultPreview.jsx, the only reader/writer of the cache
// itself, and components/JobRow.jsx, which reads useIsResultCached below
// to show a "Cached" badge without needing ResultPreview mounted at all.
//
// IndexedDB, not localStorage/sessionStorage (see storage.js) -- results
// are binary blobs that can be tens of MB, well beyond localStorage's
// ~5-10MB string-only quota, and IndexedDB stores Blobs natively.
//
// Every exported function fails soft: if IndexedDB is unavailable
// (disabled, private-mode restrictions, an old browser) or a request
// errors for any reason, reads resolve to "not cached" and writes/deletes
// just no-op. This is a cache -- losing it should never break the
// underlying fetch-on-demand preview that already works without it.

import { useEffect, useState } from 'react';

const DB_NAME = 'model-api-results';
const DB_VERSION = 1;
const STORE_NAME = 'results';

// Soft cap on total cached bytes -- once a write would push the store over
// this, the oldest entries (by cachedAt) are evicted first until it fits.
// 500MB comfortably holds a few dozen "tens of MB" results (see
// ResultPreview.jsx) without silently claiming unbounded disk space.
const MAX_CACHE_BYTES = 500 * 1024 * 1024;

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error('IndexedDB is not available.'));
      return;
    }
    const request = window.indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(STORE_NAME, { keyPath: 'jobId' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Failed to open result cache database.'));
  });
  // A failed open shouldn't stay cached forever -- clear it so a later
  // call (e.g. a transient error, or storage becoming available again)
  // can try again instead of failing for the rest of the page's life.
  dbPromise.catch(() => {
    dbPromise = null;
  });
  return dbPromise;
}

// --- change notifications ---------------------------------------------
//
// ResultPreview (the only writer) and JobRow's "Cached" badge (a reader,
// but a sibling component that stays mounted while ResultPreview itself
// mounts/unmounts with the row's expand state) need to agree on cached
// state without polling IndexedDB repeatedly. This tiny pub/sub bridges
// that: every write/delete/eviction below calls notifyChange, and
// useIsResultCached (bottom of this file) subscribes to it.
const listeners = new Set();

function notifyChange(jobId, cached) {
  for (const listener of listeners) listener(jobId, cached);
}

// --- public API ----------------------------------------------------------

// True if jobId's result is already cached -- a lightweight key-only
// lookup (no blob bytes touched) suitable for checking many jobs at once,
// e.g. once per visible row.
export async function hasCachedResult(jobId) {
  try {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const request = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).getKey(jobId);
      request.onsuccess = () => resolve(request.result !== undefined);
      request.onerror = () => reject(request.error);
    });
  } catch {
    return false;
  }
}

// The cached Blob for jobId, or null if it isn't cached (or the cache is
// unavailable). The blob's own `.type` already carries its content type
// (set when it was first constructed in api.js's fetchResultBlob), so
// there's no separate content-type field to track here.
export async function getCachedResult(jobId) {
  try {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const request = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(jobId);
      request.onsuccess = () => resolve(request.result ? request.result.blob : null);
      request.onerror = () => reject(request.error);
    });
  } catch {
    return null;
  }
}

// Stores blob under jobId, then evicts the oldest cached entries (if any)
// until the total stays under MAX_CACHE_BYTES. Best-effort throughout --
// ResultPreview already has the blob in hand for this render regardless
// of whether caching it for next time actually succeeds.
export async function putCachedResult(jobId, blob) {
  try {
    const db = await openDb();
    const record = { jobId, blob, size: blob.size, cachedAt: Date.now() };
    await new Promise((resolve, reject) => {
      const request = db.transaction(STORE_NAME, 'readwrite').objectStore(STORE_NAME).put(record);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
    notifyChange(jobId, true);
    await evictOldestUntilUnderCap(db);
  } catch {
    // See module docstring -- caching is best-effort.
  }
}

// Removes jobId's cached result, if any -- called once a job itself is
// permanently deleted (see JobHistory.jsx's handleDelete) so a purged
// job's bytes don't linger locally and its "Cached" badge disappears.
export async function deleteCachedResult(jobId) {
  try {
    const db = await openDb();
    await new Promise((resolve, reject) => {
      const request = db.transaction(STORE_NAME, 'readwrite').objectStore(STORE_NAME).delete(jobId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
    notifyChange(jobId, false);
  } catch {
    // See module docstring -- caching is best-effort.
  }
}

// Deletes oldest-first (by cachedAt) until total stored bytes fit under
// MAX_CACHE_BYTES. Reads every record's metadata (via getAll) to total
// their sizes -- expected cache sizes here are a handful to a few dozen
// entries, not thousands, so a full scan on each write stays cheap and
// keeps this simple rather than maintaining a running total separately.
async function evictOldestUntilUnderCap(db) {
  const records = await new Promise((resolve, reject) => {
    const request = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  let totalBytes = records.reduce((sum, record) => sum + record.size, 0);
  if (totalBytes <= MAX_CACHE_BYTES) return;

  const oldestFirst = [...records].sort((a, b) => a.cachedAt - b.cachedAt);
  const store = db.transaction(STORE_NAME, 'readwrite').objectStore(STORE_NAME);
  for (const record of oldestFirst) {
    if (totalBytes <= MAX_CACHE_BYTES) break;
    store.delete(record.jobId);
    totalBytes -= record.size;
    notifyChange(record.jobId, false);
  }
}

// True once jobId's result is cached, updating live if it gets cached (or
// evicted/deleted) while this stays mounted -- see JobRow.jsx, the only
// caller, which shows a "Cached" badge in the row's always-visible
// summary line so it stays correct even while the row is collapsed and
// ResultPreview isn't mounted. `resultReady` gates the check entirely --
// no point querying IndexedDB for a job that can't have a cached result
// yet (or ever, if it failed).
export function useIsResultCached(jobId, resultReady) {
  const [cached, setCached] = useState(false);

  useEffect(() => {
    if (!resultReady) {
      setCached(false);
      return undefined;
    }
    let active = true;
    function handleChange(changedJobId, isCached) {
      if (changedJobId === jobId) setCached(isCached);
    }
    hasCachedResult(jobId).then((value) => {
      if (active) setCached(value);
    });
    listeners.add(handleChange);
    return () => {
      active = false;
      listeners.delete(handleChange);
    };
  }, [jobId, resultReady]);

  return cached;
}
