// storage.js
//
// Every recipe form's field values persist two ways:
//   - sessionStorage holds a per-form "draft" of *everything* (prompt,
//     settings, and uploaded-file references), restored on refresh so a
//     reload never loses in-progress work.
//   - localStorage holds only the "settings" fields each form lists in
//     `rememberKeys` (mode/seed/shape/partition-style fields -- never
//     prompts or uploaded files), written after a successful submit and
//     used as that form's defaults the next time it's opened fresh (a new
//     tab, or after Reset) -- see useFormState below.
//
// All storage access goes through the safe helpers below: reading/writing
// browser storage can throw (disabled by the browser, private-mode quota,
// corrupted JSON left over from an older version of this UI), and none of
// that should ever crash the app -- every caller just falls back to
// whatever default it already had.
import { useEffect, useMemo, useState } from 'react';

// Bumping this discards any previously stored draft/settings that no
// longer fit -- see pickKnown below, which already guards against unknown
// keys, but a version bump is a clean way to force a fresh start after a
// bigger shape change.
const NAMESPACE = 'model-api:v1';

function namespacedKey(...parts) {
  return [NAMESPACE, ...parts].join(':');
}

function safeGet(storage, key) {
  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(storage, key, value) {
  try {
    storage.setItem(key, value);
  } catch {
    // Disabled storage, private-mode quota, etc. -- the in-memory React
    // state this backs still works for the rest of the session either way.
  }
}

function safeRemove(storage, key) {
  try {
    storage.removeItem(key);
  } catch {
    // See safeSet.
  }
}

function readJSON(storage, key, fallback) {
  const raw = safeGet(storage, key);
  if (raw == null) return fallback;
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function writeJSON(storage, key, value) {
  try {
    safeSet(storage, key, JSON.stringify(value));
  } catch {
    // JSON.stringify shouldn't throw for the plain objects/arrays/strings/
    // numbers/booleans every form's state is made of, but fall back to a
    // no-op rather than crash if it ever does.
  }
}

// Like useState, but the initial value is read from `storage` once, at
// mount, and every change is written back. `validate` optionally checks a
// restored value is still acceptable (e.g. a stored recipe id that no
// longer exists) -- falls back to `fallback` when it isn't. Used for
// app-level UI state: the active recipe, each form's Advanced-section
// open/closed state, job-history filters (see App.jsx/forms/*.jsx/
// components/JobHistory.jsx).
export function usePersistedState(key, fallback, options = {}) {
  const { storage = window.localStorage, validate } = options;
  const fullKey = namespacedKey(key);
  const [value, setValue] = useState(() => {
    const restored = readJSON(storage, fullKey, fallback);
    if (validate && !validate(restored)) return fallback;
    return restored;
  });

  useEffect(() => {
    writeJSON(storage, fullKey, value);
    // `storage`/`fullKey` are constant for a given call site.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return [value, setValue];
}

// Only copies over keys `defaults` actually declares -- an older stored
// draft/settings object (from a previous version of a form: extra fields,
// renamed fields) can't reintroduce a field the current form no longer
// has.
function pickKnown(source, defaults) {
  const result = {};
  if (!source || typeof source !== 'object') return result;
  for (const key of Object.keys(defaults)) {
    if (key in source) result[key] = source[key];
  }
  return result;
}

function isUploadingAsset(value) {
  return Boolean(value) && typeof value === 'object' && value.status === 'uploading';
}

// A file mid-upload can't resume after a refresh (the browser doesn't
// keep the actual file around for us) -- any field restored from storage
// still showing `status: 'uploading'` reverts to null (or, inside an
// image/keyframe list entry, that entry's own `asset` field) so the form
// never shows a permanently-stuck progress bar after a reload.
function clearStaleUploads(values) {
  const cleaned = { ...values };
  for (const [key, value] of Object.entries(cleaned)) {
    if (isUploadingAsset(value)) {
      cleaned[key] = null;
    } else if (Array.isArray(value)) {
      cleaned[key] = value.map((entry) =>
        entry && typeof entry === 'object' && isUploadingAsset(entry.asset) ? { ...entry, asset: null } : entry,
      );
    }
  }
  return cleaned;
}

// Backs every recipe form's field state (see forms/*.jsx). On mount:
//   - if `fromRequest` is given (Retry, see App.jsx's handleRetry), its
//     values win over anything stored -- editing/resubmitting a specific
//     past job should never get mixed up with an unrelated leftover draft.
//   - otherwise, starts from `defaults`, then overlays this form's last
//     *submitted* settings from localStorage (only the `rememberKeys`
//     fields), then overlays this form's in-progress draft from
//     sessionStorage (every field) -- the draft (more recent, mid-typing
//     state) wins over last-submitted settings wherever both have a value.
// Every change re-writes the full sessionStorage draft. A successful
// submit (rememberSettings, called by FormShell) writes just the
// `rememberKeys` fields to localStorage.
export function useFormState(formId, { defaults, fromRequest = null, rememberKeys = [] }) {
  const draftKey = namespacedKey('draft', formId);
  const lastKey = namespacedKey('last', formId);

  // Computed once, at mount -- callers remount the whole form (a fresh
  // `key` in App.jsx) rather than expecting this to react to a later
  // change in `fromRequest`, matching how each form's own per-field
  // useState initializers already only ran once per mount before this
  // file existed.
  const initial = useMemo(() => {
    if (fromRequest) {
      return { values: { ...defaults, ...fromRequest }, restoredDraft: false };
    }
    const lastUsed = readJSON(window.localStorage, lastKey, null);
    const draft = readJSON(window.sessionStorage, draftKey, null);
    const merged = {
      ...defaults,
      ...pickKnown(lastUsed, defaults),
      ...pickKnown(draft, defaults),
    };
    return { values: clearStaleUploads(merged), restoredDraft: draft != null };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [values, setValues] = useState(initial.values);
  const [restoredDraft, setRestoredDraft] = useState(initial.restoredDraft);

  useEffect(() => {
    writeJSON(window.sessionStorage, draftKey, values);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values]);

  // `valueOrUpdater` may be a plain value or a `(previousFieldValue) =>
  // nextFieldValue` updater -- the latter is what ImageConditioningList
  // uses (see its updateEntry) so a fast-finishing upload in one row can't
  // clobber a slower one still in flight in another row by writing back a
  // stale copy of the whole `images`/`keyframes` array.
  function setField(name, valueOrUpdater) {
    setValues((previous) => {
      const previousFieldValue = previous[name];
      const nextFieldValue =
        typeof valueOrUpdater === 'function' ? valueOrUpdater(previousFieldValue) : valueOrUpdater;
      return { ...previous, [name]: nextFieldValue };
    });
  }

  function rememberSettings() {
    const toRemember = {};
    for (const key of rememberKeys) toRemember[key] = values[key];
    writeJSON(window.localStorage, lastKey, toRemember);
  }

  function reset() {
    setValues(defaults);
    setRestoredDraft(false);
    safeRemove(window.sessionStorage, draftKey);
    safeRemove(window.localStorage, lastKey);
  }

  return { values, setField, setValues, rememberSettings, reset, restoredDraft };
}
