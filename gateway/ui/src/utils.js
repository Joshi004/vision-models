import { useEffect, useRef, useState } from 'react';
import { listJobs, searchJobs } from './api.js';
import { rememberSubmittedJob } from './autoCache.js';
import { listCachedJobIds } from './resultCache.js';

// Statuses that mean "not finished yet" -- the only ones a job can still be
// cancelled from, and the only ones the once-a-second elapsed-time tick
// (usePolledJobs below) runs for. The complement (succeeded/failed) is
// "finished": the only state a job can be permanently deleted from -- see
// JobRow.jsx.
export const ACTIVE_STATUSES = new Set(['queued', 'running']);

// Why a queued job is not running yet: the API's `queue_state` values (common/capacity.py's
// QUEUE_* constants), in the order they are listed on the GPU capacity page. `short` is a column heading, `label` the
// name, `explain` one sentence for a tooltip or legend, `waitingFor` completes "waiting for ...".
export const QUEUE_STATES = [
  {
    key: 'model_limit',
    short: 'Model limit',
    label: 'Model at its limit',
    explain: 'Its model already runs the most jobs it may run at once (Qwen3.8: every request slot of the server is in use).',
    waitingFor: 'a free slot of its model',
  },
  {
    key: 'gpu_limit',
    short: 'GPU limit',
    label: 'Global GPU limit',
    explain: "The service's global GPU limit is reached.",
    waitingFor: 'the global GPU limit to free up',
  },
  {
    key: 'priority',
    short: 'Priority',
    label: 'Behind a more urgent job',
    explain: 'A more urgent job of another model is waiting for GPUs and gets the free ones first.',
    waitingFor: 'a more urgent job of another model',
  },
  {
    key: 'slurm_pending',
    short: 'Slurm',
    label: 'Waiting for Slurm GPUs',
    explain: 'Waiting for the cluster (Slurm) to allocate the GPUs.',
    waitingFor: 'the cluster to allocate GPUs',
  },
  {
    key: 'model_loading',
    short: 'Loading',
    label: 'Loading the model',
    explain: 'GPUs are allocated; the worker or server is loading the model.',
    waitingFor: 'the model to load',
  },
  {
    key: 'scheduling',
    short: 'Scheduling',
    label: 'Scheduling',
    explain: 'The scheduler has not looked at it yet (a few seconds at most).',
    waitingFor: 'the scheduler',
  },
];

export const QUEUE_STATE_BY_KEY = Object.fromEntries(QUEUE_STATES.map((state) => [state.key, state]));

const POLL_MS = 10000;

// The most cached job ids sent to the server for the Cached / Not cached
// filter -- the API's own cap on `job_ids` / `exclude_job_ids`.
const MAX_FILTER_JOB_IDS = 10000;

// Polling shared by useJobList and useJobSearch: calls `fetchJobs()` (which
// resolves to `{ jobs, total?, server_time? }`) on mount, every POLL_MS, whenever
// `fetchKey` changes (the caller's way of saying "the query is different now"),
// and whenever `refreshSignal` is bumped. Ticks `now` once a second while
// anything is still queued/running so elapsed-time text (JobTiming.jsx)
// updates smoothly between polls. `setError` is returned too, not just
// `error`, so a caller's own action (cancel, delete) that fails can report
// through this same error state.
function usePolledJobs(fetchJobs, fetchKey, refreshSignal) {
  const [jobs, setJobs] = useState([]);
  const [total, setTotal] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(null);
  // server_time (from the last successful poll) minus this browser's own
  // clock at that same moment -- added back to Date.now() every tick so a
  // skewed local clock can't throw off "Running Xm Ys" (see JobTiming.jsx).
  const [clockOffsetMs, setClockOffsetMs] = useState(0);
  const [now, setNow] = useState(() => new Date());
  const timerRef = useRef(null);
  // Always the latest fetchJobs, so the interval below never calls a stale one.
  const fetchRef = useRef(fetchJobs);
  fetchRef.current = fetchJobs;
  // Counts requests so a slow response to an older query can't overwrite the
  // result of a newer one (e.g. after a quick filter change).
  const requestCounterRef = useRef(0);

  async function refresh() {
    const requestId = ++requestCounterRef.current;
    try {
      const result = await fetchRef.current();
      if (requestId !== requestCounterRef.current) return;
      setJobs(result.jobs);
      setTotal(result.total ?? result.jobs.length);
      setError(null);
      if (result.server_time) {
        setClockOffsetMs(new Date(result.server_time).getTime() - Date.now());
      }
    } catch (err) {
      if (requestId === requestCounterRef.current) setError(err.message);
    } finally {
      if (requestId === requestCounterRef.current) setLoaded(true);
    }
  }

  useEffect(() => {
    refresh();
    timerRef.current = setInterval(refresh, POLL_MS);
    return () => clearInterval(timerRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchKey]);

  // A form submission (or a delete) bumps refreshSignal right after
  // succeeding so the change shows up immediately instead of waiting up
  // to POLL_MS. Callers with nothing meaningful to bump (JobHistory.jsx)
  // simply omit the argument -- undefined > 0 is false, so this is a no-op
  // beyond the mount-time refresh() above.
  useEffect(() => {
    if (refreshSignal > 0) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshSignal]);

  const anyActive = jobs.some((job) => ACTIVE_STATUSES.has(job.status));

  // Ticks the displayed elapsed/time-left text once a second, but only
  // while there's something queued/running to show it for -- no point
  // re-rendering every second when every job is already finished.
  useEffect(() => {
    if (!anyActive) return undefined;
    const tick = () => setNow(new Date(Date.now() + clockOffsetMs));
    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, [anyActive, clockOffsetMs]);

  return { jobs, total, loaded, error, setError, now, refresh };
}

// Used by JobsPanel.jsx (the small strip shown under every form): the
// newest `limit` jobs from GET /v1/jobs.
export function useJobList(limit, refreshSignal) {
  const { jobs, loaded, error, setError, now, refresh } = usePolledJobs(() => listJobs(limit), limit, refreshSignal);
  return { jobs, loaded, error, setError, now, refresh };
}

// Used by JobHistory.jsx: one filtered page of jobs from POST /v1/jobs/search.
// `filters` is { statuses, pipelines, source, cached, q } -- `cached` is
// 'all' | 'cached' | 'uncached' and is the one filter the server can't
// know (results are cached in this browser's IndexedDB), so it is turned
// into a list of job ids on every fetch, which keeps the page following
// the cache as results are added to it or evicted. Resolves `total` too,
// the number of matching jobs across all pages.
export function useJobSearch(filters, offset, limit) {
  async function fetchPage() {
    // An empty list of pipelines (the chosen model/media combination has no recipe) matches nothing.
    if (filters.pipelines && filters.pipelines.length === 0) return { jobs: [], total: 0 };
    const body = { offset, limit };
    if (filters.statuses?.length) body.statuses = filters.statuses;
    if (filters.pipelines) body.pipelines = filters.pipelines;
    if (filters.source && filters.source !== 'all') body.source = filters.source;
    if (filters.q) body.q = filters.q;
    if (filters.cached === 'cached') {
      body.job_ids = (await listCachedJobIds()).slice(0, MAX_FILTER_JOB_IDS);
    } else if (filters.cached === 'uncached') {
      body.exclude_job_ids = (await listCachedJobIds()).slice(0, MAX_FILTER_JOB_IDS);
    }
    return searchJobs(body);
  }

  return usePolledJobs(fetchPage, JSON.stringify({ filters, offset, limit }), undefined);
}

// Shared submit-a-job state machine used by every recipe form: tracks
// in-flight/error/last-job_id state so each form only needs to build its
// own payload and call submit(() => postJSON(path, payload)). Every
// submitted job_id is also remembered so its result is downloaded into the
// local cache automatically once it succeeds -- see autoCache.js.
export function useSubmitJob(onSubmitted) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [lastJobId, setLastJobId] = useState(null);
  // The whole response of the last successful submit: for the backends that size their jobs (H3) it also says
  // how many GPUs the job got and its estimated run time (`gpus`, `estimated_run_seconds`).
  const [lastResult, setLastResult] = useState(null);

  async function submit(submitFn) {
    setSubmitting(true);
    setError(null);
    try {
      const result = await submitFn();
      rememberSubmittedJob(result.job_id);
      setLastJobId(result.job_id);
      setLastResult(result);
      if (onSubmitted) onSubmitted();
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setSubmitting(false);
    }
  }

  return { submit, submitting, error, lastJobId, lastResult };
}

// One row of ImageConditioningList, matching the API's ImageConditioning
// shape (services/ltx/schemas.py) plus `asset` -- the same
// null | { status: 'uploading', filename, sentBytes, totalBytes }
// | { status: 'ready', assetId, filename, sizeBytes }
// status AssetUpload.jsx reports for a single file field. See
// assetBlocker/imageListBlockers below for how forms turn that into a
// submit-blocker message.
export function emptyImageEntry() {
  return { asset: null, frameIdx: 0, strength: 1, crf: '' };
}

export function toImageConditioningPayload(entries) {
  return entries
    .filter((entry) => entry.asset?.status === 'ready')
    .map((entry) => {
      const payload = {
        asset_id: entry.asset.assetId,
        frame_idx: entry.frameIdx === '' ? 0 : Number(entry.frameIdx),
        strength: entry.strength === '' ? 1 : Number(entry.strength),
      };
      if (entry.crf !== '' && entry.crf !== undefined && entry.crf !== null) {
        payload.crf = Number(entry.crf);
      }
      return payload;
    });
}

// Rebuilds the { status: 'ready', assetId, ... } shape AssetUpload reports
// for an already-uploaded file, given just the asset_id a previously
// submitted job's own stored request recorded -- used by the Retry flow
// (see App.jsx/JobRow.jsx) to prefill a single-file field without
// re-uploading. `filename`/`sizeBytes` are unknown at this point (the
// stored request never kept them), so both are left null; `reused: true`
// is what AssetUpload.jsx checks to show a "reusing the original file"
// message instead of a normal upload summary. Returns null when there's no
// asset_id to reuse, matching AssetUpload's own "nothing chosen" value.
export function reusedAsset(assetId) {
  if (!assetId) return null;
  return { status: 'ready', assetId, filename: null, sizeBytes: null, reused: true };
}

// The inverse of toImageConditioningPayload -- rebuilds ImageConditioningList
// entries from a stored request's `images`/`keyframes` array (see Retry in
// App.jsx). Each entry's file is reused via reusedAsset above rather than
// re-uploaded. Returns [] for anything that isn't a non-empty array
// (including undefined, e.g. a fresh form with no initialRequest), so
// callers can use this unconditionally instead of checking first.
export function imageEntriesFromRequest(list) {
  if (!Array.isArray(list)) return [];
  return list.map((entry) => ({
    asset: reusedAsset(entry.asset_id),
    frameIdx: entry.frame_idx ?? 0,
    strength: entry.strength ?? 1,
    crf: entry.crf ?? '',
  }));
}

// Submit-blocker message for one required single-file field, given its
// AssetUpload status. [] once it's ready to use, so callers can always
// spread the result straight into a combined blockers array.
export function assetBlocker(asset, label) {
  if (!asset) return [`Upload ${label}.`];
  if (asset.status === 'uploading') return [`Wait for ${label} to finish uploading.`];
  return [];
}

// Submit-blocker messages for a whole ImageConditioningList: one per row
// that isn't ready yet (naming which row, so it's actionable), plus a
// count check for the case where every present row is ready but there
// simply aren't enough of them (e.g. a keyframe row was removed).
// `minReady` is normally read live from the schema -- see
// schema.js's useArrayMinItems -- so it can't drift from what the API
// itself requires (e.g. keyframes' >=2 rule).
export function imageListBlockers(entries, label, minReady = 0) {
  const messages = [];
  entries.forEach((entry, index) => {
    if (entry.asset?.status === 'uploading') {
      messages.push(`${label} ${index + 1} is still uploading.`);
    } else if (!entry.asset) {
      messages.push(`${label} ${index + 1} has no file: upload one or remove the row.`);
    }
  });
  if (entries.length < minReady) {
    messages.push(`${label}: at least ${minReady} required, only ${entries.length} added.`);
  }
  return messages;
}

// One row of AudioFileList (components/AudioFileList.jsx), matching
// BatchConvertRequest's own `sources` shape (services/rvc/schemas.py's
// BatchSource: just one asset_id per entry, unlike ImageConditioningList's
// richer per-row frame_idx/strength/crf) plus `asset` -- the same
// null | { status: 'uploading', ... } | { status: 'ready', ... } shape
// AssetUpload reports for a single file field.
export function emptyAudioEntry() {
  return { asset: null };
}

// The inverse pairing of toImageConditioningPayload/imageEntriesFromRequest
// above, for AudioFileList instead of ImageConditioningList.
export function toBatchSourcesPayload(entries) {
  return entries.filter((entry) => entry.asset?.status === 'ready').map((entry) => ({ asset_id: entry.asset.assetId }));
}

// Rebuilds AudioFileList entries from a stored request's `sources` array
// (see Retry in App.jsx). Each entry's file is reused via reusedAsset
// above rather than re-uploaded. Returns [] for anything that isn't a
// non-empty array (including undefined, e.g. a fresh form with no
// initialRequest), so callers can use this unconditionally instead of
// checking first.
export function audioEntriesFromRequest(sources) {
  if (!Array.isArray(sources)) return [];
  return sources.map((entry) => ({ asset: reusedAsset(entry.asset_id) }));
}

// Submit-blocker messages for a whole AudioFileList -- same shape as
// imageListBlockers above (one message per row that isn't ready yet, plus
// a count check against `minReady`), just without any per-row
// frame_idx/strength/crf to ever need checking. `minReady` is normally
// read live from the schema (BatchConvertRequest.sources' own
// min_length=1) -- see schema.js's useArrayMinItems -- same reasoning as
// imageListBlockers' own minReady.
export function audioListBlockers(entries, label, minReady = 0) {
  const messages = [];
  entries.forEach((entry, index) => {
    if (entry.asset?.status === 'uploading') {
      messages.push(`${label} ${index + 1} is still uploading.`);
    } else if (!entry.asset) {
      messages.push(`${label} ${index + 1} has no file: upload one or remove the row.`);
    }
  });
  if (entries.length < minReady) {
    messages.push(`${label}: at least ${minReady} required, only ${entries.length} added.`);
  }
  return messages;
}

// Mirrors VideoShapeMixin (services/ltx/schemas.py): exactly one of
// orientation or explicit height/width, and exactly one of duration_seconds
// or num_frames. Modeled here as two independent toggles so the UI
// naturally can't build the invalid "both" combination the API would
// otherwise reject with a 422.
export function defaultShape(frameRate = 24) {
  return {
    shapeMode: 'orientation',
    orientation: 'landscape',
    height: '',
    width: '',
    lengthMode: 'default',
    durationSeconds: 5,
    numFrames: 121,
    frameRate,
  };
}

// A blank field is left out of the payload entirely (JSON.stringify drops
// undefined-valued keys) so the API's own default applies, rather than
// silently sending 0 -- e.g. a cleared seed used to submit as seed: 0
// instead of falling back to the API's actual default of 10.
export function toShapePayload(shape) {
  const payload = { frame_rate: shape.frameRate === '' ? undefined : Number(shape.frameRate) };
  if (shape.shapeMode === 'orientation') {
    payload.orientation = shape.orientation;
  } else {
    payload.height = shape.height === '' ? undefined : Number(shape.height);
    payload.width = shape.width === '' ? undefined : Number(shape.width);
  }
  if (shape.lengthMode === 'duration') {
    payload.duration_seconds = shape.durationSeconds === '' ? undefined : Number(shape.durationSeconds);
  } else if (shape.lengthMode === 'frames') {
    payload.num_frames = shape.numFrames === '' ? undefined : Number(shape.numFrames);
  }
  return payload;
}

// The inverse of toShapePayload -- rebuilds the shapeMode/lengthMode toggle
// state from a stored request's plain height/width/orientation/
// duration_seconds/num_frames/frame_rate fields (see Retry in App.jsx).
// Starts from defaultShape() so any field the stored request left unset
// (e.g. it used 'orientation', so height/width are both null) falls back
// to the same default a fresh form would show, rather than to null/blank.
export function shapeFromRequest(request, frameRate = 24) {
  const shape = defaultShape(frameRate);
  if (request.frame_rate != null) shape.frameRate = request.frame_rate;
  if (request.height != null && request.width != null) {
    shape.shapeMode = 'custom';
    shape.height = request.height;
    shape.width = request.width;
  } else if (request.orientation) {
    shape.orientation = request.orientation;
  }
  if (request.duration_seconds != null) {
    shape.lengthMode = 'duration';
    shape.durationSeconds = request.duration_seconds;
  } else if (request.num_frames != null) {
    shape.lengthMode = 'frames';
    shape.numFrames = request.num_frames;
  }
  return shape;
}

// Submit-blocker messages for the UI-only rule that height/width (in
// custom shape mode) or duration_seconds/num_frames (in the matching
// length mode) become required only once that mode is chosen -- the
// schema itself marks all four as optional, since either alternative is
// valid on its own. See components/VideoShapeFields.jsx's matching
// `required` props (forced true only in the selected mode).
export function shapeBlockers(shape) {
  const messages = [];
  if (shape.shapeMode === 'custom') {
    if (shape.height === '') messages.push('height is required when using custom height/width.');
    if (shape.width === '') messages.push('width is required when using custom height/width.');
  }
  if (shape.lengthMode === 'duration' && shape.durationSeconds === '') {
    messages.push('duration_seconds is required when "Duration (seconds)" is selected.');
  }
  if (shape.lengthMode === 'frames' && shape.numFrames === '') {
    messages.push('num_frames is required when "Exact frame count" is selected.');
  }
  return messages;
}

// "12m 03s" / "1h 04m" / "45s" style formatting for elapsed/typical/ETA
// times throughout the Jobs panel (see components/JobTiming.jsx). Returns
// null for anything that isn't a real non-negative number, so callers can
// use `formatDuration(x) ?? fallback` instead of every call site handling
// NaN/negative itself.
export function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  const totalSeconds = Math.round(seconds);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, '0')}m`;
  if (minutes > 0) return `${minutes}m ${String(secs).padStart(2, '0')}s`;
  return `${secs}s`;
}

// "412 / 1,024 MB" style formatting for ProgressBar and AssetUpload's
// ready-state message.
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const decimals = unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(decimals)} ${units[unitIndex]}`;
}
