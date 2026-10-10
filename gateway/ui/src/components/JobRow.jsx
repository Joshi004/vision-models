import { useState } from 'react';
import { ACTIVE_STATUSES } from '../utils.js';
import { useIsResultCached } from '../resultCache.js';
import { JobTimingDetail, JobTimingSummary } from './JobTiming.jsx';
import ResultPreview from './ResultPreview.jsx';
import Button from './ui/Button.jsx';
import Badge from './ui/Badge.jsx';
import { ChevronDownIcon, CopyIcon, TrashIcon } from './icons.jsx';

const STATUS_TONE = {
  succeeded: 'success',
  failed: 'danger',
  running: 'accent',
  queued: 'neutral',
};

// One job's row: an always-visible summary line, and (once clicked)
// expanded details -- Cancel while queued/running, a result preview once
// succeeded, a Retry (re-open the submit form prefilled with this job's own
// request, see App.jsx's handleRetry) once finished, a collapsible
// "Submitted config" showing the exact request this job was sent with, and
// a permanent Delete once finished, but only when `onDelete` is actually
// passed. Shared by JobsPanel.jsx (the small strip beside every form,
// which never passes onDelete -- see that file) and JobHistory.jsx (the
// full view, which does).
//
// `onRetry`/`canRetry` are both optional -- omit both to hide Retry
// entirely (neither current caller does this, but keeps this component
// usable without them). `canRetry(job)` decides per-job whether a matching
// form actually exists (see App.jsx); `onRetry(job)` is only ever called
// once that's already true.
//
// `highlighted` (set briefly by JobsPanel for a job it just saw appear)
// adds a short-lived accent ring so a just-submitted job is easy to spot
// in the list without having to scan for it.
export default function JobRow({
  job,
  now,
  expanded,
  onToggleExpand,
  cancelling,
  onCancel,
  deleting,
  onDelete,
  onRetry,
  canRetry,
  highlighted,
}) {
  // 'copied' | 'failed' | null -- transient feedback for the Copy JSON
  // button below, cleared automatically a couple seconds later.
  const [copyState, setCopyState] = useState(null);
  // Drives the "Cached" badge below -- true once this job's result is
  // sitting in resultCache.js's IndexedDB store, so it's visible right in
  // this always-rendered summary line even while the row is collapsed and
  // ResultPreview (the only thing that reads/writes that cache) isn't
  // mounted at all.
  const isCached = useIsResultCached(job.job_id, job.result_ready);

  async function handleCopyConfig() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(job.request, null, 2));
      setCopyState('copied');
    } catch {
      // navigator.clipboard can be unavailable (e.g. no HTTPS/localhost
      // context) or reject if the page never gained focus -- either way,
      // the JSON is still right there in the <pre> below to select by hand.
      setCopyState('failed');
    }
    setTimeout(() => setCopyState(null), 2000);
  }

  const finished = !ACTIVE_STATUSES.has(job.status);
  const showRetry = Boolean(onRetry) && finished && (canRetry ? canRetry(job) : true);

  return (
    <li
      className={`rounded-lg border bg-surface transition-shadow ${
        highlighted ? 'border-accent ring-2 ring-accent/40' : 'border-border'
      }`}
    >
      <button type="button" onClick={onToggleExpand} className="flex w-full flex-wrap items-center gap-3 px-3 py-2.5 text-left text-sm">
        <Badge tone={STATUS_TONE[job.status] ?? 'neutral'}>{job.status}</Badge>
        {isCached && <Badge tone="accent">Cached</Badge>}
        <span className="text-text">{job.pipeline}</span>
        <JobTimingSummary job={job} now={now} />
        <span className="font-mono text-xs text-muted">{job.job_id.slice(0, 8)}</span>
        {job.partition && <span className="text-xs text-muted">partition: {job.partition}</span>}
        <ChevronDownIcon className={`ml-auto h-4 w-4 shrink-0 text-muted transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>

      {expanded && (
        <div className="flex flex-col gap-3 border-t border-border px-3 py-3">
          <p className="font-mono text-xs text-muted">{job.job_id}</p>
          {job.request?.prompt && <p className="text-sm text-text">prompt: {job.request.prompt}</p>}
          {job.slurm_job_id && <p className="text-xs text-muted">slurm_job_id: {job.slurm_job_id}</p>}
          <JobTimingDetail job={job} now={now} />
          {job.error && <p className="text-sm text-danger">{job.error}</p>}

          <div className="flex flex-wrap items-center gap-2">
            {ACTIVE_STATUSES.has(job.status) && (
              <Button type="button" variant="secondary" size="sm" disabled={cancelling} onClick={onCancel}>
                {cancelling ? 'Cancelling\u2026' : 'Cancel'}
              </Button>
            )}
            {showRetry && (
              <Button type="button" variant="secondary" size="sm" onClick={() => onRetry(job)}>
                Retry (edit and resubmit)
              </Button>
            )}
            {onDelete && !ACTIVE_STATUSES.has(job.status) && (
              <Button type="button" variant="danger" size="sm" disabled={deleting} onClick={onDelete}>
                <TrashIcon className="h-3.5 w-3.5" />
                {deleting ? 'Deleting\u2026' : 'Delete'}
              </Button>
            )}
          </div>

          {job.result_ready && <ResultPreview jobId={job.job_id} pipeline={job.pipeline} request={job.request} />}

          {/* Every job's exact submitted request, for debugging a failed job
             later or just double-checking what was actually sent -- the
             same object Retry above prefills a form from, shown here
             read-only for every job regardless of whether Retry applies. */}
          <details className="text-sm">
            <summary className="cursor-pointer text-muted">Submitted config</summary>
            <p className="mt-1 text-xs text-muted">Exact request this job was sent with ({job.pipeline}):</p>
            <pre className="mt-2 max-h-[20rem] overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-bg p-3 text-xs">
              {JSON.stringify(job.request, null, 2)}
            </pre>
            <Button type="button" variant="secondary" size="sm" className="mt-2" onClick={handleCopyConfig}>
              <CopyIcon className="h-3.5 w-3.5" />
              Copy JSON
            </Button>
            {copyState === 'copied' && <span className="ml-2 text-xs text-success">Copied.</span>}
            {copyState === 'failed' && (
              <span className="ml-2 text-xs text-danger">Couldn&apos;t copy automatically -- select the text above instead.</span>
            )}
          </details>
        </div>
      )}
    </li>
  );
}
