import { useEffect, useRef, useState } from 'react';
import { cancelJob } from '../api.js';
import { ACTIVE_STATUSES, useJobList } from '../utils.js';
import { useToast } from './ui/Toast.jsx';
import JobRow from './JobRow.jsx';

// Backs GET /v1/jobs -- shows every job this server still has a row for
// (any backend, including ones submitted by curl or an agent, not just
// from this browser tab), polling while any of them are still queued or
// running. Deliberately small (last 50, Cancel and Retry only, no Delete)
// since it's shown beside every form, not just its own page -- see
// components/JobHistory.jsx for the fuller, filterable view with
// permanent delete.
//
// `onRetry`/`canRetry` come from App.jsx unchanged -- see JobRow.jsx for
// what each does; this component has no retry logic of its own.
//
// Also raises a toast whenever a job this tab has already seen switch
// from active to finished, and briefly highlights a job's row the first
// time it's seen at all (e.g. one just submitted from this tab). Both are
// skipped on the very first load (seenRef starts at `null`), so opening
// the app never floods either for jobs that were already sitting there.
export default function JobsPanel({ refreshSignal, onRetry, canRetry }) {
  const { jobs, loaded, error, setError, now, refresh } = useJobList(50, refreshSignal);
  const [expandedId, setExpandedId] = useState(null);
  const [cancellingId, setCancellingId] = useState(null);
  const [highlightedId, setHighlightedId] = useState(null);
  const [filter, setFilter] = useState('active');
  const { addToast } = useToast();
  const seenRef = useRef(null);

  useEffect(() => {
    const seen = seenRef.current;
    if (seen) {
      for (const job of jobs) {
        const previousStatus = seen.get(job.job_id);
        if (previousStatus === undefined) {
          setHighlightedId(job.job_id);
          setTimeout(() => setHighlightedId((current) => (current === job.job_id ? null : current)), 2500);
        } else if (ACTIVE_STATUSES.has(previousStatus) && !ACTIVE_STATUSES.has(job.status)) {
          addToast({
            tone: job.status === 'succeeded' ? 'success' : 'danger',
            message: `Job ${job.job_id.slice(0, 8)}\u2026 ${job.status}.`,
          });
        }
      }
    }
    seenRef.current = new Map(jobs.map((job) => [job.job_id, job.status]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobs]);

  async function handleCancel(jobId) {
    setCancellingId(jobId);
    try {
      await cancelJob(jobId);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setCancellingId(null);
    }
  }

  const runningCount = jobs.filter((job) => job.status === 'running').length;
  const queuedCount = jobs.filter((job) => job.status === 'queued').length;
  const countSummary = [runningCount && `${runningCount} running`, queuedCount && `${queuedCount} queued`]
    .filter(Boolean)
    .join(', ');

  const visibleJobs = filter === 'active' ? jobs.filter((job) => ACTIVE_STATUSES.has(job.status)) : jobs;

  return (
    <section className="flex flex-col gap-3 rounded-lg border border-border bg-surface p-4">
      <h2 className="text-sm font-semibold text-text">Jobs{countSummary ? ` \u2014 ${countSummary}` : ''}</h2>

      <div className="flex gap-1 rounded-md border border-border bg-bg p-1 text-xs">
        <button
          type="button"
          onClick={() => setFilter('active')}
          className={`flex-1 rounded px-2 py-1 transition-colors ${filter === 'active' ? 'bg-accent text-white' : 'text-muted hover:text-text'}`}
        >
          Active
        </button>
        <button
          type="button"
          onClick={() => setFilter('all')}
          className={`flex-1 rounded px-2 py-1 transition-colors ${filter === 'all' ? 'bg-accent text-white' : 'text-muted hover:text-text'}`}
        >
          All
        </button>
      </div>

      {error && <p className="text-sm text-danger">{error}</p>}
      {!loaded && !error && <p className="text-sm text-muted">Loading jobs\u2026</p>}
      {loaded && visibleJobs.length === 0 && !error && (
        <p className="text-sm text-muted">{filter === 'active' ? 'Nothing running or queued.' : 'No jobs yet.'}</p>
      )}
      <ul className="flex flex-col gap-2">
        {visibleJobs.map((job) => (
          <JobRow
            key={job.job_id}
            job={job}
            now={now}
            expanded={expandedId === job.job_id}
            onToggleExpand={() => setExpandedId(expandedId === job.job_id ? null : job.job_id)}
            cancelling={cancellingId === job.job_id}
            onCancel={() => handleCancel(job.job_id)}
            onRetry={onRetry}
            canRetry={canRetry}
            highlighted={highlightedId === job.job_id}
          />
        ))}
      </ul>
    </section>
  );
}
