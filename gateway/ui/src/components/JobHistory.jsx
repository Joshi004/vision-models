import { useMemo, useState } from 'react';
import { cancelJob, purgeJob } from '../api.js';
import { deleteCachedResult } from '../resultCache.js';
import { ACTIVE_STATUSES, useJobList } from '../utils.js';
import { usePersistedState } from '../storage.js';
import Input from './ui/Input.jsx';
import Select from './ui/Select.jsx';
import JobRow from './JobRow.jsx';

const STATUS_OPTIONS = [
  { value: 'all', label: 'All' },
  { value: 'active', label: 'In progress' },
  { value: 'succeeded', label: 'Succeeded' },
  { value: 'failed', label: 'Failed' },
];

// The full job history: every job this server still has a row for (up to
// the API's own max of 500 -- see api.js's listJobs), independent of
// JobsPanel.jsx's smaller, always-visible strip, with recipe/status/text
// filters and a permanent Delete per finished job
// (DELETE /v1/jobs/{job_id}/purge, via JobRow's onDelete). Filters persist
// in sessionStorage (see storage.js) so navigating away and back (or a
// refresh) doesn't silently reset them.
//
// `onRetry`/`canRetry` come from App.jsx unchanged -- see JobRow.jsx for
// what each does; this component has no retry logic of its own.
export default function JobHistory({ jobRetentionDays, onRetry, canRetry }) {
  const { jobs, loaded, error, setError, now, refresh } = useJobList(500);
  const [expandedId, setExpandedId] = useState(null);
  const [cancellingId, setCancellingId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [deleteNote, setDeleteNote] = useState(null);
  const [pipelineFilter, setPipelineFilter] = usePersistedState('historyFilter:pipeline', 'all', {
    storage: window.sessionStorage,
  });
  const [statusFilter, setStatusFilter] = usePersistedState('historyFilter:status', 'all', {
    storage: window.sessionStorage,
  });
  const [searchText, setSearchText] = usePersistedState('historyFilter:search', '', { storage: window.sessionStorage });

  const pipelines = useMemo(() => Array.from(new Set(jobs.map((job) => job.pipeline))).sort(), [jobs]);

  const filteredJobs = jobs.filter((job) => {
    if (pipelineFilter !== 'all' && job.pipeline !== pipelineFilter) return false;
    if (statusFilter === 'active' && !ACTIVE_STATUSES.has(job.status)) return false;
    if (statusFilter !== 'all' && statusFilter !== 'active' && job.status !== statusFilter) return false;
    if (searchText.trim()) {
      const needle = searchText.trim().toLowerCase();
      const haystack = `${job.job_id} ${job.request?.prompt ?? ''}`.toLowerCase();
      if (!haystack.includes(needle)) return false;
    }
    return true;
  });

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

  async function handleDelete(job) {
    const confirmed = window.confirm(
      `Permanently delete job ${job.job_id} (${job.pipeline})?\n\n` +
        'This removes its output, logs, and any intermediate files, plus any uploaded ' +
        'input no other job still uses. This cannot be undone.',
    );
    if (!confirmed) return;
    setDeletingId(job.job_id);
    setDeleteNote(null);
    setError(null);
    try {
      const result = await purgeJob(job.job_id);
      await deleteCachedResult(job.job_id);
      const removedCount = result.removed_asset_ids.length;
      const keptCount = result.kept_asset_ids.length;
      const parts = [`Deleted job ${job.job_id}.`];
      if (removedCount > 0) parts.push(`Removed ${removedCount} uploaded input${removedCount === 1 ? '' : 's'}.`);
      if (keptCount > 0) parts.push(`Kept ${keptCount} still used by another job.`);
      setDeleteNote(parts.join(' '));
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <section className="mx-auto flex max-w-4xl flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold text-text">Job history</h2>
        <p className="text-sm text-muted">
          Everything this server still has, newest first. Finished jobs are removed automatically
          {jobRetentionDays != null ? ` ${jobRetentionDays}` : ''} days after they finish.
        </p>
      </div>

      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col gap-1 text-xs text-muted">
          Recipe
          <Select value={pipelineFilter} onChange={(event) => setPipelineFilter(event.target.value)} className="min-w-[10rem]">
            <option value="all">All</option>
            {pipelines.map((pipeline) => (
              <option key={pipeline} value={pipeline}>
                {pipeline}
              </option>
            ))}
          </Select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Status
          <Select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} className="min-w-[8rem]">
            {STATUS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
        </label>
        <label className="flex min-w-[12rem] flex-1 flex-col gap-1 text-xs text-muted">
          Search
          <Input
            type="text"
            value={searchText}
            onChange={(event) => setSearchText(event.target.value)}
            placeholder="Job ID or prompt text\u2026"
          />
        </label>
      </div>

      {error && <p className="text-sm text-danger">{error}</p>}
      {deleteNote && <p className="text-sm text-success">{deleteNote}</p>}
      {!loaded && !error && <p className="text-sm text-muted">Loading jobs\u2026</p>}
      {loaded && jobs.length === 0 && !error && <p className="text-sm text-muted">No jobs yet.</p>}
      {loaded && jobs.length > 0 && (
        <p className="text-xs text-muted">
          Showing {filteredJobs.length} of {jobs.length} job{jobs.length === 1 ? '' : 's'}
        </p>
      )}

      <ul className="flex flex-col gap-2">
        {filteredJobs.map((job) => (
          <JobRow
            key={job.job_id}
            job={job}
            now={now}
            expanded={expandedId === job.job_id}
            onToggleExpand={() => setExpandedId(expandedId === job.job_id ? null : job.job_id)}
            cancelling={cancellingId === job.job_id}
            onCancel={() => handleCancel(job.job_id)}
            deleting={deletingId === job.job_id}
            onDelete={() => handleDelete(job)}
            onRetry={onRetry}
            canRetry={canRetry}
          />
        ))}
      </ul>
    </section>
  );
}
