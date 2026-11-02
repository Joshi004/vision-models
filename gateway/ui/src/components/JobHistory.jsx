import { useEffect, useMemo, useState } from 'react';
import { cancelJob, purgeJob, setJobPriority } from '../api.js';
import { forgetSubmittedJob, useAutoCacheResults } from '../autoCache.js';
import { deleteCachedResult } from '../resultCache.js';
import { useJobSearch } from '../utils.js';
import { usePersistedState } from '../storage.js';
import Button from './ui/Button.jsx';
import Input from './ui/Input.jsx';
import Select from './ui/Select.jsx';
import JobRow from './JobRow.jsx';

const STATUS_OPTIONS = [
  { value: 'all', label: 'All' },
  { value: 'active', label: 'In progress' },
  { value: 'succeeded', label: 'Succeeded' },
  { value: 'failed', label: 'Failed' },
];

const MEDIA_OPTIONS = [
  { value: 'all', label: 'All' },
  { value: 'video', label: 'Video' },
  { value: 'audio', label: 'Audio' },
  { value: 'image', label: 'Image' },
  { value: 'text', label: 'Text' },
];

const SOURCE_OPTIONS = [
  { value: 'all', label: 'All' },
  { value: 'ui', label: 'Sent from UI' },
  { value: 'api', label: 'Sent by clients' },
  { value: 'unknown', label: 'Unknown (older jobs)' },
];

const CACHED_OPTIONS = [
  { value: 'all', label: 'All' },
  { value: 'cached', label: 'Cached in this browser' },
  { value: 'uncached', label: 'Not cached' },
];

const PAGE_SIZE_OPTIONS = [25, 50, 100];
const SEARCH_DEBOUNCE_MS = 300;

// The full job history, one page at a time: POST /v1/jobs/search does the
// filtering and paging on the server (status, model, job type, media, who
// sent it, cached in this browser, and text search), independent of
// JobsPanel.jsx's smaller, always-visible strip, with a permanent Delete per
// finished job (DELETE /v1/jobs/{job_id}/purge, via JobRow's onDelete).
// Filters persist in sessionStorage (see storage.js) so navigating away and
// back (or a refresh) doesn't silently reset them.
//
// `groups` (a "model": one backend) and `recipes` (a "job type": one
// recipe, with its result's `media`) come from App.jsx. The server only
// knows pipelines, so the Model/Job type/Media choices are turned into a
// list of pipelines here before asking.
//
// `onRetry`/`canRetry` come from App.jsx unchanged -- see JobRow.jsx for
// what each does; this component has no retry logic of its own.
export default function JobHistory({ groups, recipes, jobRetentionDays, onRetry, canRetry, onRetryLipSync }) {
  const [expandedId, setExpandedId] = useState(null);
  const [cancellingId, setCancellingId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [priorityChangingId, setPriorityChangingId] = useState(null);
  const [deleteNote, setDeleteNote] = useState(null);
  const session = { storage: window.sessionStorage };
  const [statusFilter, setStatusFilter] = usePersistedState('historyFilter:status', 'all', session);
  const [modelFilter, setModelFilter] = usePersistedState('historyFilter:model', 'all', session);
  // Kept under its old key: the job type is the pipeline, as the old "Recipe" filter was.
  const [jobTypeFilter, setJobTypeFilter] = usePersistedState('historyFilter:pipeline', 'all', session);
  const [mediaFilter, setMediaFilter] = usePersistedState('historyFilter:media', 'all', session);
  const [sourceFilter, setSourceFilter] = usePersistedState('historyFilter:source', 'all', session);
  const [cachedFilter, setCachedFilter] = usePersistedState('historyFilter:cached', 'all', session);
  const [searchText, setSearchText] = usePersistedState('historyFilter:search', '', session);
  const [pageSize, setPageSize] = usePersistedState('historyPageSize', 50, {
    validate: (value) => PAGE_SIZE_OPTIONS.includes(value),
  });
  // 0-based. Every filter change goes back to the first page (see `changeFilter`).
  const [page, setPage] = useState(0);

  // The search box is applied a moment after typing stops, not on every keystroke.
  const [debouncedSearch, setDebouncedSearch] = useState(searchText.trim());
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(searchText.trim()), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [searchText]);

  const groupLabels = useMemo(() => new Map(groups.map((group) => [group.id, group.label])), [groups]);

  // Recipes the Model and Media filters leave in; they are the Job type choices.
  const recipeChoices = useMemo(
    () =>
      recipes.filter(
        (recipe) =>
          (modelFilter === 'all' || recipe.group === modelFilter) &&
          (mediaFilter === 'all' || recipe.media === mediaFilter),
      ),
    [recipes, modelFilter, mediaFilter],
  );
  // A stored job type the other filters now rule out (or one that no longer exists) counts as "All".
  const jobType = recipeChoices.some((recipe) => recipe.pipeline === jobTypeFilter) ? jobTypeFilter : 'all';

  // null = no restriction. Model, Job type and Media are all just ways of choosing pipelines.
  let pipelines = null;
  if (jobType !== 'all') pipelines = [jobType];
  else if (modelFilter !== 'all' || mediaFilter !== 'all') pipelines = recipeChoices.map((recipe) => recipe.pipeline);

  const filters = {
    statuses: statusFilter === 'active' ? ['queued', 'running'] : statusFilter === 'all' ? null : [statusFilter],
    pipelines,
    source: sourceFilter,
    cached: cachedFilter,
    q: debouncedSearch,
  };
  const { jobs, total, loaded, error, setError, now, refresh } = useJobSearch(filters, page * pageSize, pageSize);
  useAutoCacheResults(jobs);

  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const firstShown = total === 0 ? 0 : page * pageSize + 1;
  const lastShown = page * pageSize + jobs.length;
  const filtersActive =
    statusFilter !== 'all' ||
    modelFilter !== 'all' ||
    jobType !== 'all' ||
    mediaFilter !== 'all' ||
    sourceFilter !== 'all' ||
    cachedFilter !== 'all' ||
    searchText.trim() !== '';

  // After a delete (or the jobs expiring) the last pages can disappear: step back to the new last page.
  useEffect(() => {
    if (loaded && page > 0 && page >= pageCount) setPage(pageCount - 1);
  }, [loaded, page, pageCount]);

  // A filter change goes back to the first page, whatever page the old filter left us on.
  function changeFilter(setter) {
    return (event) => {
      setter(event.target.value);
      setPage(0);
    };
  }

  // Model and Media change which job types exist, so they also clear the Job type choice.
  function changeModelOrMedia(setter) {
    return (event) => {
      setter(event.target.value);
      setJobTypeFilter('all');
      setPage(0);
    };
  }

  function clearFilters() {
    setStatusFilter('all');
    setModelFilter('all');
    setJobTypeFilter('all');
    setMediaFilter('all');
    setSourceFilter('all');
    setCachedFilter('all');
    setSearchText('');
    setPage(0);
  }

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

  async function handleChangePriority(job, priority) {
    setPriorityChangingId(job.job_id);
    setError(null);
    try {
      await setJobPriority(job.job_id, priority);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setPriorityChangingId(null);
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
      forgetSubmittedJob(job.job_id);
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
          {jobRetentionDays != null ? ` ${jobRetentionDays}` : ''} days after they finish. A job submitted with
          Incognito on is the exception: its result can be downloaded once and is then deleted, and only a minimal
          record of it stays here.
        </p>
      </div>

      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col gap-1 text-xs text-muted">
          Status
          <Select value={statusFilter} onChange={changeFilter(setStatusFilter)} className="min-w-[8rem]">
            {STATUS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Model
          <Select value={modelFilter} onChange={changeModelOrMedia(setModelFilter)} className="min-w-[10rem]">
            <option value="all">All</option>
            {groups.map((group) => (
              <option key={group.id} value={group.id}>
                {group.label}
              </option>
            ))}
          </Select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Job type
          <Select value={jobType} onChange={changeFilter(setJobTypeFilter)} className="min-w-[10rem]">
            <option value="all">All</option>
            {recipeChoices.map((recipe) => (
              <option key={recipe.pipeline} value={recipe.pipeline}>
                {modelFilter === 'all' ? `${groupLabels.get(recipe.group) ?? recipe.group}: ` : ''}
                {recipe.label}
              </option>
            ))}
          </Select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Media
          <Select value={mediaFilter} onChange={changeModelOrMedia(setMediaFilter)} className="min-w-[7rem]">
            {MEDIA_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Sent by
          <Select value={sourceFilter} onChange={changeFilter(setSourceFilter)} className="min-w-[10rem]">
            {SOURCE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Cached
          <Select value={cachedFilter} onChange={changeFilter(setCachedFilter)} className="min-w-[10rem]">
            {CACHED_OPTIONS.map((option) => (
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
            onChange={(event) => {
              setSearchText(event.target.value);
              setPage(0);
            }}
            maxLength={200}
            placeholder="Job ID or prompt text\u2026"
          />
        </label>
      </div>

      {error && <p className="text-sm text-danger">{error}</p>}
      {deleteNote && <p className="text-sm text-success">{deleteNote}</p>}
      {!loaded && !error && <p className="text-sm text-muted">Loading jobs\u2026</p>}
      {loaded && total === 0 && !error && (
        <p className="text-sm text-muted">{filtersActive ? 'No matching jobs.' : 'No jobs yet.'}</p>
      )}
      {loaded && total > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
          <span>
            Showing {firstShown}&ndash;{lastShown} of {total} job{total === 1 ? '' : 's'}
          </span>
          {filtersActive && (
            <Button type="button" variant="ghost" size="sm" onClick={clearFilters}>
              Clear filters
            </Button>
          )}
        </div>
      )}

      <ul className="flex flex-col gap-2">
        {jobs.map((job) => (
          <JobRow
            key={job.job_id}
            job={job}
            now={now}
            expanded={expandedId === job.job_id}
            onToggleExpand={() => setExpandedId(expandedId === job.job_id ? null : job.job_id)}
            cancelling={cancellingId === job.job_id}
            onCancel={() => handleCancel(job.job_id)}
            onChangePriority={handleChangePriority}
            changingPriority={priorityChangingId === job.job_id}
            deleting={deletingId === job.job_id}
            onDelete={() => handleDelete(job)}
            onRetry={onRetry}
            canRetry={canRetry}
            onRetryLipSync={onRetryLipSync}
          />
        ))}
      </ul>

      {loaded && total > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 text-sm">
          <div className="flex items-center gap-2">
            <Button type="button" variant="secondary" size="sm" disabled={page === 0} onClick={() => setPage(page - 1)}>
              Previous
            </Button>
            <span className="text-xs text-muted">
              Page {page + 1} of {pageCount}
            </span>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              disabled={page + 1 >= pageCount}
              onClick={() => setPage(page + 1)}
            >
              Next
            </Button>
          </div>
          <label className="flex items-center gap-2 text-xs text-muted">
            Per page
            <Select
              value={pageSize}
              onChange={(event) => {
                setPageSize(Number(event.target.value));
                setPage(0);
              }}
            >
              {PAGE_SIZE_OPTIONS.map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </Select>
          </label>
        </div>
      )}
    </section>
  );
}
