import { formatDuration } from '../utils.js';

// Friendly names for JobStage.name (common/schemas.py) -- currently only
// wan-animate:replace jobs ever populate `progress.stages`, but this maps
// whatever stage names show up regardless of pipeline.
const STAGE_LABELS = {
  preprocess: 'Preprocessing',
  generate: 'Generating',
  mux_audio: 'Adding audio',
};

function stageLabel(name) {
  return STAGE_LABELS[name] ?? name;
}

// Seconds elapsed between an ISO timestamp and `now` (a Date) -- never
// negative, so a slightly-stale `now` (between this tick and the next
// server poll) can't render "-2s".
function secondsSince(isoTimestamp, now) {
  if (!isoTimestamp) return null;
  const seconds = (now.getTime() - new Date(isoTimestamp).getTime()) / 1000;
  return Math.max(seconds, 0);
}

function clipStepText(progress) {
  if (!progress || progress.clip == null || progress.step == null) return null;
  const clipPart = progress.clip_count ? `clip ${progress.clip} of ${progress.clip_count}` : `clip ${progress.clip}`;
  return `${clipPart}, step ${progress.step}/${progress.step_count ?? '?'}`;
}

// Fraction complete across the *whole* generate stage (not just the
// current clip) -- used for the <progress> bar in JobTimingDetail, so it
// fills smoothly across clips instead of resetting to 0% at each one.
function overallFraction(progress) {
  if (!progress?.clip || !progress?.clip_count || !progress?.step_count) return null;
  const done = (progress.clip - 1) * progress.step_count + (progress.step ?? 0);
  const total = progress.clip_count * progress.step_count;
  return total > 0 ? Math.min(1, done / total) : null;
}

// How long a running job is expected to take, and where that number came
// from: the estimate the server made from this request's own size
// (`estimated_run_seconds`, only some backends -- H3 today) if there is one,
// otherwise the median of earlier runs of the same recipe.
function expectedRun(job) {
  if (job.estimated_run_seconds != null) return { seconds: job.estimated_run_seconds, label: 'estimated' };
  if (job.typical_run_seconds != null) return { seconds: job.typical_run_seconds, label: 'typical' };
  return null;
}

// One-line summary shown directly in the (always-visible) job row --
// elapsed time, current stage/clip if there is one, and a rough
// time-left estimate once one exists (the job's own live ETA if
// available, otherwise expected-minus-elapsed).
export function JobTimingSummary({ job, now }) {
  if (job.status === 'queued') {
    const elapsed = secondsSince(job.created_at, now);
    const waitingFor = job.gpus != null ? `${job.gpus} GPUs` : 'a GPU';
    const estimate =
      job.estimated_run_seconds != null ? ` \u2014 est. ~${formatDuration(job.estimated_run_seconds)} once started` : '';
    return (
      <span className="text-xs text-muted">
        Queued {formatDuration(elapsed)}, waiting for {waitingFor}
        {estimate}
      </span>
    );
  }

  if (job.status === 'running') {
    const elapsed = secondsSince(job.started_at ?? job.created_at, now);
    const stage = job.progress?.current_stage;
    const clipStep = clipStepText(job.progress);

    let remainingText = null;
    if (job.progress?.eta_seconds != null) {
      remainingText = `about ${formatDuration(job.progress.eta_seconds)} left`;
    } else if (expectedRun(job) != null && elapsed != null) {
      const expected = expectedRun(job);
      const remaining = expected.seconds - elapsed;
      remainingText =
        remaining > 0
          ? `about ${formatDuration(remaining)} left`
          : `running longer than ${expected.label} (${expected.label} ~${formatDuration(expected.seconds)})`;
    }

    return (
      <span className="text-xs text-muted">
        Running {formatDuration(elapsed)}
        {stage && ` \u2014 ${stageLabel(stage)}${clipStep ? `, ${clipStep}` : ''}`}
        {remainingText && ` \u2014 ${remainingText}`}
      </span>
    );
  }

  if (job.status === 'succeeded' || job.status === 'failed') {
    const started = job.started_at ?? job.created_at;
    const duration = job.finished_at && started ? (new Date(job.finished_at).getTime() - new Date(started).getTime()) / 1000 : null;
    const formatted = duration != null ? formatDuration(duration) : null;
    if (!formatted) return null;
    return (
      <span className="text-xs text-muted">
        {job.status === 'succeeded' ? 'Took' : 'Ran for'} {formatted}
      </span>
    );
  }

  return null;
}

// Fuller breakdown for the expanded job details: when it was submitted,
// queue wait, each stage's own duration, an overall progress bar for
// wan-animate's clip/step tracking, and where the typical-time figure
// came from.
export function JobTimingDetail({ job, now }) {
  const queueWaitSeconds = job.started_at
    ? (new Date(job.started_at).getTime() - new Date(job.created_at).getTime()) / 1000
    : job.status === 'queued'
      ? secondsSince(job.created_at, now)
      : null;

  const fraction = overallFraction(job.progress);
  const clipStep = clipStepText(job.progress);

  return (
    <div className="flex flex-col gap-1.5 text-xs text-muted">
      <p>Submitted: {new Date(job.created_at).toLocaleString()}</p>
      {queueWaitSeconds != null && <p>Queue wait: {formatDuration(queueWaitSeconds) ?? '0s'}</p>}

      {job.progress?.stages?.length > 0 && (
        <ul className="flex flex-col gap-0.5">
          {job.progress.stages.map((stage) => (
            <li key={stage.name}>
              {stageLabel(stage.name)}:{' '}
              {stage.state === 'done'
                ? formatDuration(stage.seconds)
                : stage.state === 'running'
                  ? 'in progress'
                  : 'not started yet'}
            </li>
          ))}
        </ul>
      )}

      {fraction != null && (
        <div className="flex items-center gap-2">
          <progress
            value={fraction}
            max={1}
            className="h-1.5 w-40 overflow-hidden rounded-full [&::-webkit-progress-bar]:bg-surface-2 [&::-webkit-progress-value]:bg-accent [&::-moz-progress-bar]:bg-accent"
          />
          <span>{clipStep}</span>
        </div>
      )}

      {job.gpus != null && <p>GPUs: {job.gpus}</p>}

      {job.estimated_run_seconds != null && (
        <p>Estimated run time for this request: ~{formatDuration(job.estimated_run_seconds)} (from its size, once started)</p>
      )}

      {job.typical_run_seconds != null && (
        <p>
          Typical for this recipe: ~{formatDuration(job.typical_run_seconds)} ({job.typical_basis ?? 'estimate'})
        </p>
      )}
    </div>
  );
}
