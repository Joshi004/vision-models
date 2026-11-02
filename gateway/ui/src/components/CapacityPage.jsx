import { useEffect, useState } from 'react';
import { getCapacity, getPartitions, putCapacity } from '../api.js';
import Badge from './ui/Badge.jsx';
import Button from './ui/Button.jsx';
import Input from './ui/Input.jsx';
import { useToast } from './ui/Toast.jsx';
import { QUEUE_STATES } from '../utils.js';

const POLL_MS = 10000;

// A limit the user typed: a whole number of 1 or more, else null (not valid yet).
function parseLimit(text) {
  const trimmed = String(text).trim();
  if (!/^\d+$/.test(trimmed)) return null;
  const value = Number(trimmed);
  return value >= 1 ? value : null;
}

// Polls GET /v1/capacity (this service's limits and usage) and, as extra
// information, GET /v1/partitions (the cluster's own free GPUs). A failed
// poll keeps the last good data on screen.
function useCapacityData() {
  const [capacity, setCapacity] = useState(null);
  const [partitions, setPartitions] = useState(null);
  const [error, setError] = useState(null);

  async function refresh() {
    try {
      setCapacity(await getCapacity());
      setError(null);
    } catch (err) {
      setError(err.message);
    }
    try {
      setPartitions((await getPartitions()).partitions);
    } catch {
      // The partition table is only extra context; keep whatever was shown.
    }
  }

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, POLL_MS);
    return () => clearInterval(interval);
  }, []);

  return { capacity, setCapacity, partitions, error };
}

function Count({ value }) {
  return <td className="px-2 py-2 text-right tabular-nums">{value > 0 ? value : <span className="text-muted">0</span>}</td>;
}

// A cell for something this kind of model does not have (for example warm workers of a one-shot model).
function NotApplicable() {
  return <td className="px-2 py-2 text-right text-muted">&mdash;</td>;
}

// One labelled number in a summary row.
function Stat({ label, value, unit, tone = 'text-text', title }) {
  return (
    <div className="text-sm" title={title}>
      <p className="text-muted">{label}</p>
      <p className={`text-lg font-semibold tabular-nums ${tone}`}>
        {value} {unit && <span className="text-sm font-normal text-muted">{unit}</span>}
      </p>
    </div>
  );
}

function queuedFor(row, reason) {
  return row.queued_by_reason?.[reason] ?? 0;
}

function sumOf(rows, pick) {
  return rows.reduce((sum, row) => sum + (pick(row) ?? 0), 0);
}

// Where the global GPU limit and each model's job limit are changed. The limits live on the
// server (no environment variables): saving applies them at once and never cancels a job that
// is running. The per-model limits, each counted at its GPUs per job, may not add up to more
// than the global limit; the server enforces the same rule.
export default function CapacityPage() {
  const { addToast } = useToast();
  const { capacity, setCapacity, partitions, error } = useCapacityData();
  // What the user has typed, as strings, once they edit anything; null means "show what the server has".
  const [draft, setDraft] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);

  if (!capacity) {
    return (
      <section className="mx-auto max-w-6xl">
        <h2 className="text-lg font-semibold text-text">GPU capacity</h2>
        <p className="mt-2 text-sm text-muted">{error ? `Could not load the limits: ${error}` : 'Loading\u2026'}</p>
      </section>
    );
  }

  const limitModels = capacity.models.filter((model) => model.kind !== 'fixed');
  const fixedModels = capacity.models.filter((model) => model.kind === 'fixed');
  const fixedGpus = fixedModels.reduce((sum, model) => sum + model.planning_gpus, 0);

  const saved = {
    globalGpus: String(capacity.global_gpus),
    jobs: Object.fromEntries(limitModels.map((model) => [model.key, String(model.max_jobs)])),
  };
  const values = draft ?? saved;
  const changed =
    draft !== null &&
    (draft.globalGpus !== saved.globalGpus || limitModels.some((model) => draft.jobs[model.key] !== saved.jobs[model.key]));

  const globalLimit = parseLimit(values.globalGpus);
  const parsedJobs = limitModels.map((model) => parseLimit(values.jobs[model.key]));
  const anyInvalid = globalLimit === null || parsedJobs.some((value) => value === null);
  const planned = anyInvalid
    ? null
    : limitModels.reduce((sum, model, index) => sum + parsedJobs[index] * model.planning_gpus, fixedGpus);
  const overPlan = planned !== null && planned > globalLimit;
  const canSave = changed && !anyInvalid && !overPlan && !saving;
  const heavyModels = limitModels.filter((model) => model.planning_gpus > 1);

  async function save() {
    setSaving(true);
    setSaveError(null);
    try {
      const updated = await putCapacity({
        global_gpus: globalLimit,
        max_jobs: Object.fromEntries(limitModels.map((model, index) => [model.key, parsedJobs[index]])),
      });
      setCapacity(updated);
      setDraft(null);
      addToast({ tone: 'success', message: 'Capacity limits saved.' });
    } catch (err) {
      setSaveError(err.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="mx-auto flex max-w-6xl flex-col gap-5">
      <div>
        <h2 className="text-lg font-semibold text-text">GPU capacity</h2>
        <p className="text-sm text-muted">
          How many GPUs this service may use in total, and how many jobs (and so warm workers) each model may run at
          once. Changes apply immediately. Running jobs are never cancelled: if a limit is lowered, busy workers finish
          their request, idle workers above the limit are stopped, and nothing new starts until usage is back under the
          limit. If a limit is raised, waiting requests start at once.
        </p>
      </div>

      {error && <p className="text-xs text-warning">Could not refresh ({error}); showing the last data received.</p>}

      <div className="rounded-lg border border-border bg-surface p-4">
        <div className="flex flex-wrap items-end gap-6">
          <label className="flex flex-col gap-1 text-xs text-muted">
            Global GPU limit
            <Input
              type="number"
              min="1"
              step="1"
              value={values.globalGpus}
              onChange={(event) => setDraft({ ...values, globalGpus: event.target.value })}
              className="w-32"
            />
          </label>
          <Stat
            label="GPUs allocated"
            value={capacity.gpus_allocated}
            unit={`of ${capacity.global_gpus} GPUs`}
            title="GPUs Slurm has given this service: running jobs, and workers or the Qwen3.8 server that have a GPU."
          />
          <Stat
            label="GPUs pending in Slurm"
            value={capacity.gpus_pending}
            unit="GPUs"
            tone={capacity.gpus_pending > 0 ? 'text-warning' : 'text-text'}
            title="GPUs this service asked Slurm for that are not allocated yet. They already count toward the global limit."
          />
          <Stat
            label="Counted toward the limit"
            value={capacity.gpus_in_use}
            unit={`of ${capacity.global_gpus} GPUs`}
            title="Allocated plus pending."
          />
          <Stat
            label="Planned by the limits below"
            value={planned === null ? '\u2014' : planned}
            unit={`of ${globalLimit ?? '\u2014'} GPUs`}
            tone={overPlan ? 'text-danger' : 'text-text'}
          />
          <Stat label="Running jobs" value={capacity.running_jobs} />
          <Stat label="Queued jobs" value={capacity.queued_jobs} />
        </div>
        {capacity.gpus_in_use > capacity.global_gpus && (
          <p className="mt-3 text-xs text-warning">
            Usage is above the global limit. Running jobs will finish, and no new worker starts until usage drops below
            {` ${capacity.global_gpus}`}.
          </p>
        )}
      </div>

      {partitions && partitions.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <h3 className="text-sm font-semibold text-text">Cluster GPUs available</h3>
          <ul className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-sm">
            {partitions.map((partition) => (
              <li key={partition.name} className="flex items-center gap-2">
                <span className="text-text">{partition.name}</span>
                <span className="tabular-nums text-muted">
                  {partition.free_gpus} free of {partition.total_gpus}
                </span>
                {partition.is_default && <Badge tone="accent">default</Badge>}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="rounded-lg border border-border bg-surface p-4">
        <h3 className="text-sm font-semibold text-text">
          Waiting jobs: {capacity.queued_jobs} <span className="font-normal text-muted">(why they are not running yet)</span>
        </h3>
        <ul className="mt-2 grid gap-x-8 gap-y-1 text-sm sm:grid-cols-2">
          {QUEUE_STATES.map((state) => {
            const count = capacity.queued_by_reason?.[state.key] ?? 0;
            return (
              <li key={state.key} className="flex items-baseline gap-2">
                <span className={`w-8 text-right tabular-nums ${count > 0 ? 'font-semibold text-text' : 'text-muted'}`}>{count}</span>
                <span className="text-text">{state.label}</span>
                <span className="text-xs text-muted">{state.explain}</span>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="overflow-x-auto rounded-lg border border-border bg-surface">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-muted">
              <th rowSpan={2} className="px-3 py-2 text-left align-bottom font-medium">
                Model
              </th>
              <th rowSpan={2} className="px-2 py-2 text-right align-bottom font-medium">
                Running jobs
              </th>
              <th colSpan={QUEUE_STATES.length + 1} className="border-l border-border px-2 pt-2 text-center font-medium">
                Waiting jobs, by reason
              </th>
              <th colSpan={4} className="border-l border-border px-2 pt-2 text-center font-medium">
                Warm workers
              </th>
              <th colSpan={2} className="border-l border-border px-2 pt-2 text-center font-medium">
                GPUs
              </th>
            </tr>
            <tr className="border-b border-border text-xs text-muted">
              {QUEUE_STATES.map((state, index) => (
                <th
                  key={state.key}
                  title={state.explain}
                  className={`px-2 py-2 text-right font-medium ${index === 0 ? 'border-l border-border' : ''}`}
                >
                  {state.short}
                </th>
              ))}
              <th className="px-2 py-2 text-right font-medium">Total</th>
              <th className="border-l border-border px-2 py-2 text-right font-medium">Busy</th>
              <th className="px-2 py-2 text-right font-medium">Idle</th>
              <th className="px-2 py-2 text-right font-medium" title="Workers whose Slurm job is waiting for a GPU">
                Pending
              </th>
              <th className="px-2 py-2 text-right font-medium" title="Workers that have a GPU and are loading the model">
                Loading
              </th>
              <th className="border-l border-border px-2 py-2 text-right font-medium" title="Allocated by Slurm">
                Allocated
              </th>
              <th className="px-3 py-2 text-right font-medium" title="Asked of Slurm, not allocated yet">
                Pending
              </th>
            </tr>
          </thead>
          <tbody>
            {[...limitModels, ...fixedModels].map((model) => (
              <tr key={model.key} className="border-b border-border last:border-b-0">
                <td className="px-3 py-2 text-text">{model.label}</td>
                <Count value={model.running_jobs} />
                {QUEUE_STATES.map((state) => (
                  <Count key={state.key} value={queuedFor(model, state.key)} />
                ))}
                <td className="px-2 py-2 text-right font-medium tabular-nums text-text">{model.queued}</td>
                {model.kind === 'pool' ? (
                  <>
                    <Count value={model.busy} />
                    <Count value={model.idle} />
                    <Count value={model.workers_pending} />
                    <Count value={model.workers_loading} />
                  </>
                ) : (
                  <>
                    <NotApplicable />
                    <NotApplicable />
                    <NotApplicable />
                    <NotApplicable />
                  </>
                )}
                <Count value={model.gpus_allocated} />
                <td className="px-3 py-2 text-right tabular-nums">
                  {model.gpus_pending > 0 ? <span className="text-warning">{model.gpus_pending}</span> : <span className="text-muted">0</span>}
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="border-t border-border bg-surface-2 font-medium text-text">
              <td className="px-3 py-2">Total</td>
              <td className="px-2 py-2 text-right tabular-nums">{capacity.running_jobs}</td>
              {QUEUE_STATES.map((state) => (
                <td key={state.key} className="px-2 py-2 text-right tabular-nums">
                  {capacity.queued_by_reason?.[state.key] ?? 0}
                </td>
              ))}
              <td className="px-2 py-2 text-right tabular-nums">{capacity.queued_jobs}</td>
              <td className="px-2 py-2 text-right tabular-nums">{sumOf(capacity.models.filter((m) => m.kind === 'pool'), (m) => m.busy)}</td>
              <td className="px-2 py-2 text-right tabular-nums">{sumOf(capacity.models.filter((m) => m.kind === 'pool'), (m) => m.idle)}</td>
              <td className="px-2 py-2 text-right tabular-nums">{sumOf(capacity.models, (m) => m.workers_pending)}</td>
              <td className="px-2 py-2 text-right tabular-nums">{sumOf(capacity.models, (m) => m.workers_loading)}</td>
              <td className="px-2 py-2 text-right tabular-nums">{capacity.gpus_allocated}</td>
              <td className="px-3 py-2 text-right tabular-nums">{capacity.gpus_pending}</td>
            </tr>
          </tfoot>
        </table>
      </div>

      <div className="overflow-x-auto rounded-lg border border-border bg-surface">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-muted">
              <th className="px-3 py-2 font-medium">Model</th>
              <th className="px-2 py-2 font-medium">Max jobs</th>
              <th className="px-2 py-2 text-right font-medium">GPUs per job</th>
            </tr>
          </thead>
          <tbody>
            {limitModels.map((model, index) => (
              <tr key={model.key} className="border-b border-border last:border-b-0">
                <td className="px-3 py-2 text-text">
                  {model.label}
                  {model.kind === 'oneshot' && <span className="ml-2 text-xs text-muted">one Slurm job per request</span>}
                </td>
                <td className="px-2 py-2">
                  <Input
                    type="number"
                    min="1"
                    step="1"
                    value={values.jobs[model.key]}
                    onChange={(event) => setDraft({ ...values, jobs: { ...values.jobs, [model.key]: event.target.value } })}
                    className={`w-24 ${parsedJobs[index] === null ? 'border-danger' : ''}`}
                  />
                </td>
                <td className="px-2 py-2 text-right tabular-nums text-muted">{model.planning_gpus}</td>
              </tr>
            ))}
            {fixedModels.map((model) => (
              <tr key={model.key} className="border-b border-border last:border-b-0">
                <td className="px-3 py-2 text-text">
                  {model.label}
                  <span className="ml-2 text-xs text-muted">one shared server, no limit of its own</span>
                </td>
                <td className="px-2 py-2 text-muted">&mdash;</td>
                <td className="px-2 py-2 text-right tabular-nums text-muted">{model.planning_gpus}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-muted">
        &ldquo;Max jobs&rdquo; is the most jobs of a model running at once; for a model with warm workers it is also the most
        warm workers it keeps.
        {heavyModels.length > 0 &&
          ` For the sum check, ${heavyModels
            .map((model) => `${model.label} counts ${model.planning_gpus} GPUs per job`)
            .join(', ')}; a job that really needs more waits in the queue until it fits under the global limit.`}
      </p>

      {overPlan && (
        <p className="text-sm text-danger">
          The limits above add up to {planned} GPUs, more than the global limit of {globalLimit}. Lower a model&apos;s limit
          or raise the global limit.
        </p>
      )}
      {anyInvalid && <p className="text-sm text-danger">Every limit must be a whole number of 1 or more.</p>}
      {saveError && <p className="text-sm text-danger">{saveError}</p>}

      <div className="flex gap-3">
        <Button type="button" onClick={save} disabled={!canSave}>
          {saving ? 'Saving\u2026' : 'Save limits'}
        </Button>
        <Button
          type="button"
          variant="secondary"
          onClick={() => {
            setDraft(null);
            setSaveError(null);
          }}
          disabled={!changed || saving}
        >
          Discard changes
        </Button>
      </div>
    </section>
  );
}
