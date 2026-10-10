import { useEffect, useRef, useState } from 'react';
import { getPartitions } from '../api.js';
import FieldHeading from './FieldHeading.jsx';
import FieldLabel from './FieldLabel.jsx';
import Badge from './ui/Badge.jsx';
import Button from './ui/Button.jsx';
import Input from './ui/Input.jsx';

const POLL_MS = 30000;
// How often the "updated Xs ago" text re-renders between polls -- coarser
// than a 1s tick (unlike JobTiming.jsx's elapsed-time ticker) since this is
// only ever a rough "how stale is this" hint, not a running clock.
const TICK_MS = 5000;

// Polls GET /v1/partitions -- same shape as utils.js's useJobList (refetch
// on an interval, expose a manual refresh, keep the last good result on a
// failed poll rather than blanking the screen). Scoped to this file since
// PartitionField is its only consumer.
function usePartitionStatus() {
  const [data, setData] = useState(null); // { partitions, generated_at } once a request has ever succeeded
  const [error, setError] = useState(null);
  const timerRef = useRef(null);

  async function refresh() {
    try {
      const result = await getPartitions();
      setData(result);
      setError(null);
    } catch (err) {
      // Keep showing the last good snapshot (if any) rather than clearing
      // the table over one transient failure -- see the component's own
      // handling of `error` below for what a caller sees either way.
      setError(err.message);
    }
  }

  useEffect(() => {
    refresh();
    timerRef.current = setInterval(refresh, POLL_MS);
    return () => clearInterval(timerRef.current);
  }, []);

  return { data, error, refresh };
}

// Forces a re-render every TICK_MS so "updated Xs ago" stays roughly
// current between polls -- a plain local tick, not shared state, since
// nothing else on the page needs it.
function useTick(ms) {
  const [, setTick] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => setTick((t) => t + 1), ms);
    return () => clearInterval(interval);
  }, [ms]);
}

function agoText(isoTimestamp) {
  if (!isoTimestamp) return null;
  const seconds = Math.max(0, Math.round((Date.now() - new Date(isoTimestamp).getTime()) / 1000));
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  return `${Math.round(seconds / 60)}m ago`;
}

// Plain free-text fallback -- used before the first live response ever
// arrives, and permanently for a deployment where GET /v1/partitions keeps
// failing (a stale-but-working form beats a broken one). Identical to this
// field's own pre-availability-table behavior.
function FallbackInput({ model, value, onChange, defaultPartition, warning }) {
  return (
    <FieldLabel model={model} field="partition" text="partition">
      <Input
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={defaultPartition ? `default: ${defaultPartition}` : 'e.g. main'}
      />
      {warning && <p className="mt-1 text-xs text-warning">{warning}</p>}
    </FieldLabel>
  );
}

function PartitionRow({ partition, checked, groupName, reclaimableFromName, onSelect }) {
  return (
    <label
      className={`flex flex-wrap items-center gap-x-3 gap-y-1 rounded-[6px] px-2.5 py-1.5 text-sm transition-colors cursor-pointer ${
        checked ? 'bg-accent/15' : 'hover:bg-surface-2'
      }`}
    >
      <input
        type="radio"
        name={groupName}
        checked={checked}
        onChange={onSelect}
        className="h-3.5 w-3.5 shrink-0 accent-accent"
      />
      <span className="flex min-w-0 items-center gap-1.5 font-medium text-text">
        {partition.name}
        {partition.is_default && <Badge tone="accent">default</Badge>}
        {partition.preemptible && <Badge tone="warning">preemptible</Badge>}
        {partition.state !== 'UP' && <Badge tone="danger">{partition.state}</Badge>}
      </span>
      <span className="ml-auto flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
        <span>{partition.free_gpus} free</span>
        <span>
          {partition.reclaimable_gpus != null
            ? `${partition.reclaimable_gpus} from ${reclaimableFromName ?? 'preemptible jobs'}`
            : '\u2014'}
        </span>
        <span>{partition.waiting_jobs} waiting</span>
      </span>
    </label>
  );
}

// Adds an optional `partition` field to a generation request -- shows each
// partition's live GPU availability and queue length (GET /v1/partitions)
// so a caller can actually see which one has room before choosing, instead
// of guessing. Falls back to a plain text input if that endpoint isn't
// reachable, so a form never gets stuck on this field alone.
//
// `model` selects which concrete request schema this field's hover
// description is read from (identical text today across every one of them,
// since they all inherit it unchanged from the same mixin -- but read live
// per-model anyway rather than assumed, in case that ever changes).
//
// `value`/`onChange` behave exactly as before this field grew a live
// table: `value` is `''` when no partition was explicitly chosen (the
// server-side default applies), or the exact partition name otherwise --
// selecting the row marked "default" calls `onChange('')`, same as leaving
// the old free-text box blank.
export default function PartitionField({ model, value, onChange, defaultPartition }) {
  const { data, error, refresh } = usePartitionStatus();
  useTick(TICK_MS);

  if (!data) {
    // Either still loading the very first response, or it already failed
    // once -- either way, there's nothing live to show yet.
    return (
      <FallbackInput
        model={model}
        value={value}
        onChange={onChange}
        defaultPartition={defaultPartition}
        warning={error ? `Live availability unavailable (${error}) -- you can still type a partition name directly.` : null}
      />
    );
  }

  const { partitions, generated_at: generatedAt } = data;
  const groupName = `partition-${model}`;
  const knownNames = new Set(partitions.map((p) => p.name));
  const preemptiblePartition = partitions.find((p) => p.preemptible);
  // The row a blank `value` (today's "use the server default" signal)
  // corresponds to -- whichever partition GET /v1/partitions itself marks
  // is_default, so this can never drift from GET /v1/health's own
  // default_partition.
  const selectedName = value || partitions.find((p) => p.is_default)?.name || '';

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <FieldHeading model={model} field="partition" text="partition" />
        <div className="flex items-center gap-2 text-xs text-muted">
          {generatedAt && <span>updated {agoText(generatedAt)}</span>}
          <Button type="button" variant="ghost" size="sm" onClick={refresh}>
            Refresh
          </Button>
        </div>
      </div>

      <div className="flex flex-col gap-0.5 rounded-md border border-border bg-surface p-1" role="radiogroup">
        {partitions.map((partition) => (
          <PartitionRow
            key={partition.name}
            partition={partition}
            checked={partition.name === selectedName}
            groupName={groupName}
            reclaimableFromName={preemptiblePartition?.name}
            onSelect={() => onChange(partition.is_default ? '' : partition.name)}
          />
        ))}
      </div>

      {value && !knownNames.has(value) && (
        <p className="text-xs text-warning">
          Currently set to &quot;{value}&quot;, which isn&apos;t one of the options above (it may no longer exist on
          this cluster) -- selecting a row above will replace it.
        </p>
      )}
      {error && <p className="text-xs text-warning">Live availability may be stale ({error}).</p>}

      <p className="text-xs text-muted">
        Snapshot, not a guarantee: free GPUs can be taken by another request the instant after this loads, and a
        preemptible job can be interrupted and automatically restarted at any time.
      </p>
    </div>
  );
}
