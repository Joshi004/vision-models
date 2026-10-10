import { useEffect, useRef } from 'react';
import { RequiredLegend } from './FieldLabel.jsx';
import Button from './ui/Button.jsx';
import { useToast } from './ui/Toast.jsx';

// Shared chrome for every recipe form: header (title/description/
// endpoint), the Retry prefill banner, a "draft restored" hint, and a
// sticky footer with the blockers list, Reset, and the submit button --
// replaces the near-identical header/footer JSX that used to be
// duplicated across all six form files (see forms/*.jsx). Each form still
// owns its own field state (useFormState, see storage.js), payload, and
// submit handler -- this component is purely presentational chrome around
// that; `children` is the form's own Section/CollapsibleSection groups.
//
// Ctrl/Cmd+Enter submits from anywhere inside the form (a common
// "compose" convention), in addition to the normal submit button click.
//
// On every *new* successful submission (lastJobId changing to a fresh,
// non-null value), this remembers the form's current settings
// (rememberSettings, from useFormState) and raises a confirmation toast.
export default function FormShell({
  title,
  description,
  endpoint,
  prefillJob,
  onClearPrefill,
  restoredDraft,
  onReset,
  rememberSettings,
  blockers,
  submitting,
  error,
  lastJobId,
  submitLabel = 'Generate',
  submittingLabel = 'Submitting\u2026',
  onSubmit,
  children,
}) {
  const formRef = useRef(null);
  const { addToast } = useToast();
  const previousJobIdRef = useRef(lastJobId);

  useEffect(() => {
    if (lastJobId && lastJobId !== previousJobIdRef.current) {
      rememberSettings?.();
      addToast({ tone: 'success', message: `Submitted \u2014 job_id ${lastJobId}` });
    }
    previousJobIdRef.current = lastJobId;
  }, [lastJobId, rememberSettings, addToast]);

  useEffect(() => {
    function handleKeyDown(event) {
      const isSubmitCombo = (event.metaKey || event.ctrlKey) && event.key === 'Enter';
      if (!isSubmitCombo) return;
      const form = formRef.current;
      if (form && form.contains(document.activeElement)) {
        event.preventDefault();
        form.requestSubmit();
      }
    }
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <form ref={formRef} className="flex flex-col gap-6 pb-4" onSubmit={onSubmit}>
      <div className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-lg font-semibold text-text">{title}</h2>
          {endpoint && <code className="rounded bg-surface-2 px-1.5 py-0.5 text-[11px] text-muted">{endpoint}</code>}
        </div>
        {description && <p className="text-sm text-muted">{description}</p>}
        <RequiredLegend />
      </div>

      {prefillJob && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-accent/50 bg-surface px-4 py-3">
          <p className="text-sm text-text">
            Prefilled from job <code className="font-mono text-xs">{prefillJob.job_id}</code> ({prefillJob.status}).
            Edit anything, then submit &mdash; this creates a new job; the original stays in history.
          </p>
          <Button type="button" variant="ghost" size="sm" onClick={onClearPrefill}>
            Clear
          </Button>
        </div>
      )}

      {!prefillJob && restoredDraft && <p className="text-xs text-muted">Draft restored from your last visit.</p>}

      <div className="flex flex-col gap-6">{children}</div>

      <div className="sticky bottom-0 z-10 -mx-1 flex flex-col gap-2 border-t border-border bg-bg/95 px-1 pt-3 backdrop-blur">
        {blockers.length > 0 && (
          <details className="text-sm">
            <summary className="cursor-pointer text-muted">Can&apos;t submit yet ({blockers.length})</summary>
            <ul className="mt-1 list-disc pl-5">
              {blockers.map((message) => (
                <li key={message} className="text-danger">
                  {message}
                </li>
              ))}
            </ul>
          </details>
        )}
        {error && <p className="text-sm text-danger">{error}</p>}
        {lastJobId && <p className="text-sm text-success">Submitted: job_id {lastJobId}</p>}
        <div className="flex flex-wrap items-center gap-3 pb-3">
          <Button type="submit" disabled={submitting || blockers.length > 0}>
            {submitting ? submittingLabel : submitLabel}
          </Button>
          <Button type="button" variant="ghost" onClick={onReset}>
            Reset
          </Button>
          <span className="ml-auto hidden text-xs text-muted sm:inline">Ctrl/Cmd+Enter to submit</span>
        </div>
      </div>
    </form>
  );
}
