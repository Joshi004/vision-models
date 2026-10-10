import { useId } from 'react';
import { ChevronDownIcon } from '../icons.jsx';

// An expand/collapse section for a group of fields -- used for each
// form's "Advanced" group (see FormShell usage in forms/*.jsx). `open`/
// `onToggle` are controlled by the caller (usually backed by
// usePersistedState, see storage.js) so the open/closed state survives a
// refresh instead of always starting collapsed.
export default function CollapsibleSection({ title, open, onToggle, children }) {
  const contentId = useId();
  return (
    <div className="rounded-lg border border-border bg-surface">
      <button
        type="button"
        onClick={() => onToggle(!open)}
        aria-expanded={open}
        aria-controls={contentId}
        className="flex w-full items-center justify-between px-4 py-2.5 text-left text-sm font-medium text-text"
      >
        {title}
        <ChevronDownIcon className={`h-4 w-4 text-muted transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div id={contentId} className="flex flex-col gap-4 border-t border-border px-4 py-4">
          {children}
        </div>
      )}
    </div>
  );
}
