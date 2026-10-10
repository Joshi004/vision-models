import { useId } from 'react';
import { useFieldDoc } from '../schema.js';

// The small "?" bubble with a hover/focus tooltip -- shared by FieldLabel
// below, FieldHeading.jsx (for control groups that can't be wrapped in a
// <label>, e.g. SegmentedControl), and AssetUpload.jsx (which also can't
// use a wrapping <label>, since it nests a real file input -- see that
// file's comment for why).
export function InfoBubble({ description }) {
  const tooltipId = useId();
  if (!description) return null;
  return (
    <span
      className="group relative inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full bg-surface-2 text-[10px] text-muted"
      tabIndex={0}
      aria-describedby={tooltipId}
    >
      <span aria-hidden="true">?</span>
      <span
        role="tooltip"
        id={tooltipId}
        className="pointer-events-none absolute bottom-full left-0 z-10 mb-1.5 w-64 rounded-md border border-border bg-surface-2 px-2.5 py-1.5 text-xs font-normal leading-snug text-text opacity-0 shadow-lg transition-opacity group-hover:opacity-100 group-focus:opacity-100 group-focus-visible:opacity-100"
      >
        {description}
      </span>
    </span>
  );
}

// A <label> that reads its required-ness and hover/focus description
// straight from the API's own OpenAPI schema (see schema.js), so neither
// can silently drift from what the API actually validates -- wrap the
// real <input>/<select>/<textarea> as children.
//
// `required` can also be forced explicitly for rules that only exist in
// the UI, not the schema itself (e.g. height/width only become required
// once "custom" shape mode is chosen -- see VideoShapeFields.jsx); when
// omitted, it falls back to whatever the schema says.
//
// `checkbox` swaps the layout to input-then-text, matching a checkbox's
// own control conventionally coming before its label text. This is safe
// to wrap in one <label> because there's exactly one control inside it;
// see FieldHeading.jsx for the multi-control case where that isn't true.
export default function FieldLabel({ model, field, text, required, checkbox, children }) {
  const doc = useFieldDoc(model, field);
  const isRequired = required ?? doc.required;

  const textRow = (
    <span className="flex items-center gap-1.5 text-xs font-medium text-muted">
      <span>
        {text}
        {isRequired && (
          <span className="text-danger" aria-hidden="true">
            {' '}
            *
          </span>
        )}
      </span>
      <InfoBubble description={doc.description} />
    </span>
  );

  if (checkbox) {
    return (
      <label className="flex flex-row items-center gap-2">
        {children}
        {textRow}
      </label>
    );
  }

  return (
    <label className="flex flex-col gap-1.5">
      {textRow}
      {children}
    </label>
  );
}

export function RequiredLegend() {
  return (
    <p className="text-xs text-muted">
      <span className="text-danger">*</span> required
    </p>
  );
}
