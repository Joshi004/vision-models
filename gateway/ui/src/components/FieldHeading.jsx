import { useFieldDoc } from '../schema.js';
import { InfoBubble } from './FieldLabel.jsx';

// Like FieldLabel's text row, but rendered as a plain non-<label> heading
// -- for a group of controls where wrapping the whole group in one
// <label> would be wrong. A native <label> only ever activates the
// *first* labelable control nested inside it, so wrapping e.g. a
// multi-option SegmentedControl (several radios) in one label would make
// clicking the heading jump straight to whichever option happens to be
// first. Single-control fields (a text input, a lone checkbox) use
// FieldLabel instead, which doesn't have this problem.
export default function FieldHeading({ model, field, text, required }) {
  const doc = useFieldDoc(model, field);
  const isRequired = required ?? doc.required;

  return (
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
}
