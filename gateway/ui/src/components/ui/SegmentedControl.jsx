// A row of mutually-exclusive options rendered as native radio inputs,
// visually styled as a single segmented control -- keeps native keyboard
// (arrow keys move between options within the same `name` group) and
// accessibility behavior instead of reimplementing it with plain divs.
//
// Deliberately NOT wrapped in a <label> at the call site for the group as
// a whole -- see FieldHeading.jsx for why (a native <label> only ever
// activates the *first* control nested inside it, so wrapping several
// radios in one outer label would make clicking a heading jump to
// whichever option happens to be first).
export default function SegmentedControl({ name, value, onChange, options }) {
  return (
    <div className="inline-flex flex-wrap gap-1 rounded-md border border-border bg-surface p-1" role="radiogroup">
      {options.map((option) => {
        const checked = option.value === value;
        return (
          <label
            key={option.value}
            className={`cursor-pointer rounded-[6px] px-3 py-1.5 text-sm transition-colors ${
              checked ? 'bg-accent text-white' : 'text-muted hover:bg-surface-2 hover:text-text'
            }`}
          >
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={checked}
              onChange={() => onChange(option.value)}
              className="sr-only"
            />
            {option.label}
          </label>
        );
      })}
    </div>
  );
}
