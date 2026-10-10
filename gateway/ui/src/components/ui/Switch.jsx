// A styled checkbox rendered as a toggle switch -- the input itself stays
// a real, focusable, keyboard-toggleable <input type="checkbox">; only its
// visual box is replaced (via `peer` + `peer-checked:` on the two sibling
// spans that draw the track/thumb), so screen readers and forms still see
// a normal checkbox.
export default function Switch({ checked, onChange, label, disabled }) {
  return (
    <label className={`inline-flex items-center gap-2 ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}>
      <span className="relative inline-flex h-5 w-9 shrink-0 items-center">
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) => onChange(event.target.checked)}
          disabled={disabled}
          className="peer sr-only"
        />
        <span className="absolute inset-0 rounded-full border border-border bg-surface-2 transition-colors peer-checked:border-accent peer-checked:bg-accent" />
        <span className="absolute left-0.5 h-4 w-4 rounded-full bg-white transition-transform peer-checked:translate-x-4" />
      </span>
      {label && <span className="text-sm text-text">{label}</span>}
    </label>
  );
}
