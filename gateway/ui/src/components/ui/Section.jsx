// A titled, always-visible group of fields within a form (e.g. "Prompt",
// "Settings") -- unlike CollapsibleSection ("Advanced"), this never
// collapses; it's just a heading plus consistent vertical spacing.
export default function Section({ title, description, children }) {
  return (
    <div className="flex flex-col gap-4">
      {title && (
        <div>
          <h3 className="text-sm font-semibold text-text">{title}</h3>
          {description && <p className="text-xs text-muted">{description}</p>}
        </div>
      )}
      {children}
    </div>
  );
}
