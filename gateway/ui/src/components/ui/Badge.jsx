const TONES = {
  neutral: 'bg-surface-2 text-muted',
  accent: 'bg-accent/15 text-accent-2',
  success: 'bg-success/15 text-success',
  danger: 'bg-danger/15 text-danger',
  warning: 'bg-warning/15 text-warning',
};

// Small status pill -- used for job status (JobRow.jsx) and anywhere else
// a short, colored label is useful.
export default function Badge({ tone = 'neutral', className = '', children }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide ${TONES[tone] ?? TONES.neutral} ${className}`}
    >
      {children}
    </span>
  );
}
