const BASE =
  'w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent/50 disabled:opacity-50';

export default function Input({ className = '', ...props }) {
  return <input className={`${BASE} ${className}`} {...props} />;
}
