const BASE =
  'w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text focus:outline-none focus:ring-2 focus:ring-accent/50 disabled:opacity-50';

export default function Select({ className = '', children, ...props }) {
  return (
    <select className={`${BASE} ${className}`} {...props}>
      {children}
    </select>
  );
}
