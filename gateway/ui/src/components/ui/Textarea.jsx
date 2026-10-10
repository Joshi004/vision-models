import { useEffect, useRef } from 'react';

const BASE =
  'w-full resize-none rounded-md border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent/50 disabled:opacity-50';

// A <textarea> that grows to fit its content instead of scrolling
// internally -- `rows` still sets the initial/minimum height. Every
// prompt/negative-prompt field in the six recipe forms uses this.
export default function Textarea({ className = '', value, onChange, rows = 3, ...props }) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }, [value]);

  return (
    <textarea ref={ref} className={`${BASE} ${className}`} value={value} onChange={onChange} rows={rows} {...props} />
  );
}
