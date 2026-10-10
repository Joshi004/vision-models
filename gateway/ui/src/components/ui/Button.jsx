const VARIANTS = {
  primary: 'bg-accent hover:bg-accent-2 text-white',
  secondary: 'bg-surface-2 hover:bg-border text-text border border-border',
  ghost: 'bg-transparent hover:bg-surface-2 text-muted hover:text-text',
  danger: 'bg-danger hover:bg-danger/90 text-white',
};

const SIZES = {
  sm: 'px-2.5 py-1 text-xs',
  md: 'px-4 py-2 text-sm',
};

// Shared button styling for every action in the UI (submit, cancel,
// retry, delete, toolbar toggles) -- centralizes the variant/size scale so
// new buttons stay visually consistent without repeating Tailwind classes
// at every call site.
//
// `type` is intentionally NOT defaulted here (unlike most button
// wrappers) -- every call site must say `type="button"` or
// `type="submit"` explicitly, matching this codebase's existing
// discipline of never relying on a native <button>'s default type="submit"
// inside a <form> (which would submit the form from e.g. a "Remove row" or
// "Add image" click).
export default function Button({ variant = 'primary', size = 'md', className = '', children, ...props }) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${SIZES[size]} ${VARIANTS[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
