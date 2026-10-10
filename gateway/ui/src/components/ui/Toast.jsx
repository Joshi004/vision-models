import { createContext, useCallback, useContext, useState } from 'react';

const ToastContext = createContext(null);

let idCounter = 0;

const TONE_BORDER = {
  neutral: 'border-border',
  success: 'border-success',
  danger: 'border-danger',
};

function ToastViewport({ toasts, onDismiss }) {
  if (toasts.length === 0) return null;
  return (
    <div className="fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`rounded-lg border ${TONE_BORDER[toast.tone] ?? TONE_BORDER.neutral} bg-surface px-4 py-3 shadow-lg`}
        >
          <div className="flex items-start justify-between gap-2">
            <p className="text-sm text-text">{toast.message}</p>
            <button
              type="button"
              onClick={() => onDismiss(toast.id)}
              className="text-muted hover:text-text"
              aria-label="Dismiss"
            >
              &times;
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

// Minimal toast/notification system: ToastProvider holds the current
// list and renders the fixed-position viewport; any descendant calls
// useToast().addToast({ tone, message }) -- used for a submit
// confirmation (FormShell.jsx) and "job succeeded/failed" notifications
// (JobsPanel.jsx). Each toast auto-dismisses after `durationMs`.
export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const dismiss = useCallback((id) => {
    setToasts((previous) => previous.filter((toast) => toast.id !== id));
  }, []);

  const addToast = useCallback(
    ({ tone = 'neutral', message, durationMs = 5000 }) => {
      const id = ++idCounter;
      setToasts((previous) => [...previous, { id, tone, message }]);
      if (durationMs > 0) {
        setTimeout(() => dismiss(id), durationMs);
      }
      return id;
    },
    [dismiss],
  );

  return (
    <ToastContext.Provider value={{ addToast, dismiss }}>
      {children}
      <ToastViewport toasts={toasts} onDismiss={dismiss} />
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) throw new Error('useToast must be used within a ToastProvider');
  return context;
}
