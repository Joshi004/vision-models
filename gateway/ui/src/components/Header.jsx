function StatusPill({ health, healthError }) {
  const status = health ? 'ok' : healthError ? 'down' : 'checking';
  const label = health ? 'API online' : healthError ? 'API unreachable' : 'Checking\u2026';
  const dotClass = status === 'ok' ? 'bg-success' : status === 'down' ? 'bg-danger' : 'bg-muted';
  return (
    <div
      className="flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 text-xs text-muted"
      title={healthError || label}
    >
      <span className={`h-2 w-2 rounded-full ${dotClass}`} />
      <span>{label}</span>
      {health?.default_partition && <span className="text-muted/70">&middot; {health.default_partition}</span>}
    </div>
  );
}

// Top bar: app title and a live health pill (polled by App.jsx).
export default function Header({ health, healthError }) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-surface px-5 py-3">
      <div className="flex items-center gap-3">
        <img
          src={`${import.meta.env.BASE_URL}favicon.svg`}
          alt=""
          width={28}
          height={28}
          className="h-7 w-7 shrink-0"
        />
        <h1 className="text-base font-semibold text-text">model-api</h1>
        <StatusPill health={health} healthError={healthError} />
      </div>
    </header>
  );
}
