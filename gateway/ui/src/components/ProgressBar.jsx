import { formatBytes } from '../utils.js';

const BAR_CLASS =
  'h-1.5 w-40 overflow-hidden rounded-full [&::-webkit-progress-bar]:bg-surface-2 [&::-webkit-progress-value]:bg-accent [&::-moz-progress-bar]:bg-accent';

// Native <progress> plus a human-readable byte count -- used for both
// upload (AssetUpload.jsx) and download (ResultPreview.jsx) progress.
// `indeterminate` covers a phase where there's genuinely nothing to report
// yet (e.g. the server processing an already-fully-sent upload, or a
// download whose total size isn't known upfront).
export default function ProgressBar({ sentBytes, totalBytes, indeterminate, label }) {
  if (indeterminate || !totalBytes) {
    return (
      <div className="flex items-center gap-2">
        <progress className={BAR_CLASS} />
        <span className="text-xs text-muted">{label}</span>
      </div>
    );
  }
  const percent = Math.min(100, Math.round((sentBytes / totalBytes) * 100));
  return (
    <div className="flex items-center gap-2">
      <progress value={sentBytes} max={totalBytes} className={BAR_CLASS} />
      <span className="text-xs text-muted">
        {label}: {formatBytes(sentBytes)} / {formatBytes(totalBytes)} ({percent}%)
      </span>
    </div>
  );
}
