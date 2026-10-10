// Small inline stroke icons (24x24 viewBox, currentColor) for the sidebar
// and a handful of action buttons -- plain SVG paths rather than an icon
// library dependency, so the whole icon set is a few tiny components
// instead of a new npm package.
function Icon({ children, className = 'h-4 w-4' }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

// LTX \u00b7 Text/image to video
export function FilmIcon(props) {
  return (
    <Icon {...props}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <line x1="3" y1="9" x2="21" y2="9" />
      <line x1="3" y1="15" x2="21" y2="15" />
      <line x1="8" y1="4" x2="8" y2="9" />
      <line x1="16" y1="4" x2="16" y2="9" />
    </Icon>
  );
}

// LTX \u00b7 Keyframe interpolation
export function LayersIcon(props) {
  return (
    <Icon {...props}>
      <polygon points="12 3 21 8 12 13 3 8 12 3" />
      <polyline points="3 13 12 18 21 13" />
      <polyline points="3 17.5 12 22 21 17.5" />
    </Icon>
  );
}

// LTX \u00b7 Audio to video
export function WaveIcon(props) {
  return (
    <Icon {...props}>
      <path d="M3 12h2l2 -6 3 12 3 -14 3 12 2 -4h3" />
    </Icon>
  );
}

// LTX \u00b7 Retake
export function RedoIcon(props) {
  return (
    <Icon {...props}>
      <path d="M4 12a8 8 0 1 0 3-6.2" />
      <polyline points="3 3 4 8 9 7" />
    </Icon>
  );
}

// LTX \u00b7 Text to audio
export function MusicIcon(props) {
  return (
    <Icon {...props}>
      <path d="M9 18V5l11 -2v13" />
      <circle cx="6" cy="18" r="3" />
      <circle cx="17" cy="16" r="3" />
    </Icon>
  );
}

// Wan-Animate \u00b7 Replace character
export function SwapUserIcon(props) {
  return (
    <Icon {...props}>
      <circle cx="9" cy="7" r="3" />
      <path d="M3 20c0 -3.3 2.7 -6 6 -6s6 2.7 6 6" />
      <path d="M16 4l3 3l-3 3" />
      <path d="M21 7h-7" />
    </Icon>
  );
}

// RVC \u00b7 Convert voice
export function MicIcon(props) {
  return (
    <Icon {...props}>
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <line x1="12" y1="18" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
    </Icon>
  );
}

// RVC \u00b7 Batch convert
export function StackIcon(props) {
  return (
    <Icon {...props}>
      <rect x="7" y="8" width="13" height="13" rx="2" />
      <path d="M4 15V6a2 2 0 0 1 2 -2h9" />
    </Icon>
  );
}

// Parakeet \u00b7 Transcribe
export function TranscriptIcon(props) {
  return (
    <Icon {...props}>
      <path d="M7 3h7l4 4v13a1 1 0 0 1 -1 1H7a1 1 0 0 1 -1 -1V4a1 1 0 0 1 1 -1z" />
      <polyline points="14 3 14 7 18 7" />
      <line x1="8" y1="12" x2="16" y2="12" />
      <line x1="8" y1="15" x2="16" y2="15" />
      <line x1="8" y1="18" x2="13" y2="18" />
    </Icon>
  );
}

// Job history
export function HistoryIcon(props) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="9" />
      <polyline points="12 7 12 12 16 14" />
    </Icon>
  );
}

export function CopyIcon(props) {
  return (
    <Icon {...props}>
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15V5a2 2 0 0 1 2 -2h10" />
    </Icon>
  );
}

export function TrashIcon(props) {
  return (
    <Icon {...props}>
      <polyline points="4 7 20 7" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
      <path d="M6 7l1 13a2 2 0 0 0 2 2h6a2 2 0 0 0 2 -2l1 -13" />
      <path d="M9 7V4a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v3" />
    </Icon>
  );
}

export function XIcon(props) {
  return (
    <Icon {...props}>
      <line x1="5" y1="5" x2="19" y2="19" />
      <line x1="19" y1="5" x2="5" y2="19" />
    </Icon>
  );
}

export function DiceIcon(props) {
  return (
    <Icon {...props}>
      <rect x="3" y="3" width="18" height="18" rx="3" />
      <circle cx="8" cy="8" r="1" fill="currentColor" stroke="none" />
      <circle cx="16" cy="8" r="1" fill="currentColor" stroke="none" />
      <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
      <circle cx="8" cy="16" r="1" fill="currentColor" stroke="none" />
      <circle cx="16" cy="16" r="1" fill="currentColor" stroke="none" />
    </Icon>
  );
}

export function ChevronDownIcon(props) {
  return (
    <Icon {...props}>
      <polyline points="6 9 12 15 18 9" />
    </Icon>
  );
}

export function UploadIcon(props) {
  return (
    <Icon {...props}>
      <path d="M12 16V4" />
      <polyline points="7 8 12 3 17 8" />
      <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-3" />
    </Icon>
  );
}

// Settings popover trigger in the header (rendered as "sliders", a
// standard settings glyph that's simple enough to draw exactly right by
// hand -- three lines, each with one small filled "handle" circle).
export function SettingsIcon(props) {
  return (
    <Icon {...props}>
      <line x1="4" y1="7" x2="20" y2="7" />
      <circle cx="9" cy="7" r="1.6" fill="currentColor" stroke="none" />
      <line x1="4" y1="12" x2="20" y2="12" />
      <circle cx="15" cy="12" r="1.6" fill="currentColor" stroke="none" />
      <line x1="4" y1="17" x2="20" y2="17" />
      <circle cx="11" cy="17" r="1.6" fill="currentColor" stroke="none" />
    </Icon>
  );
}
