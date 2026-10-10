import { useState } from 'react';
import CollapsibleSection from './ui/CollapsibleSection.jsx';
import Button from './ui/Button.jsx';
import { CopyIcon } from './icons.jsx';

// mm:ss formatting for segment/word timestamps -- always minutes:seconds
// (unlike utils.js's formatDuration, which grows an hours part for long
// elapsed/typical job times), since these are positions within a single
// recording, easiest to scan down a long list when every row has the same
// two-part shape.
function formatTimestamp(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '--:--';
  const totalSeconds = Math.floor(seconds);
  const minutes = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${minutes}:${String(secs).padStart(2, '0')}`;
}

// Renders a parsed Parakeet transcript in place of ResultPreview.jsx's
// usual <audio>/<video> element -- this result is text with timestamps,
// not a playable media file. `transcript` is already-parsed JSON
// (ResultPreview does the fetch + JSON.parse; this component is a pure
// renderer) matching services/parakeet/openapi_docs.py's TRANSCRIBE
// text: { transcription, processing_time, word_timestamps[{word,start,end}],
// segment_timestamps[{text,start,end,word_count}], metadata{total_segments,
// total_words,duration} }.
//
// The word-level list is the one part gated behind a closed-by-default
// CollapsibleSection, and only actually built (the `words.map(...)` below)
// once `wordsOpen` is true -- a long recording can produce tens of
// thousands of words, and CollapsibleSection alone only skips *mounting*
// its children when collapsed, not evaluating the JSX expression that
// builds them in the first place.
export default function TranscriptView({ transcript }) {
  const [copyState, setCopyState] = useState(null);
  const [wordsOpen, setWordsOpen] = useState(false);

  const {
    transcription,
    processing_time: processingTime,
    word_timestamps: words = [],
    segment_timestamps: segments = [],
    metadata,
  } = transcript;

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(transcription);
      setCopyState('copied');
    } catch {
      // See JobRow.jsx's handleCopyConfig for why this can fail even on a
      // legitimate click -- the full text is still selectable below either way.
      setCopyState('failed');
    }
    setTimeout(() => setCopyState(null), 2000);
  }

  return (
    <div className="flex w-full max-w-2xl flex-col gap-3 rounded-lg border border-border bg-surface p-3">
      <div className="flex items-start justify-between gap-3">
        <p className="whitespace-pre-wrap text-sm text-text">{transcription || '(no speech detected)'}</p>
        <Button type="button" variant="secondary" size="sm" onClick={handleCopy} className="shrink-0">
          <CopyIcon className="h-3.5 w-3.5" />
          Copy
        </Button>
      </div>
      {copyState === 'copied' && <span className="text-xs text-success">Copied.</span>}
      {copyState === 'failed' && (
        <span className="text-xs text-danger">Couldn&apos;t copy automatically -- select the text above instead.</span>
      )}

      {metadata && (
        <p className="text-xs text-muted">
          {metadata.total_words} word{metadata.total_words === 1 ? '' : 's'}, {metadata.total_segments} segment
          {metadata.total_segments === 1 ? '' : 's'}, {formatTimestamp(metadata.duration)} duration
          {typeof processingTime === 'number' && ` \u2014 processed in ${processingTime.toFixed(2)}s`}
        </p>
      )}

      {segments.length > 0 && (
        <div className="flex max-h-64 flex-col gap-1 overflow-y-auto rounded-md border border-border bg-bg p-2">
          {segments.map((segment, index) => (
            <p key={index} className="text-sm text-text">
              <span className="mr-2 font-mono text-xs text-muted">
                {formatTimestamp(segment.start)}-{formatTimestamp(segment.end)}
              </span>
              {segment.text}
            </p>
          ))}
        </div>
      )}

      {words.length > 0 && (
        <CollapsibleSection title={`Word-level timestamps (${words.length})`} open={wordsOpen} onToggle={setWordsOpen}>
          {wordsOpen && (
            <div className="flex max-h-64 flex-col gap-0.5 overflow-y-auto">
              {words.map((word, index) => (
                <span key={index} className="text-xs text-text">
                  <span className="mr-1.5 font-mono text-muted">{formatTimestamp(word.start)}</span>
                  {word.word}
                </span>
              ))}
            </div>
          )}
        </CollapsibleSection>
      )}
    </div>
  );
}
