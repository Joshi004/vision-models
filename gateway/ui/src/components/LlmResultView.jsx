import { useState } from 'react';
import CollapsibleSection from './ui/CollapsibleSection.jsx';
import Button from './ui/Button.jsx';
import { CopyIcon } from './icons.jsx';

// Renders a parsed qwen38:generate result in place of ResultPreview.jsx's
// usual <audio>/<video> element. `result` is already-parsed JSON matching
// services/qwen38/openapi_docs.py: { content, json, reasoning,
// finish_reason, usage{prompt_tokens,completion_tokens,total_tokens},
// timings{server_wait_seconds,generation_seconds}, model{name,revision,
// precision} }.
//
// When the request had a json_schema and the answer parsed, the parsed
// JSON is shown pretty-printed; otherwise the raw text is shown.
export default function LlmResultView({ result }) {
  const [copyState, setCopyState] = useState(null);
  const [reasoningOpen, setReasoningOpen] = useState(false);

  const { content, json, reasoning, finish_reason: finishReason, usage, timings } = result;
  const text = json !== null && json !== undefined ? JSON.stringify(json, null, 2) : content || '';

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopyState('copied');
    } catch {
      // See JobRow.jsx's handleCopyConfig for why this can fail on a legitimate click.
      setCopyState('failed');
    }
    setTimeout(() => setCopyState(null), 2000);
  }

  return (
    <div className="flex w-full max-w-2xl flex-col gap-3 rounded-lg border border-border bg-surface p-3">
      <div className="flex items-start justify-between gap-3">
        <pre className="max-h-96 min-w-0 flex-1 overflow-auto whitespace-pre-wrap break-words text-sm text-text">
          {text || '(empty answer)'}
        </pre>
        <Button type="button" variant="secondary" size="sm" onClick={handleCopy} className="shrink-0">
          <CopyIcon className="h-3.5 w-3.5" />
          Copy
        </Button>
      </div>
      {copyState === 'copied' && <span className="text-xs text-success">Copied.</span>}
      {copyState === 'failed' && (
        <span className="text-xs text-danger">Couldn&apos;t copy automatically -- select the text above instead.</span>
      )}

      {finishReason === 'length' && (
        <p className="text-xs text-danger">
          The answer was cut off by max_tokens (reasoning counts toward it). Raise max_tokens or turn thinking off.
        </p>
      )}

      <p className="text-xs text-muted">
        {usage?.completion_tokens ?? '?'} tokens generated ({usage?.prompt_tokens ?? '?'} in prompt)
        {timings && ` \u2014 ${timings.generation_seconds}s generating, ${timings.server_wait_seconds}s waiting for the server`}
      </p>

      {reasoning && (
        <CollapsibleSection title="Reasoning" open={reasoningOpen} onToggle={setReasoningOpen}>
          <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs text-muted">{reasoning}</pre>
        </CollapsibleSection>
      )}
    </div>
  );
}
