import { useEffect, useState } from 'react';
import { getQwen38Server, postJSON, warmQwen38Server } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Textarea from '../components/ui/Textarea.jsx';
import Input from '../components/ui/Input.jsx';
import Switch from '../components/ui/Switch.jsx';
import Button from '../components/ui/Button.jsx';

const MODEL = 'Qwen38GenerateRequest';
const FORM_ID = 'qwen38-generate';
const SERVER_POLL_MS = 5000;

const DEFAULTS = {
  system: '',
  prompt: '',
  jsonSchema: '',
  thinking: true,
  includeReasoning: false,
  maxTokens: '4096',
  temperature: '0.6',
  seed: '',
};

const REMEMBER_KEYS = ['thinking', 'includeReasoning', 'maxTokens', 'temperature'];

// Retry prefill: this form holds one optional system prompt and one user
// prompt. A multi-message request keeps its first system message and its
// last user message only.
function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  const messages = initialRequest.messages || [];
  const system = messages.find((m) => m.role === 'system');
  const lastUser = [...messages].reverse().find((m) => m.role === 'user');
  return {
    system: system?.content ?? '',
    prompt: lastUser?.content ?? '',
    jsonSchema: initialRequest.json_schema ? JSON.stringify(initialRequest.json_schema, null, 2) : '',
    thinking: initialRequest.thinking ?? true,
    includeReasoning: initialRequest.include_reasoning ?? false,
    maxTokens: String(initialRequest.max_tokens ?? 4096),
    temperature: String(initialRequest.temperature ?? 0.6),
    seed: initialRequest.seed == null ? '' : String(initialRequest.seed),
  };
}

// Returns { schema, error }: the parsed JSON Schema object, or why the text
// is not usable. Empty text means "no schema" (free-text answer).
function parseSchema(text) {
  if (!text.trim()) return { schema: null, error: null };
  try {
    const parsed = JSON.parse(text);
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { schema: null, error: 'The JSON schema must be a JSON object.' };
    }
    return { schema: parsed, error: null };
  } catch (err) {
    return { schema: null, error: `The JSON schema is not valid JSON: ${err.message}` };
  }
}

const STATE_TEXT = {
  stopped: 'Server stopped -- the first request starts it (a few minutes).',
  starting: 'Server starting -- requests wait until it is ready.',
  ready: 'Server ready -- answers take seconds.',
};

// One line showing the shared model server's state, with a button to start
// it ahead of time. Polls GET /v1/qwen38/server while the form is open.
function ServerStatusLine() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [warming, setWarming] = useState(false);

  async function refresh() {
    try {
      setStatus(await getQwen38Server());
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, SERVER_POLL_MS);
    return () => clearInterval(timer);
  }, []);

  async function handleWarm() {
    setWarming(true);
    try {
      setStatus(await warmQwen38Server());
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setWarming(false);
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2 text-xs text-muted">
      <span>{error ? `Could not read server status: ${error}` : status ? STATE_TEXT[status.state] : 'Checking server...'}</span>
      {status?.state === 'ready' && status.idle_shutdown_at && (
        <span>Stops when idle at {new Date(status.idle_shutdown_at).toLocaleTimeString()}.</span>
      )}
      {status?.state === 'stopped' && (
        <Button type="button" variant="secondary" size="sm" onClick={handleWarm} disabled={warming}>
          {warming ? 'Starting...' : 'Warm up'}
        </Button>
      )}
    </div>
  );
}

// Maps onto Qwen38GenerateRequest (services/qwen38/schemas.py): a text-only
// model with an optional JSON Schema. See TextToVideoForm.jsx for how
// initialRequest/useFormState/FormShell fit together.
export default function Qwen38GenerateForm({ onSubmitted, initialRequest = null, prefillJob = null, onClearPrefill }) {
  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(FORM_ID, {
    defaults: DEFAULTS,
    fromRequest: fromRequest(initialRequest),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${FORM_ID}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { system, prompt, jsonSchema, thinking, includeReasoning, maxTokens, temperature, seed } = values;
  const { schema, error: schemaError } = parseSchema(jsonSchema);

  const messages = [];
  if (system.trim()) messages.push({ role: 'system', content: system });
  if (prompt.trim()) messages.push({ role: 'user', content: prompt });

  const payload = {
    messages: messages.length > 0 ? messages : undefined,
    thinking,
    include_reasoning: thinking && includeReasoning,
    max_tokens: Number(maxTokens),
    temperature: Number(temperature),
  };
  if (schema) payload.json_schema = schema;
  if (seed.trim()) payload.seed = Number(seed);

  const blockers = [...missingRequired(schemas, MODEL, payload)];
  if (!prompt.trim()) blockers.push('Write a prompt.');
  if (schemaError) blockers.push(schemaError);
  if (!Number.isInteger(payload.max_tokens) || payload.max_tokens < 1) blockers.push('max_tokens must be a whole number of 1 or more.');
  if (!Number.isFinite(payload.temperature)) blockers.push('temperature must be a number.');
  if (payload.seed !== undefined && !Number.isInteger(payload.seed)) blockers.push('seed must be a whole number.');

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/qwen38/generate', payload));
  }

  return (
    <FormShell
      title="Qwen3.8: text and JSON"
      description="Text-only reasoning model (Qwen3.8-27B, FP8). Ask for free text, or add a JSON Schema to get an answer that always matches it. The first request after an idle period waits a few minutes while the server starts."
      endpoint="POST /v1/qwen38/generate"
      prefillJob={prefillJob}
      onClearPrefill={onClearPrefill}
      restoredDraft={restoredDraft}
      onReset={reset}
      rememberSettings={rememberSettings}
      blockers={blockers}
      submitting={submitting}
      error={error}
      lastJobId={lastJobId}
      submitLabel="Generate"
      onSubmit={handleSubmit}
    >
      <ServerStatusLine />

      <Section title="Prompt">
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-muted">System prompt (optional)</span>
          <Textarea
            rows={2}
            value={system}
            onChange={(event) => setField('system', event.target.value)}
            placeholder="You plan video scenes. Answer only with JSON."
          />
        </label>
        <FieldLabel model={MODEL} field="messages" text="Prompt" required>
          <Textarea rows={6} value={prompt} onChange={(event) => setField('prompt', event.target.value)} />
        </FieldLabel>
      </Section>

      <Section title="Answer format">
        <FieldLabel model={MODEL} field="json_schema" text="JSON Schema (optional -- leave empty for free text)">
          <Textarea
            rows={6}
            value={jsonSchema}
            onChange={(event) => setField('jsonSchema', event.target.value)}
            className="font-mono text-xs"
            placeholder={'{"type": "object", "properties": {"scenes": {"type": "array"}}, "required": ["scenes"]}'}
            spellCheck={false}
          />
        </FieldLabel>
        {schemaError && <p className="text-xs text-danger">{schemaError}</p>}
        <Switch checked={thinking} onChange={(value) => setField('thinking', value)} label="Thinking (better on hard planning, slower)" />
        <Switch
          checked={includeReasoning}
          onChange={(value) => setField('includeReasoning', value)}
          label="Show the reasoning in the result"
          disabled={!thinking}
        />
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <FieldLabel model={MODEL} field="max_tokens" text="max_tokens (reasoning counts toward it)">
          <Input
            type="number"
            min="1"
            step="1"
            value={maxTokens}
            onChange={(event) => setField('maxTokens', event.target.value)}
            className="max-w-[10rem]"
          />
        </FieldLabel>
        <FieldLabel model={MODEL} field="temperature" text="temperature">
          <Input
            type="number"
            min="0"
            max="2"
            step="0.1"
            value={temperature}
            onChange={(event) => setField('temperature', event.target.value)}
            className="max-w-[8rem]"
          />
        </FieldLabel>
        <FieldLabel model={MODEL} field="seed" text="seed (optional)">
          <Input
            type="number"
            min="0"
            step="1"
            value={seed}
            onChange={(event) => setField('seed', event.target.value)}
            className="max-w-[10rem]"
          />
        </FieldLabel>
      </CollapsibleSection>
    </FormShell>
  );
}
