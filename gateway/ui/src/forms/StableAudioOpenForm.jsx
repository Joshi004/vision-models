import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { assetBlocker, reusedAsset, useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Textarea from '../components/ui/Textarea.jsx';
import Input from '../components/ui/Input.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';

// The two Stable Audio Open endpoints share most fields, so one component serves both; `VARIANTS` holds the
// differences. Each maps 1:1 onto its request model (services/stable_audio_open/schemas.py).
const VARIANTS = {
  generate: {
    model: 'StableAudioOpenGenerateRequest',
    formId: 'stable-audio-open-generate',
    endpoint: '/v1/stable-audio-open/audio/generate',
    title: 'Stable Audio Open 1.0: text to audio',
    description:
      'Generate a sound effect, foley hit, drum loop, ambience or short clip (up to 47 s). Write a short, concrete description in English. No vocals; for longer music use Stable Audio 3.',
    submitLabel: 'Generate audio',
  },
  transform: {
    model: 'StableAudioOpenTransformRequest',
    formId: 'stable-audio-open-transform',
    endpoint: '/v1/stable-audio-open/audio/transform',
    title: 'Stable Audio Open 1.0: transform audio',
    description:
      'Re-imagine an existing recording from a prompt. Only its first 47 s are used. A low noise level stays close to the source; a higher one follows the prompt more. Keep the seed fixed while trying values.',
    submitLabel: 'Transform audio',
  },
};

// Prompts and source recordings are never remembered between sessions; the settings are -- see storage.js.
const REMEMBER_KEYS = ['steps', 'cfgScale', 'seed', 'partition'];

function makeDefaults() {
  return {
    prompt: '',
    negativePrompt: '',
    duration: '30',
    source: null,
    initNoiseLevel: '0.5',
    steps: '100',
    cfgScale: '7',
    seed: 42,
    partition: '',
  };
}

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    prompt: initialRequest.prompt ?? '',
    negativePrompt: initialRequest.negative_prompt ?? '',
    duration: String(initialRequest.duration ?? ''),
    source: reusedAsset(initialRequest.source_audio_asset_id),
    initNoiseLevel: String(initialRequest.init_noise_level ?? 0.5),
    steps: String(initialRequest.num_inference_steps ?? 100),
    cfgScale: String(initialRequest.cfg_scale ?? 7),
    seed: initialRequest.seed ?? 42,
    partition: initialRequest.partition ?? '',
  };
}

function StableAudioOpenForm({ variantId, defaultPartition, onSubmitted, initialRequest = null, prefillJob = null, onClearPrefill }) {
  const variant = VARIANTS[variantId];
  const { model } = variant;
  const isTransform = variantId === 'transform';

  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(variant.formId, {
    defaults: makeDefaults(),
    fromRequest: fromRequest(initialRequest),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${variant.formId}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { prompt, negativePrompt, duration, source, initNoiseLevel, steps, cfgScale, seed, partition } = values;

  const payload = { prompt, seed: seed === '' ? undefined : Number(seed) };
  if (negativePrompt.trim()) payload.negative_prompt = negativePrompt;
  if (duration !== '') payload.duration = Number(duration);
  if (steps !== '') payload.num_inference_steps = Number(steps);
  if (cfgScale !== '') payload.cfg_scale = Number(cfgScale);
  if (isTransform) {
    payload.source_audio_asset_id = source?.status === 'ready' ? source.assetId : undefined;
    if (initNoiseLevel !== '') payload.init_noise_level = Number(initNoiseLevel);
  }
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [...missingRequired(schemas, model, payload, isTransform ? ['source_audio_asset_id'] : [])];
  if (prompt.trim() === '' && !blockers.some((message) => message.startsWith('prompt'))) {
    blockers.push('prompt is required.');
  }
  if (isTransform) blockers.push(...assetBlocker(source, 'the source recording'));

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON(variant.endpoint, payload));
  }

  return (
    <FormShell
      title={variant.title}
      description={variant.description}
      endpoint={`POST ${variant.endpoint}`}
      prefillJob={prefillJob}
      onClearPrefill={onClearPrefill}
      restoredDraft={restoredDraft}
      onReset={reset}
      rememberSettings={rememberSettings}
      blockers={blockers}
      submitting={submitting}
      error={error}
      lastJobId={lastJobId}
      submitLabel={variant.submitLabel}
      onSubmit={handleSubmit}
    >
      {isTransform && (
        <Section title="Source recording">
          <AssetUpload
            model={model}
            field="source_audio_asset_id"
            label="source recording"
            accept="audio/*"
            value={source}
            onChange={(next) => setField('source', next)}
            required
          />
        </Section>
      )}

      <Section title="Prompt">
        <FieldLabel model={model} field="prompt" text="prompt" required>
          <Textarea
            value={prompt}
            onChange={(event) => setField('prompt', event.target.value)}
            rows={3}
            placeholder="128 BPM tech house drum loop"
          />
        </FieldLabel>
        <FieldLabel model={model} field="negative_prompt" text="negative_prompt (optional)">
          <Textarea value={negativePrompt} onChange={(event) => setField('negativePrompt', event.target.value)} rows={2} placeholder="Low quality." />
        </FieldLabel>
      </Section>

      <Section title="Settings">
        <div className="grid grid-cols-2 gap-3">
          <FieldLabel model={model} field="duration" text={isTransform ? 'duration (s, max 47; empty = source length)' : 'duration (s, max 47)'}>
            <Input type="number" min="1" max="47" step="1" value={duration} onChange={(event) => setField('duration', event.target.value)} />
          </FieldLabel>
          {isTransform && (
            <FieldLabel model={model} field="init_noise_level" text="init_noise_level (0.05-20; ~0.1 stays close)">
              <Input type="number" min="0.05" max="20" step="0.05" value={initNoiseLevel} onChange={(event) => setField('initNoiseLevel', event.target.value)} />
            </FieldLabel>
          )}
        </div>
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <div className="grid grid-cols-2 gap-3">
          <FieldLabel model={model} field="num_inference_steps" text="num_inference_steps (default 100)">
            <Input type="number" min="1" max="250" step="1" value={steps} onChange={(event) => setField('steps', event.target.value)} />
          </FieldLabel>
          <FieldLabel model={model} field="cfg_scale" text="cfg_scale (default 7)">
            <Input type="number" min="0" max="30" step="0.5" value={cfgScale} onChange={(event) => setField('cfgScale', event.target.value)} />
          </FieldLabel>
        </div>
        <SeedField model={model} value={seed} onChange={(value) => setField('seed', value)} />
        <PartitionField model={model} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}

export function StableAudioOpenGenerateForm(props) {
  return <StableAudioOpenForm variantId="generate" {...props} />;
}

export function StableAudioOpenTransformForm(props) {
  return <StableAudioOpenForm variantId="transform" {...props} />;
}
