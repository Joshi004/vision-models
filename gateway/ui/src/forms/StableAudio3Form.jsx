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

// The three Stable Audio 3 endpoints share most fields, so one component serves all of them; `VARIANTS` holds
// the differences. Each maps 1:1 onto its request model (services/stable_audio3/schemas.py).
const VARIANTS = {
  generate: {
    model: 'StableAudio3GenerateRequest',
    formId: 'stable-audio3-generate',
    endpoint: '/v1/stable-audio3/audio/generate',
    title: 'Stable Audio 3: text to audio',
    description:
      'Generate instrumental music, sound effects, foley or ambience up to about 6 minutes, fast (8 steps). Write a descriptive prompt in English: genre, instruments, mood, tempo and production in a few lines. It does not sing lyrics.',
    submitLabel: 'Generate audio',
  },
  transform: {
    model: 'StableAudio3TransformRequest',
    formId: 'stable-audio3-transform',
    endpoint: '/v1/stable-audio3/audio/transform',
    title: 'Stable Audio 3: transform audio',
    description:
      'Re-imagine an existing recording from a prompt. A low noise level stays close to the source; a high one follows the prompt more. Keep the seed fixed while trying values.',
    submitLabel: 'Transform audio',
  },
  inpaint: {
    model: 'StableAudio3InpaintRequest',
    formId: 'stable-audio3-inpaint',
    endpoint: '/v1/stable-audio3/audio/inpaint',
    title: 'Stable Audio 3: inpaint or extend',
    description:
      'Regenerate one time window of a recording and keep the rest. Set the end past the recording\'s length to extend it.',
    submitLabel: 'Inpaint audio',
  },
};

// Prompts and source recordings are never remembered between sessions; the settings are -- see storage.js.
const REMEMBER_KEYS = ['steps', 'seed', 'partition'];

function makeDefaults() {
  return {
    prompt: '',
    duration: '',
    source: null,
    initNoiseLevel: '0.5',
    maskStart: '',
    maskEnd: '',
    steps: '8',
    seed: 42,
    partition: '',
  };
}

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    prompt: initialRequest.prompt ?? '',
    duration: String(initialRequest.duration ?? ''),
    source: reusedAsset(initialRequest.source_audio_asset_id),
    initNoiseLevel: String(initialRequest.init_noise_level ?? 0.5),
    maskStart: String(initialRequest.mask_start ?? ''),
    maskEnd: String(initialRequest.mask_end ?? ''),
    steps: String(initialRequest.num_inference_steps ?? 8),
    seed: initialRequest.seed ?? 42,
    partition: initialRequest.partition ?? '',
  };
}

function StableAudio3Form({ variantId, defaultPartition, onSubmitted, initialRequest = null, prefillJob = null, onClearPrefill }) {
  const variant = VARIANTS[variantId];
  const { model } = variant;
  const isGenerate = variantId === 'generate';
  const isTransform = variantId === 'transform';
  const isInpaint = variantId === 'inpaint';

  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(variant.formId, {
    // A text-to-audio request needs a length, so it starts at the API's default of 30 s; the others default to the source.
    defaults: { ...makeDefaults(), duration: isGenerate ? '30' : '' },
    fromRequest: fromRequest(initialRequest),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${variant.formId}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { prompt, duration, source, initNoiseLevel, maskStart, maskEnd, steps, seed, partition } = values;

  const payload = { prompt, seed: seed === '' ? undefined : Number(seed) };
  if (duration !== '') payload.duration = Number(duration);
  if (steps !== '') payload.num_inference_steps = Number(steps);
  if (!isGenerate) payload.source_audio_asset_id = source?.status === 'ready' ? source.assetId : undefined;
  if (isTransform && initNoiseLevel !== '') payload.init_noise_level = Number(initNoiseLevel);
  if (isInpaint) {
    if (maskStart !== '') payload.mask_start = Number(maskStart);
    if (maskEnd !== '') payload.mask_end = Number(maskEnd);
  }
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [...missingRequired(schemas, model, payload, isGenerate ? [] : ['source_audio_asset_id'])];
  if (prompt.trim() === '' && !blockers.some((message) => message.startsWith('prompt'))) {
    blockers.push('prompt is required.');
  }
  if (!isGenerate) blockers.push(...assetBlocker(source, 'the source recording'));
  if (isInpaint && maskStart !== '' && maskEnd !== '' && Number(maskEnd) <= Number(maskStart)) {
    blockers.push('mask_end must be greater than mask_start.');
  }

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
      {!isGenerate && (
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
            rows={5}
            placeholder="Warm lo-fi hip hop, dusty drums, mellow Rhodes piano, vinyl crackle, 80 BPM, relaxed and nostalgic"
          />
        </FieldLabel>
      </Section>

      <Section title="Settings">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <FieldLabel model={model} field="duration" text={isGenerate ? 'duration (s)' : 'duration (s, empty = automatic)'}>
            <Input type="number" min="1" step="1" value={duration} onChange={(event) => setField('duration', event.target.value)} />
          </FieldLabel>
          {isTransform && (
            <FieldLabel model={model} field="init_noise_level" text="init_noise_level (0-1)">
              <Input type="number" min="0" max="1" step="0.05" value={initNoiseLevel} onChange={(event) => setField('initNoiseLevel', event.target.value)} />
            </FieldLabel>
          )}
          {isInpaint && (
            <>
              <FieldLabel model={model} field="mask_start" text="mask_start (s)" required>
                <Input type="number" min="0" step="0.1" value={maskStart} onChange={(event) => setField('maskStart', event.target.value)} />
              </FieldLabel>
              <FieldLabel model={model} field="mask_end" text="mask_end (s)" required>
                <Input type="number" min="0" step="0.1" value={maskEnd} onChange={(event) => setField('maskEnd', event.target.value)} />
              </FieldLabel>
            </>
          )}
        </div>
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <FieldLabel model={model} field="num_inference_steps" text="num_inference_steps (default 8)">
          <Input type="number" min="1" max="50" step="1" value={steps} onChange={(event) => setField('steps', event.target.value)} className="max-w-[8rem]" />
        </FieldLabel>
        <SeedField model={model} value={seed} onChange={(value) => setField('seed', value)} />
        <PartitionField model={model} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}

export function StableAudio3GenerateForm(props) {
  return <StableAudio3Form variantId="generate" {...props} />;
}

export function StableAudio3TransformForm(props) {
  return <StableAudio3Form variantId="transform" {...props} />;
}

export function StableAudio3InpaintForm(props) {
  return <StableAudio3Form variantId="inpaint" {...props} />;
}
