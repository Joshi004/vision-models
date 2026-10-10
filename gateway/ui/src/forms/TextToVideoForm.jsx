import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import {
  defaultShape,
  imageEntriesFromRequest,
  shapeBlockers,
  shapeFromRequest,
  toImageConditioningPayload,
  toShapePayload,
  useSubmitJob,
} from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Textarea from '../components/ui/Textarea.jsx';
import SegmentedControl from '../components/ui/SegmentedControl.jsx';
import Switch from '../components/ui/Switch.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import FieldHeading from '../components/FieldHeading.jsx';
import SeedField from '../components/SeedField.jsx';
import ImageConditioningList from '../components/ImageConditioningList.jsx';
import PartitionField from '../components/PartitionField.jsx';
import VideoShapeFields from '../components/VideoShapeFields.jsx';

const MODEL = 'TextToVideoRequest';
const FORM_ID = 'ltx-generate';

const DEFAULTS = {
  prompt: '',
  mode: 'fast',
  negativePrompt: '',
  seed: 10,
  enhancePrompt: false,
  images: [],
  shape: defaultShape(),
  partition: '',
};

// Only these carry over as this form's defaults the next time it's
// opened fresh (a new tab, or after Reset) -- see storage.js. The
// prompt, negative_prompt, and any uploaded images are intentionally left
// out: they're specific to one generation, not a "setting" worth reusing.
const REMEMBER_KEYS = ['mode', 'seed', 'enhancePrompt', 'shape', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    prompt: initialRequest.prompt ?? '',
    mode: initialRequest.mode ?? 'fast',
    negativePrompt: initialRequest.negative_prompt ?? '',
    seed: initialRequest.seed ?? 10,
    enhancePrompt: initialRequest.enhance_prompt ?? false,
    images: imageEntriesFromRequest(initialRequest.images),
    shape: shapeFromRequest(initialRequest),
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto TextToVideoRequest (services/ltx/schemas.py). `initialRequest`
// -- a previously submitted job's own stored request body -- prefills every
// field below for the Retry flow (see App.jsx/JobRow.jsx); omit it (or
// pass null) for the normal blank-form case. Field values otherwise come
// from useFormState (see storage.js): a sessionStorage draft restores
// everything after a refresh, and mode/seed/enhance_prompt/shape/partition
// (not the prompt or uploaded images) carry over as defaults for a fresh
// form via localStorage.
export default function TextToVideoForm({
  defaultPartition,
  onSubmitted,
  initialRequest = null,
  prefillJob = null,
  onClearPrefill,
}) {
  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(FORM_ID, {
    defaults: DEFAULTS,
    fromRequest: fromRequest(initialRequest),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${FORM_ID}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { prompt, mode, negativePrompt, seed, enhancePrompt, images, shape, partition } = values;

  const payload = {
    prompt,
    mode,
    seed: seed === '' ? undefined : Number(seed),
    enhance_prompt: enhancePrompt,
    images: toImageConditioningPayload(images),
    ...toShapePayload(shape),
  };
  if (mode === 'quality' && negativePrompt.trim()) payload.negative_prompt = negativePrompt;
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [...missingRequired(schemas, MODEL, payload), ...shapeBlockers(shape)];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/ltx/videos/generate', payload));
  }

  return (
    <FormShell
      title="Text / image to video"
      description="Fast mode ignores negative_prompt."
      endpoint="POST /v1/ltx/videos/generate"
      prefillJob={prefillJob}
      onClearPrefill={onClearPrefill}
      restoredDraft={restoredDraft}
      onReset={reset}
      rememberSettings={rememberSettings}
      blockers={blockers}
      submitting={submitting}
      error={error}
      lastJobId={lastJobId}
      submitLabel="Generate video"
      onSubmit={handleSubmit}
    >
      <Section title="Prompt">
        <FieldLabel model={MODEL} field="prompt" text="prompt">
          <Textarea value={prompt} onChange={(event) => setField('prompt', event.target.value)} rows={4} />
        </FieldLabel>
        <ImageConditioningList
          label="Images (optional \u2014 image-to-video)"
          entries={images}
          onChange={(next) => setField('images', next)}
        />
      </Section>

      <Section title="Settings">
        <div className="flex flex-col gap-1.5">
          <FieldHeading model={MODEL} field="mode" text="mode" />
          <SegmentedControl
            name="mode"
            value={mode}
            onChange={(value) => setField('mode', value)}
            options={[
              { value: 'fast', label: 'Fast (distilled)' },
              { value: 'quality', label: 'Quality (slower)' },
            ]}
          />
        </div>
        <Switch
          checked={enhancePrompt}
          onChange={(checked) => setField('enhancePrompt', checked)}
          label="Enhance prompt (rewrite with Gemma before generating)"
        />
        <VideoShapeFields model={MODEL} shape={shape} onChange={(next) => setField('shape', next)} />
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <FieldLabel
          model={MODEL}
          field="negative_prompt"
          text={`negative_prompt${mode === 'fast' ? ' (ignored in fast mode)' : ''}`}
        >
          <Textarea
            value={negativePrompt}
            onChange={(event) => setField('negativePrompt', event.target.value)}
            rows={2}
            disabled={mode === 'fast'}
          />
        </FieldLabel>
        <SeedField model={MODEL} value={seed} onChange={(value) => setField('seed', value)} />
        <PartitionField model={MODEL} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}
