import { postJSON } from '../api.js';
import { missingRequired, useArrayMinItems, useSchemas } from '../schema.js';
import {
  defaultShape,
  emptyImageEntry,
  imageEntriesFromRequest,
  imageListBlockers,
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
import FieldLabel from '../components/FieldLabel.jsx';
import SeedField from '../components/SeedField.jsx';
import ImageConditioningList from '../components/ImageConditioningList.jsx';
import PartitionField from '../components/PartitionField.jsx';
import VideoShapeFields from '../components/VideoShapeFields.jsx';

const MODEL = 'KeyframeInterpolationRequest';
const FORM_ID = 'ltx-keyframe';

const DEFAULTS = {
  prompt: '',
  negativePrompt: '',
  seed: 10,
  keyframes: [emptyImageEntry(), emptyImageEntry()],
  shape: defaultShape(),
  partition: '',
};

const REMEMBER_KEYS = ['seed', 'shape', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  const keyframes = imageEntriesFromRequest(initialRequest.keyframes);
  return {
    prompt: initialRequest.prompt ?? '',
    negativePrompt: initialRequest.negative_prompt ?? '',
    seed: initialRequest.seed ?? 10,
    keyframes: keyframes.length > 0 ? keyframes : [emptyImageEntry(), emptyImageEntry()],
    shape: shapeFromRequest(initialRequest),
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto KeyframeInterpolationRequest (services/ltx/schemas.py) --
// quality recipe only, so negative_prompt is always honored (unlike on
// the text-to-video form, where it's ignored in fast mode). See
// TextToVideoForm.jsx for how initialRequest/useFormState/FormShell fit
// together; the pattern is identical here.
export default function KeyframeInterpolationForm({
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
  const minKeyframes = useArrayMinItems(MODEL, 'keyframes') || 2;
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { prompt, negativePrompt, seed, keyframes, shape, partition } = values;

  const payload = {
    prompt,
    seed: seed === '' ? undefined : Number(seed),
    keyframes: toImageConditioningPayload(keyframes),
    ...toShapePayload(shape),
  };
  if (negativePrompt.trim()) payload.negative_prompt = negativePrompt;
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [
    ...missingRequired(schemas, MODEL, payload),
    ...imageListBlockers(keyframes, 'Keyframe', minKeyframes),
    ...shapeBlockers(shape),
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/ltx/videos/keyframe-interpolation', payload));
  }

  return (
    <FormShell
      title="Keyframe interpolation"
      description={`At least ${minKeyframes} keyframes, quality recipe only (always honors negative_prompt).`}
      endpoint="POST /v1/ltx/videos/keyframe-interpolation"
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
        <FieldLabel
          model={MODEL}
          field="prompt"
          text="prompt (describes the motion connecting the keyframes, not their static appearance)"
        >
          <Textarea value={prompt} onChange={(event) => setField('prompt', event.target.value)} rows={3} />
        </FieldLabel>
        <ImageConditioningList
          label={`Keyframes (at least ${minKeyframes}, first at frame_idx=0)`}
          entries={keyframes}
          onChange={(next) => setField('keyframes', next)}
          minEntries={minKeyframes}
        />
      </Section>

      <Section title="Settings">
        <VideoShapeFields model={MODEL} shape={shape} onChange={(next) => setField('shape', next)} />
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <FieldLabel model={MODEL} field="negative_prompt" text="negative_prompt">
          <Textarea value={negativePrompt} onChange={(event) => setField('negativePrompt', event.target.value)} rows={2} />
        </FieldLabel>
        <SeedField model={MODEL} value={seed} onChange={(value) => setField('seed', value)} />
        <PartitionField model={MODEL} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}
