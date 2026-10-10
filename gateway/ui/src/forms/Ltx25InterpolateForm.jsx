import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { assetBlocker, defaultShape, reusedAsset, shapeBlockers, shapeFromRequest, toShapePayload, useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Textarea from '../components/ui/Textarea.jsx';
import SegmentedControl from '../components/ui/SegmentedControl.jsx';
import Switch from '../components/ui/Switch.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import FieldHeading from '../components/FieldHeading.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';
import VideoShapeFields from '../components/VideoShapeFields.jsx';

const MODEL = 'Ltx25InterpolateRequest';
const FORM_ID = 'ltx25-interpolate';

const DEFAULTS = {
  firstFrame: null,
  lastFrame: null,
  prompt: '',
  mode: 'fast',
  seed: 10,
  enhancePrompt: false,
  shape: defaultShape(),
  partition: '',
};

// Only these carry over as this form's defaults the next time it's opened
// fresh -- see storage.js. The two frames and the prompt are specific to one
// generation, so they're left out.
const REMEMBER_KEYS = ['mode', 'seed', 'enhancePrompt', 'shape', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    firstFrame: reusedAsset(initialRequest.first_frame_asset_id),
    lastFrame: reusedAsset(initialRequest.last_frame_asset_id),
    prompt: initialRequest.prompt ?? '',
    mode: initialRequest.mode ?? 'fast',
    seed: initialRequest.seed ?? 10,
    enhancePrompt: initialRequest.enhance_prompt ?? false,
    shape: shapeFromRequest(initialRequest),
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto Ltx25InterpolateRequest (services/ltx25/schemas.py): a
// first-frame image, a last-frame image, and a prompt describing the motion
// between them. Same Fast / Quality (DFR) modes as Ltx25TextToVideoForm.jsx,
// but no auto-length choice (the last frame's position needs a known clip
// length) and no negative_prompt. See TextToVideoForm.jsx for how
// initialRequest/useFormState/FormShell fit together.
export default function Ltx25InterpolateForm({
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

  const { firstFrame, lastFrame, prompt, mode, seed, enhancePrompt, shape, partition } = values;

  const payload = {
    first_frame_asset_id: firstFrame?.status === 'ready' ? firstFrame.assetId : undefined,
    last_frame_asset_id: lastFrame?.status === 'ready' ? lastFrame.assetId : undefined,
    prompt,
    mode,
    seed: seed === '' ? undefined : Number(seed),
    enhance_prompt: enhancePrompt,
    ...toShapePayload(shape),
  };
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [
    // The two asset fields are reported by assetBlocker below instead (it
    // knows "still uploading" from "not chosen yet"), so skip the generic
    // "is required" message for them -- same approach as RetakeForm.jsx.
    ...missingRequired(schemas, MODEL, payload, ['first_frame_asset_id', 'last_frame_asset_id']),
    ...assetBlocker(firstFrame, 'the first frame'),
    ...assetBlocker(lastFrame, 'the last frame'),
    ...shapeBlockers(shape),
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/ltx25/videos/interpolate', payload));
  }

  return (
    <FormShell
      title="First / last frame interpolation (LTX-2.5)"
      description="Upload the frame the video should start on and the frame it should end on; the prompt describes the motion between them. Works best with two frames of the same scene (completely different shots tend to cross-dissolve). Give both the output's aspect ratio; they're center-cropped to fit."
      endpoint="POST /v1/ltx25/videos/interpolate"
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
      <Section title="Frames & prompt">
        <div className="grid gap-4 sm:grid-cols-2">
          <AssetUpload
            model={MODEL}
            field="first_frame_asset_id"
            label="first frame"
            accept="image/*"
            value={firstFrame}
            onChange={(next) => setField('firstFrame', next)}
            required
          />
          <AssetUpload
            model={MODEL}
            field="last_frame_asset_id"
            label="last frame"
            accept="image/*"
            value={lastFrame}
            onChange={(next) => setField('lastFrame', next)}
            required
          />
        </div>
        <FieldLabel model={MODEL} field="prompt" text="prompt (describes the motion connecting the frames, not their appearance)">
          <Textarea value={prompt} onChange={(event) => setField('prompt', event.target.value)} rows={4} />
        </FieldLabel>
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
              { value: 'quality', label: 'Quality (DFR, slower)' },
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
        <SeedField model={MODEL} value={seed} onChange={(value) => setField('seed', value)} />
        <PartitionField model={MODEL} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}
