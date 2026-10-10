import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import {
  assetBlocker,
  defaultShape,
  imageEntriesFromRequest,
  imageListBlockers,
  reusedAsset,
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
import Input from '../components/ui/Input.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import SeedField from '../components/SeedField.jsx';
import ImageConditioningList from '../components/ImageConditioningList.jsx';
import PartitionField from '../components/PartitionField.jsx';
import VideoShapeFields from '../components/VideoShapeFields.jsx';

const MODEL = 'AudioToVideoRequest';
const FORM_ID = 'ltx-audio-to-video';

const DEFAULTS = {
  prompt: '',
  negativePrompt: '',
  seed: 10,
  audioAsset: null,
  audioStartTime: 0,
  audioMaxDuration: '',
  images: [],
  shape: defaultShape(),
  partition: '',
};

const REMEMBER_KEYS = ['seed', 'shape', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    prompt: initialRequest.prompt ?? '',
    negativePrompt: initialRequest.negative_prompt ?? '',
    seed: initialRequest.seed ?? 10,
    audioAsset: reusedAsset(initialRequest.audio_asset_id),
    audioStartTime: initialRequest.audio_start_time ?? 0,
    audioMaxDuration: initialRequest.audio_max_duration ?? '',
    images: imageEntriesFromRequest(initialRequest.images),
    shape: shapeFromRequest(initialRequest),
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto AudioToVideoRequest (services/ltx/schemas.py). See
// TextToVideoForm.jsx for how initialRequest/useFormState/FormShell fit
// together. `audioStartTime`/`audioMaxDuration` describe a window on
// whichever audio file is currently attached, so (like the audio file
// itself) they aren't remembered as this form's "last settings" -- see
// REMEMBER_KEYS -- only kept in the session draft.
export default function AudioToVideoForm({
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

  const { prompt, negativePrompt, seed, audioAsset, audioStartTime, audioMaxDuration, images, shape, partition } = values;

  const payload = {
    prompt,
    seed: seed === '' ? undefined : Number(seed),
    audio_asset_id: audioAsset?.status === 'ready' ? audioAsset.assetId : undefined,
    audio_start_time: audioStartTime === '' ? undefined : Number(audioStartTime),
    images: toImageConditioningPayload(images),
    ...toShapePayload(shape),
  };
  if (negativePrompt.trim()) payload.negative_prompt = negativePrompt;
  if (audioMaxDuration !== '') payload.audio_max_duration = Number(audioMaxDuration);
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [
    ...missingRequired(schemas, MODEL, payload, ['audio_asset_id']),
    ...assetBlocker(audioAsset, 'the audio/video file'),
    ...imageListBlockers(images, 'Image'),
    ...shapeBlockers(shape),
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/ltx/videos/audio-to-video', payload));
  }

  return (
    <FormShell
      title="Audio to video"
      description="The source audio is muxed into the output unchanged, not regenerated."
      endpoint="POST /v1/ltx/videos/audio-to-video"
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
      <Section title="Prompt & input">
        <FieldLabel
          model={MODEL}
          field="prompt"
          text="prompt (describes the visible speaker/scene that should match the audio)"
        >
          <Textarea value={prompt} onChange={(event) => setField('prompt', event.target.value)} rows={3} />
        </FieldLabel>
        <AssetUpload
          model={MODEL}
          field="audio_asset_id"
          label="audio/video file to condition on"
          accept="audio/*,video/*"
          value={audioAsset}
          onChange={(next) => setField('audioAsset', next)}
          required
        />
        <ImageConditioningList
          label="Images (optional \u2014 pin the speaker's appearance)"
          entries={images}
          onChange={(next) => setField('images', next)}
        />
      </Section>

      <Section title="Settings">
        <div className="grid grid-cols-2 gap-3">
          <FieldLabel model={MODEL} field="audio_start_time" text="audio_start_time (seconds)">
            <Input
              type="number"
              min="0"
              step="0.1"
              value={audioStartTime}
              onChange={(event) => setField('audioStartTime', event.target.value)}
            />
          </FieldLabel>
          <FieldLabel model={MODEL} field="audio_max_duration" text="audio_max_duration (defaults to the video's duration)">
            <Input
              type="number"
              min="0"
              step="0.1"
              value={audioMaxDuration}
              onChange={(event) => setField('audioMaxDuration', event.target.value)}
            />
          </FieldLabel>
        </div>
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
