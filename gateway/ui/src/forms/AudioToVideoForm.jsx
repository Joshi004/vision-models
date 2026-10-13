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
import Switch from '../components/ui/Switch.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import SeedField from '../components/SeedField.jsx';
import ImageConditioningList from '../components/ImageConditioningList.jsx';
import PartitionField from '../components/PartitionField.jsx';
import VideoShapeFields from '../components/VideoShapeFields.jsx';

// Defaults are LTX-2.3's. App.jsx reuses this form for LTX-2.5's
// audio-to-video by passing its own formId/model/endpoint/title plus
// `matchAudioLength` (see the 'ltx25-audio-to-video' recipe's formProps).
const DEFAULT_MODEL = 'AudioToVideoRequest';
const DEFAULT_FORM_ID = 'ltx-audio-to-video';
const DEFAULT_ENDPOINT = '/v1/ltx/videos/audio-to-video';

const DEFAULTS = {
  prompt: '',
  negativePrompt: '',
  seed: 10,
  audioAsset: null,
  audioStartTime: 0,
  audioMaxDuration: '',
  images: [],
  lipSync: false,
  shape: defaultShape(),
  partition: '',
};

const REMEMBER_KEYS = ['seed', 'lipSync', 'shape', 'partition'];

function fromRequest(initialRequest, matchAudioLength) {
  if (!initialRequest) return null;
  const shape = shapeFromRequest(initialRequest);
  // On LTX-2.5 a stored audio_max_duration means the clip followed the
  // audio, i.e. the 'auto' ("Match audio") length mode.
  if (matchAudioLength && initialRequest.audio_max_duration != null) shape.lengthMode = 'auto';
  return {
    prompt: initialRequest.prompt ?? '',
    negativePrompt: initialRequest.negative_prompt ?? '',
    seed: initialRequest.seed ?? 10,
    audioAsset: reusedAsset(initialRequest.audio_asset_id),
    audioStartTime: initialRequest.audio_start_time ?? 0,
    audioMaxDuration: initialRequest.audio_max_duration ?? '',
    images: imageEntriesFromRequest(initialRequest.images),
    lipSync: initialRequest.lip_sync ?? false,
    shape,
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto AudioToVideoRequest (services/ltx/schemas.py), or
// Ltx25AudioToVideoRequest (services/ltx25/schemas.py) when given LTX-2.5's
// formProps. See TextToVideoForm.jsx for how initialRequest/useFormState/
// FormShell fit together. `audioStartTime`/`audioMaxDuration` describe a
// window on whichever audio file is currently attached, so (like the audio
// file itself) they aren't remembered as this form's "last settings" -- see
// REMEMBER_KEYS -- only kept in the session draft.
//
// `matchAudioLength` (LTX-2.5): that backend won't take a fixed length and
// audio_max_duration together, so audio_max_duration only appears (and is
// required) under the "Match audio" length choice, where it sets the clip
// length; any other length choice sends no audio_max_duration.
export default function AudioToVideoForm({
  defaultPartition,
  onSubmitted,
  initialRequest = null,
  prefillJob = null,
  onClearPrefill,
  formId: FORM_ID = DEFAULT_FORM_ID,
  model: MODEL = DEFAULT_MODEL,
  endpoint = DEFAULT_ENDPOINT,
  title = 'Audio to video',
  matchAudioLength = false,
}) {
  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(FORM_ID, {
    defaults: DEFAULTS,
    fromRequest: fromRequest(initialRequest, matchAudioLength),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${FORM_ID}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { prompt, negativePrompt, seed, audioAsset, audioStartTime, audioMaxDuration, images, lipSync, shape, partition } = values;

  const payload = {
    prompt,
    seed: seed === '' ? undefined : Number(seed),
    audio_asset_id: audioAsset?.status === 'ready' ? audioAsset.assetId : undefined,
    audio_start_time: audioStartTime === '' ? undefined : Number(audioStartTime),
    images: toImageConditioningPayload(images),
    lip_sync: lipSync,
    ...toShapePayload(shape),
  };
  const lengthFromAudio = matchAudioLength && shape.lengthMode === 'auto';
  const showAudioMaxDuration = !matchAudioLength || lengthFromAudio;
  if (negativePrompt.trim()) payload.negative_prompt = negativePrompt;
  if (showAudioMaxDuration && audioMaxDuration !== '') payload.audio_max_duration = Number(audioMaxDuration);
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [
    ...missingRequired(schemas, MODEL, payload, ['audio_asset_id']),
    ...assetBlocker(audioAsset, 'the audio/video file'),
    ...imageListBlockers(images, 'Image'),
    ...shapeBlockers(shape),
    ...(lengthFromAudio && audioMaxDuration === ''
      ? ['audio_max_duration is required when "Match audio" is selected.']
      : []),
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON(endpoint, payload));
  }

  return (
    <FormShell
      title={title}
      description="The source audio is muxed into the output unchanged, not regenerated. Tick Lip sync to also have the mouth redrawn to match it exactly."
      endpoint={`POST ${endpoint}`}
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
          {showAudioMaxDuration && (
            <FieldLabel
              model={MODEL}
              field="audio_max_duration"
              text={
                matchAudioLength
                  ? 'audio_max_duration (seconds; the clip matches the audio read)'
                  : "audio_max_duration (defaults to the video's duration)"
              }
              required={lengthFromAudio}
            >
              <Input
                type="number"
                min="0"
                max={matchAudioLength ? '20' : undefined}
                step="0.1"
                value={audioMaxDuration}
                onChange={(event) => setField('audioMaxDuration', event.target.value)}
              />
            </FieldLabel>
          )}
        </div>
        <VideoShapeFields
          model={MODEL}
          shape={shape}
          onChange={(next) => setField('shape', next)}
          allowAutoLength={matchAudioLength}
          autoLengthLabel="Match audio"
        />
        <div className="flex flex-col gap-1">
          <Switch
            checked={lipSync}
            onChange={(checked) => setField('lipSync', checked)}
            label="Lip sync (re-draw the mouth to match the audio exactly, TBDub)"
          />
          {lipSync && (
            <p className="text-xs text-muted">
              Runs in the same job, right after the video is generated. Forces frame_rate to 25 (an exact frame count
              is converted to the same duration) and adds several minutes. Only the mouth is redrawn, so describe any
              other reaction (eyebrows, a laugh) in the prompt. If lip sync fails you still get the video without it,
              flagged on the job, and can retry just the sync.
            </p>
          )}
        </div>
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
