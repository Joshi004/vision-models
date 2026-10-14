import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { assetBlocker, imageListBlockers, reusedAsset, useSubmitJob } from '../utils.js';
import { REFERENCE_PROMPT_TEMPLATE, defaultH3Shape, h3ShapeBlockers, h3ShapeFromRequest, toH3ShapePayload } from '../h3.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Textarea from '../components/ui/Textarea.jsx';
import Input from '../components/ui/Input.jsx';
import Button from '../components/ui/Button.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import ReferenceImageList from '../components/ReferenceImageList.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';
import H3ShapeFields from '../components/H3ShapeFields.jsx';

const MODEL = 'H3AudioToVideoRequest';
const FORM_ID = 'h3-audio-to-video';
const MAX_IMAGES = 9;
const MAX_VIDEOS = 3;

const DEFAULTS = {
  prompt: '',
  audio: null,
  // One empty row to show where the first reference goes; an empty row blocks
  // submitting until a file is added or the row is removed.
  images: [{ asset: null }],
  videos: [],
  steps: '',
  seed: 10,
  shape: defaultH3Shape(),
  partition: '',
};

// Only these carry over as this form's defaults the next time it's opened
// fresh -- see storage.js. The prompt and every uploaded file are specific to
// one generation, so they're left out.
const REMEMBER_KEYS = ['steps', 'seed', 'shape', 'partition'];

const toEntries = (assetIds) => (assetIds ?? []).map((assetId) => ({ asset: reusedAsset(assetId) }));
const readyIds = (entries) => entries.filter((entry) => entry.asset?.status === 'ready').map((entry) => entry.asset.assetId);

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    prompt: initialRequest.prompt ?? '',
    audio: reusedAsset(initialRequest.audio_asset_id),
    images: toEntries(initialRequest.image_asset_ids),
    videos: toEntries(initialRequest.video_asset_ids),
    steps: initialRequest.num_inference_steps ?? '',
    seed: initialRequest.seed ?? 10,
    shape: h3ShapeFromRequest(initialRequest),
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto H3AudioToVideoRequest (services/h3/schemas.py): an audio
// reference plus image and/or video references. The audio is a REFERENCE the
// model reads and then generates its own soundtrack from -- it is not muxed into
// the result unchanged, unlike the LTX audio-to-video forms. See
// TextToVideoForm.jsx for how initialRequest/useFormState/FormShell fit together.
export default function H3AudioToVideoForm({ defaultPartition, onSubmitted, initialRequest = null, prefillJob = null, onClearPrefill }) {
  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(FORM_ID, {
    defaults: DEFAULTS,
    fromRequest: fromRequest(initialRequest),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${FORM_ID}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { prompt, audio, images, videos, steps, seed, shape, partition } = values;

  const imageIds = readyIds(images);
  const videoIds = readyIds(videos);
  const payload = {
    prompt,
    audio_asset_id: audio?.status === 'ready' ? audio.assetId : undefined,
    image_asset_ids: imageIds,
    video_asset_ids: videoIds,
    match_audio_duration: shape.lengthMode === 'match' ? true : undefined,
    seed: seed === '' ? undefined : Number(seed),
    num_inference_steps: steps === '' ? undefined : Number(steps),
    ...toH3ShapePayload(shape),
  };
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [
    // The audio file is reported by assetBlocker below instead (it knows
    // "still uploading" from "not chosen yet"), so skip the generic message.
    ...missingRequired(schemas, MODEL, payload, ['audio_asset_id']),
    ...assetBlocker(audio, 'the audio reference'),
    ...imageListBlockers(images, 'Reference image'),
    ...imageListBlockers(videos, 'Reference video'),
    ...(imageIds.length + videoIds.length === 0
      ? ['Add at least one reference image or video: an audio reference cannot be used on its own.']
      : []),
    ...h3ShapeBlockers(shape),
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/h3/videos/audio-to-video', payload));
  }

  return (
    <FormShell
      title="Audio to video (MiniMax H3)"
      description="Video with generated stereo sound from an audio reference plus image and/or video references. The audio is a reference the model reads and then makes its own soundtrack from; it is NOT muxed into the result unchanged (for your exact recording, use LTX audio to video with lip sync). Every job holds two GPUs and starts cold. The prompt reaches the model exactly as typed and is expected in H3's six-part reference format -- see the MiniMax H3 section of /v1/guide."
      endpoint="POST /v1/h3/videos/audio-to-video"
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
      <Section
        title="References"
        description="In the prompt they are <Picture 1>..., <Video 1>... and <Audio 1>, in the order shown, each kind counted on its own."
      >
        <AssetUpload
          model={MODEL}
          field="audio_asset_id"
          label="audio reference (2-15 s; <Audio 1>)"
          accept="audio/*,video/*"
          value={audio}
          onChange={(next) => setField('audio', next)}
          required
        />
        <ReferenceImageList
          model={MODEL}
          field="image_asset_ids"
          images={images}
          onChange={(next) => setField('images', next)}
          minImages={0}
          maxImages={MAX_IMAGES}
        />
        <ReferenceImageList
          model={MODEL}
          field="video_asset_ids"
          images={videos}
          onChange={(next) => setField('videos', next)}
          minImages={0}
          maxImages={MAX_VIDEOS}
          itemLabel="Reference video"
          accept="video/*"
          addLabel="Add reference video"
        />
      </Section>

      <Section title="Prompt">
        <FieldLabel model={MODEL} field="prompt" text="prompt">
          <Textarea
            value={prompt}
            onChange={(event) => setField('prompt', event.target.value)}
            rows={12}
            className="font-mono text-xs"
          />
        </FieldLabel>
        {prompt.trim() === '' && (
          <Button type="button" variant="secondary" size="sm" onClick={() => setField('prompt', REFERENCE_PROMPT_TEMPLATE)} className="self-start">
            Insert prompt template
          </Button>
        )}
      </Section>

      <Section title="Settings">
        <H3ShapeFields
          model={MODEL}
          shape={shape}
          onChange={(next) => setField('shape', next)}
          defaultCanvasHint="H3's own 16:9 canvas (1344x768). References do not set the shape."
          allowMatchAudio
        />
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <SeedField model={MODEL} value={seed} onChange={(value) => setField('seed', value)} />
        <FieldLabel model={MODEL} field="num_inference_steps" text="num_inference_steps (blank = default, 50)">
          <Input
            type="number"
            min="4"
            max="100"
            step="1"
            value={steps}
            onChange={(event) => setField('steps', event.target.value)}
            placeholder="50"
            className="max-w-[10rem]"
          />
        </FieldLabel>
        <PartitionField model={MODEL} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}
