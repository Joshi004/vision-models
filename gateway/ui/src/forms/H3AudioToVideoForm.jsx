import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { audioListBlockers, imageListBlockers, reusedAsset, useSubmitJob } from '../utils.js';
import { REFERENCE_IMAGES_PROMPT_TEMPLATE, REFERENCE_PROMPT_TEMPLATE, defaultH3Shape, h3ShapeBlockers, h3ShapeFromRequest, toH3ShapePayload } from '../h3.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Textarea from '../components/ui/Textarea.jsx';
import Input from '../components/ui/Input.jsx';
import Button from '../components/ui/Button.jsx';
import Switch from '../components/ui/Switch.jsx';
import ReferenceImageList from '../components/ReferenceImageList.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';
import H3ShapeFields from '../components/H3ShapeFields.jsx';

const MODEL = 'H3AudioToVideoRequest';
const FORM_ID = 'h3-audio-to-video';
// MiniMax H3's reference limits (services/h3/config.py): per kind, and in all.
const MAX_IMAGES = 9;
const MAX_VIDEOS = 3;
const MAX_AUDIOS = 3;
const MAX_REFERENCES = 12;

const DEFAULTS = {
  prompt: '',
  // One empty row to show where the first reference goes; an empty row blocks
  // submitting until a file is added or the row is removed.
  images: [{ asset: null }],
  // Each video entry is { asset, useAudio }: useAudio is the "use this video's
  // soundtrack" switch, on unless it is exactly false (so a row without the
  // field, e.g. an older stored draft, counts as on).
  videos: [],
  // Audio references start empty: they are optional.
  audios: [],
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
const isReady = (entry) => entry.asset?.status === 'ready';
const readyIds = (entries) => entries.filter(isReady).map((entry) => entry.asset.assetId);
const usesSoundtrack = (video) => video.useAudio !== false;

// `video_audio` holds one flag per video; absent means every soundtrack is used.
const toVideoEntries = (assetIds, videoAudio) =>
  (assetIds ?? []).map((assetId, index) => ({ asset: reusedAsset(assetId), useAudio: videoAudio?.[index] !== false }));

// What the model will call each reference, in order, as a short line for the form.
// Pictures and videos count their own kind; <Audio N> counts all audio, and a video
// with its soundtrack on takes the next one just before its <Video N>, then the audio
// files follow. A video without sound takes none, which the form cannot know, so it
// is assumed here that a video with its soundtrack on has one.
function referenceLabels(images, videos, audios) {
  const labels = images.map((_, index) => `<Picture ${index + 1}>`);
  let audioCount = 0;
  videos.forEach((video, index) => {
    audioCount += usesSoundtrack(video) ? 1 : 0;
    labels.push(usesSoundtrack(video) ? `<Audio ${audioCount}> + <Video ${index + 1}>` : `<Video ${index + 1}>`);
  });
  audios.forEach(() => {
    audioCount += 1;
    labels.push(`<Audio ${audioCount}>`);
  });
  return labels;
}

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  // Jobs submitted before 0.14.0 stored a single `audio_asset_id`.
  const audioIds = initialRequest.audio_asset_ids ?? (initialRequest.audio_asset_id ? [initialRequest.audio_asset_id] : []);
  return {
    prompt: initialRequest.prompt ?? '',
    images: toEntries(initialRequest.image_asset_ids),
    videos: toVideoEntries(initialRequest.video_asset_ids, initialRequest.video_audio),
    audios: toEntries(audioIds),
    steps: initialRequest.num_inference_steps ?? '',
    seed: initialRequest.seed ?? 10,
    shape: h3ShapeFromRequest(initialRequest),
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto H3AudioToVideoRequest (services/h3/schemas.py): image and/or
// video references (up to 9 and 3), plus optional audio references (up to 3), 12
// in all. Audio is a REFERENCE the model reads and then generates its own
// soundtrack from -- it is not muxed into the result unchanged, unlike the LTX
// audio-to-video forms. See TextToVideoForm.jsx for how
// initialRequest/useFormState/FormShell fit together.
export default function H3AudioToVideoForm({ defaultPartition, onSubmitted, initialRequest = null, prefillJob = null, onClearPrefill }) {
  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(FORM_ID, {
    defaults: DEFAULTS,
    fromRequest: fromRequest(initialRequest),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${FORM_ID}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { prompt, images, videos, audios, steps, seed, shape, partition } = values;

  const imageIds = readyIds(images);
  const audioIds = readyIds(audios);
  // video_audio lines up with video_asset_ids, so both come from the same ready videos.
  const readyVideos = videos.filter(isReady);
  const videoIds = readyVideos.map((video) => video.asset.assetId);
  const videoAudio = readyVideos.map(usesSoundtrack);
  const payload = {
    prompt,
    image_asset_ids: imageIds,
    video_asset_ids: videoIds,
    // Only sent when a soundtrack is switched off: left out, every video's soundtrack is used.
    video_audio: videoAudio.includes(false) ? videoAudio : undefined,
    audio_asset_ids: audioIds,
    match_audio_duration: shape.lengthMode === 'match' ? true : undefined,
    seed: seed === '' ? undefined : Number(seed),
    num_inference_steps: steps === '' ? undefined : Number(steps),
    ...toH3ShapePayload(shape),
  };
  if (partition.trim()) payload.partition = partition.trim();

  // Rows count whether or not they have a file yet, so the 12-reference limit is held while adding.
  const rowTotal = images.length + videos.length + audios.length;
  // The most rows a list can hold: its own limit, or what is left of the 12 after the other two lists.
  const roomFor = (list, kindMax) => Math.min(kindMax, MAX_REFERENCES - (rowTotal - list.length));

  const blockers = [
    ...missingRequired(schemas, MODEL, payload),
    ...imageListBlockers(images, 'Reference image'),
    ...imageListBlockers(videos, 'Reference video'),
    // Audio is optional: only a row that was added but has no file, or is still uploading, blocks the form.
    ...audioListBlockers(audios, 'Reference audio'),
    ...(imageIds.length + videoIds.length === 0
      ? ['Add at least one reference image or video: audio references cannot be used on their own.']
      : []),
    ...(rowTotal > MAX_REFERENCES
      ? [`At most ${MAX_REFERENCES} references in all (images + videos + audio): ${rowTotal} added.`]
      : []),
    // A row that is empty or uploading is reported by audioListBlockers above, so only "no rows at all" is added here.
    ...(shape.lengthMode === 'match' && audios.length === 0
      ? ['"Match audio" needs an audio reference: add one or choose another length.']
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
      description="Video with generated stereo sound from up to 9 reference images, 3 videos and 3 audio clips (12 in all; at least one image or video, audio is optional). Images are references for how people, places and styles look, not first or last frames: the prompt says who appears and when. Audio is a reference the model reads and then makes its own soundtrack from; it is NOT muxed into the result unchanged (for your exact recording, use LTX audio to video with lip sync). A reference video that has sound also gives its soundtrack to the model unless you switch that off. Every job holds two GPUs and starts cold. The prompt reaches the model exactly as typed and is expected in H3's six-part reference format -- see the MiniMax H3 section of /v1/guide."
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
        description={`Up to ${MAX_IMAGES} images, ${MAX_VIDEOS} videos (2-15 s each) and ${MAX_AUDIOS} audio clips (2-15 s each), ${MAX_REFERENCES} in all. At least one image or video is needed; audio is optional. In the prompt they are <Picture 1>..., <Video 1>... and <Audio 1>..., in the order shown (see the labels below).`}
      >
        <ReferenceImageList
          model={MODEL}
          field="image_asset_ids"
          images={images}
          onChange={(next) => setField('images', next)}
          minImages={0}
          maxImages={roomFor(images, MAX_IMAGES)}
        />
        <ReferenceImageList
          model={MODEL}
          field="video_asset_ids"
          images={videos}
          onChange={(next) => setField('videos', next)}
          minImages={0}
          maxImages={roomFor(videos, MAX_VIDEOS)}
          itemLabel="Reference video"
          accept="video/*"
          addLabel="Add reference video"
          renderExtra={(entry, index) => (
            <div className="mt-2">
              <Switch
                checked={usesSoundtrack(entry)}
                onChange={(checked) =>
                  setField('videos', (previous) => previous.map((item, i) => (i === index ? { ...item, useAudio: checked } : item)))
                }
                label="Use this video's soundtrack as an audio reference"
              />
            </div>
          )}
        />
        <ReferenceImageList
          model={MODEL}
          field="audio_asset_ids"
          images={audios}
          onChange={(next) => setField('audios', next)}
          minImages={0}
          maxImages={roomFor(audios, MAX_AUDIOS)}
          itemLabel="Reference audio"
          accept="audio/*,video/*"
          addLabel="Add reference audio"
        />
        {rowTotal > 0 && (
          <div className="flex flex-col gap-1 text-xs text-muted">
            <p>
              Labels in the prompt, in order: <span className="font-mono text-text">{referenceLabels(images, videos, audios).join(', ')}</span>
            </p>
            {videos.some(usesSoundtrack) && (
              <p>
                A video with its soundtrack on takes the next &lt;Audio N&gt; just before its &lt;Video N&gt;, and the audio files
                follow. A video that has no sound takes none, so every &lt;Audio N&gt; after it is one lower than shown.
              </p>
            )}
          </div>
        )}
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
          <Button type="button" variant="secondary" size="sm" onClick={() => setField('prompt', audios.length > 0 ? REFERENCE_PROMPT_TEMPLATE : REFERENCE_IMAGES_PROMPT_TEMPLATE)} className="self-start">
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
