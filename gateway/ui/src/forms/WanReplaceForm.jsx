import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { assetBlocker, reusedAsset, useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Switch from '../components/ui/Switch.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';

const MODEL = 'ReplaceRequest';
const FORM_ID = 'wan-replace';

const DEFAULTS = {
  video: null,
  image: null,
  seed: 10,
  useRelightingLora: true,
  keepAudio: true,
  partition: '',
};

const REMEMBER_KEYS = ['seed', 'useRelightingLora', 'keepAudio', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    video: reusedAsset(initialRequest.video_asset_id),
    image: reusedAsset(initialRequest.image_asset_id),
    seed: initialRequest.seed ?? 10,
    useRelightingLora: initialRequest.use_relighting_lora ?? true,
    keepAudio: initialRequest.keep_audio ?? true,
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto ReplaceRequest (services/wan_animate/schemas.py) -- no
// prompt field on this endpoint at all; the underlying pipeline never
// forwards one to the model. See TextToVideoForm.jsx for how
// initialRequest/useFormState/FormShell fit together.
export default function WanReplaceForm({
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

  const { video, image, seed, useRelightingLora, keepAudio, partition } = values;

  const payload = {
    video_asset_id: video?.status === 'ready' ? video.assetId : undefined,
    image_asset_id: image?.status === 'ready' ? image.assetId : undefined,
    seed: seed === '' ? undefined : Number(seed),
    use_relighting_lora: useRelightingLora,
    keep_audio: keepAudio,
  };
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [
    ...missingRequired(schemas, MODEL, payload, ['video_asset_id', 'image_asset_id']),
    ...assetBlocker(video, 'the source video'),
    ...assetBlocker(image, 'the reference photo'),
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/wan-animate/videos/replace', payload));
  }

  return (
    <FormShell
      title="Wan-Animate: replace character"
      description={
        'Single-person source video only, no prompt field on this endpoint. Roughly 25 minutes and ~75GB VRAM ' +
        'for a ~7s clip \u2014 consider partition "main" to avoid losing the whole run to pre-emption on the ' +
        'default "background" partition.'
      }
      endpoint="POST /v1/wan-animate/videos/replace"
      prefillJob={prefillJob}
      onClearPrefill={onClearPrefill}
      restoredDraft={restoredDraft}
      onReset={reset}
      rememberSettings={rememberSettings}
      blockers={blockers}
      submitting={submitting}
      error={error}
      lastJobId={lastJobId}
      submitLabel="Replace character"
      onSubmit={handleSubmit}
    >
      <Section title="Source & reference">
        <AssetUpload
          model={MODEL}
          field="video_asset_id"
          label="source video (single person, throughout)"
          accept="video/*"
          value={video}
          onChange={(next) => setField('video', next)}
          required
        />
        <AssetUpload
          model={MODEL}
          field="image_asset_id"
          label="reference character photo (front-facing, well-lit, unoccluded)"
          accept="image/*"
          value={image}
          onChange={(next) => setField('image', next)}
          required
        />
      </Section>

      <Section title="Settings">
        <Switch
          checked={useRelightingLora}
          onChange={(checked) => setField('useRelightingLora', checked)}
          label="use_relighting_lora (match the swapped-in character's lighting/tone to the scene)"
        />
        <Switch
          checked={keepAudio}
          onChange={(checked) => setField('keepAudio', checked)}
          label="keep_audio (re-attach the source video's original audio track)"
        />
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <SeedField model={MODEL} value={seed} onChange={(value) => setField('seed', value)} />
        <PartitionField model={MODEL} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}
