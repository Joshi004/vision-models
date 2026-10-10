import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { assetBlocker, reusedAsset, useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import PartitionField from '../components/PartitionField.jsx';

const MODEL = 'TranscribeRequest';
const FORM_ID = 'parakeet-transcribe';

const DEFAULTS = {
  audio: null,
  partition: '',
};

const REMEMBER_KEYS = ['partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    audio: reusedAsset(initialRequest.audio_asset_id),
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto TranscribeRequest (services/parakeet/schemas.py) -- the
// simplest form here: one file, one optional partition override, nothing
// else. No `language` field (the model auto-detects it) and no `seed`
// field (transcription is deterministic) -- see that schema's own
// docstring. See TextToVideoForm.jsx for how
// initialRequest/useFormState/FormShell fit together.
export default function TranscribeForm({
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

  const { audio, partition } = values;

  const payload = {
    audio_asset_id: audio?.status === 'ready' ? audio.assetId : undefined,
  };
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [
    ...missingRequired(schemas, MODEL, payload, ['audio_asset_id']),
    ...assetBlocker(audio, 'the recording to transcribe'),
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/parakeet/transcribe', payload));
  }

  return (
    <FormShell
      title="Parakeet: transcribe"
      description="Transcribes a recording (or a video's audio track) to text, with word- and segment-level timestamps. Long recordings are chunked automatically; no action needed on your part."
      endpoint="POST /v1/parakeet/transcribe"
      prefillJob={prefillJob}
      onClearPrefill={onClearPrefill}
      restoredDraft={restoredDraft}
      onReset={reset}
      rememberSettings={rememberSettings}
      blockers={blockers}
      submitting={submitting}
      error={error}
      lastJobId={lastJobId}
      submitLabel="Transcribe"
      onSubmit={handleSubmit}
    >
      <Section title="Recording">
        <AssetUpload
          model={MODEL}
          field="audio_asset_id"
          label="recording to transcribe (audio or video)"
          accept="audio/*,video/*"
          value={audio}
          onChange={(next) => setField('audio', next)}
          required
        />
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <PartitionField
          model={MODEL}
          value={partition}
          onChange={(value) => setField('partition', value)}
          defaultPartition={defaultPartition}
        />
      </CollapsibleSection>
    </FormShell>
  );
}
