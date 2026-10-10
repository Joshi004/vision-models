import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { assetBlocker, reusedAsset, useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Textarea from '../components/ui/Textarea.jsx';
import Input from '../components/ui/Input.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';

// Defaults are LTX-2.3's. LTX-2.5's retake has identical fields and rules, so
// App.jsx reuses this same form for it by passing its own formId/model/
// endpoint (see the 'ltx25-retake' recipe's formProps there) instead of
// duplicating the whole file.
const DEFAULT_MODEL = 'RetakeRequest';
const DEFAULT_FORM_ID = 'ltx-retake';
const DEFAULT_ENDPOINT = '/v1/ltx/videos/retake';

const DEFAULTS = {
  video: null,
  prompt: '',
  startTime: 0,
  endTime: 1,
  seed: 10,
  partition: '',
};

const REMEMBER_KEYS = ['seed', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    video: reusedAsset(initialRequest.video_asset_id),
    prompt: initialRequest.prompt ?? '',
    startTime: initialRequest.start_time ?? 0,
    endTime: initialRequest.end_time ?? 1,
    seed: initialRequest.seed ?? 10,
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto RetakeRequest (services/ltx/schemas.py), or Ltx25RetakeRequest
// (services/ltx25/schemas.py) when given LTX-2.5's formProps -- no shape
// fields here, since height/width/frame-count are inherited from the source
// video. See TextToVideoForm.jsx for how initialRequest/useFormState/
// FormShell fit together.
export default function RetakeForm({
  defaultPartition,
  onSubmitted,
  initialRequest = null,
  prefillJob = null,
  onClearPrefill,
  formId: FORM_ID = DEFAULT_FORM_ID,
  model: MODEL = DEFAULT_MODEL,
  endpoint = DEFAULT_ENDPOINT,
  title = 'Retake',
}) {
  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(FORM_ID, {
    defaults: DEFAULTS,
    fromRequest: fromRequest(initialRequest),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${FORM_ID}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { video, prompt, startTime, endTime, seed, partition } = values;

  const payload = {
    video_asset_id: video?.status === 'ready' ? video.assetId : undefined,
    prompt,
    start_time: startTime === '' ? undefined : Number(startTime),
    end_time: endTime === '' ? undefined : Number(endTime),
    seed: seed === '' ? undefined : Number(seed),
  };
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [
    ...missingRequired(schemas, MODEL, payload, ['video_asset_id']),
    ...assetBlocker(video, 'the source video'),
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON(endpoint, payload));
  }

  return (
    <FormShell
      title={title}
      description="Regenerates only the [start_time, end_time] window. Source video must already have 8k+1 frames and height/width that are multiples of 32 (not checked until the job runs)."
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
      submitLabel="Regenerate window"
      onSubmit={handleSubmit}
    >
      <Section title="Source & prompt">
        <AssetUpload
          model={MODEL}
          field="video_asset_id"
          label="source video"
          accept="video/*"
          value={video}
          onChange={(next) => setField('video', next)}
          required
        />
        <FieldLabel model={MODEL} field="prompt" text="prompt (describes only the content of the regenerated window)">
          <Textarea value={prompt} onChange={(event) => setField('prompt', event.target.value)} rows={3} />
        </FieldLabel>
      </Section>

      <Section title="Settings">
        <div className="grid grid-cols-2 gap-3">
          <FieldLabel model={MODEL} field="start_time" text="start_time (seconds)">
            <Input type="number" min="0" step="0.1" value={startTime} onChange={(event) => setField('startTime', event.target.value)} />
          </FieldLabel>
          <FieldLabel model={MODEL} field="end_time" text="end_time (seconds)">
            <Input type="number" min="0" step="0.1" value={endTime} onChange={(event) => setField('endTime', event.target.value)} />
          </FieldLabel>
        </div>
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <SeedField model={MODEL} value={seed} onChange={(value) => setField('seed', value)} />
        <PartitionField model={MODEL} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}
