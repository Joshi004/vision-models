import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { assetBlocker, reusedAsset, useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import RvcSettingsFields, { EXPORT_FORMATS } from '../components/RvcSettingsFields.jsx';

const MODEL = 'ConvertRequest';
const FORM_ID = 'rvc-convert';

const DEFAULTS = {
  sourceAudio: null,
  voice: '',
  pitch: 0,
  f0Method: 'rmvpe',
  indexRate: 0.75,
  protect: 0.33,
  exportFormat: 'WAV',
  partition: '',
};

// Everything except sourceAudio carries over as this form's defaults the
// next time it's opened fresh -- see storage.js. `voice` is included: if
// you're converting several clips to the same target voice in a row,
// re-picking it every time would be annoying.
const REMEMBER_KEYS = ['voice', 'pitch', 'f0Method', 'indexRate', 'protect', 'exportFormat', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    sourceAudio: reusedAsset(initialRequest.source_audio_asset_id),
    voice: initialRequest.voice ?? '',
    pitch: initialRequest.pitch ?? 0,
    f0Method: initialRequest.f0_method ?? 'rmvpe',
    indexRate: initialRequest.index_rate ?? 0.75,
    protect: initialRequest.protect ?? 0.33,
    exportFormat: initialRequest.export_format ?? 'WAV',
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto ConvertRequest (services/rvc/schemas.py). Unlike every
// other form here, `voice` isn't a file upload -- it's one name out of a
// small, curated, server-side list (GET /v1/rvc/voices) -- see
// components/RvcSettingsFields.jsx, which owns that field (and every
// other conversion setting) as a single copy shared with
// RvcBatchForm.jsx, since both request schemas inherit them from the same
// base. No seed field: RVC conversion is deterministic given the same
// inputs, unlike LTX/Wan-Animate's diffusion sampling. See
// TextToVideoForm.jsx for how initialRequest/useFormState/FormShell fit
// together.
export default function RvcConvertForm({
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

  const { sourceAudio, voice, pitch, f0Method, indexRate, protect, exportFormat, partition } = values;

  const payload = {
    source_audio_asset_id: sourceAudio?.status === 'ready' ? sourceAudio.assetId : undefined,
    voice: voice || undefined,
    pitch: pitch === '' ? undefined : Number(pitch),
    f0_method: f0Method,
    index_rate: indexRate === '' ? undefined : Number(indexRate),
    protect: protect === '' ? undefined : Number(protect),
    export_format: exportFormat,
  };
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [
    ...missingRequired(schemas, MODEL, payload, ['source_audio_asset_id']),
    ...assetBlocker(sourceAudio, 'the recording to convert'),
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/rvc/convert', payload));
  }

  return (
    <FormShell
      title="RVC: convert voice"
      description="Converts an existing recording to one of the installed voices, keeping its own timing, pauses, and delivery. Not text-to-speech \u2014 there's no text field."
      endpoint="POST /v1/rvc/convert"
      prefillJob={prefillJob}
      onClearPrefill={onClearPrefill}
      restoredDraft={restoredDraft}
      onReset={reset}
      rememberSettings={rememberSettings}
      blockers={blockers}
      submitting={submitting}
      error={error}
      lastJobId={lastJobId}
      submitLabel="Convert voice"
      onSubmit={handleSubmit}
    >
      <Section title="Source">
        <AssetUpload
          model={MODEL}
          field="source_audio_asset_id"
          label="recording to convert"
          accept="audio/*"
          value={sourceAudio}
          onChange={(next) => setField('sourceAudio', next)}
          required
        />
      </Section>

      <RvcSettingsFields
        model={MODEL}
        values={values}
        setField={setField}
        advancedOpen={advancedOpen}
        onAdvancedToggle={setAdvancedOpen}
        defaultPartition={defaultPartition}
        exportFormats={EXPORT_FORMATS}
      />
    </FormShell>
  );
}
