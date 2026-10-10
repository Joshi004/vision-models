import { postJSON } from '../api.js';
import { missingRequired, useArrayMinItems, useSchemas } from '../schema.js';
import { audioEntriesFromRequest, audioListBlockers, emptyAudioEntry, toBatchSourcesPayload, useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import AudioFileList from '../components/AudioFileList.jsx';
import RvcSettingsFields, { EXPORT_FORMATS } from '../components/RvcSettingsFields.jsx';

const MODEL = 'BatchConvertRequest';
const FORM_ID = 'rvc-batch';

const DEFAULTS = {
  sources: [emptyAudioEntry()],
  voice: '',
  pitch: 0,
  f0Method: 'rmvpe',
  indexRate: 0.75,
  protect: 0.33,
  exportFormat: 'WAV',
  partition: '',
};

// Same reasoning as RvcConvertForm.jsx's own REMEMBER_KEYS -- everything
// except the recordings themselves carries over as this form's defaults
// the next time it's opened fresh.
const REMEMBER_KEYS = ['voice', 'pitch', 'f0Method', 'indexRate', 'protect', 'exportFormat', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  const sources = audioEntriesFromRequest(initialRequest.sources);
  return {
    sources: sources.length > 0 ? sources : [emptyAudioEntry()],
    voice: initialRequest.voice ?? '',
    pitch: initialRequest.pitch ?? 0,
    f0Method: initialRequest.f0_method ?? 'rmvpe',
    indexRate: initialRequest.index_rate ?? 0.75,
    protect: initialRequest.protect ?? 0.33,
    exportFormat: initialRequest.export_format ?? 'WAV',
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto BatchConvertRequest (services/rvc/schemas.py) -- the same
// conversion as POST /v1/rvc/convert (see RvcConvertForm.jsx), just for
// several recordings converted to the same voice with the same settings
// in one Slurm job, instead of one job per recording. voice/pitch/
// f0_method/index_rate/protect/export_format/partition are all handled by
// components/RvcSettingsFields.jsx, shared with RvcConvertForm.jsx since
// both request schemas inherit them from the same base
// (ConversionSettings) -- this form only owns `sources` itself.
//
// Result is a single output.zip, one converted file per source, in the
// same order -- see components/ResultPreview.jsx for how that's
// downloaded (no inline audio player, unlike a single conversion's
// result). All-or-nothing: if any one file fails, the whole job fails.
export default function RvcBatchForm({
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
  const minSources = useArrayMinItems(MODEL, 'sources') || 1;
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { sources, voice, pitch, f0Method, indexRate, protect, exportFormat, partition } = values;

  const payload = {
    sources: toBatchSourcesPayload(sources),
    voice: voice || undefined,
    pitch: pitch === '' ? undefined : Number(pitch),
    f0_method: f0Method,
    index_rate: indexRate === '' ? undefined : Number(indexRate),
    protect: protect === '' ? undefined : Number(protect),
    export_format: exportFormat,
  };
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [
    ...missingRequired(schemas, MODEL, payload),
    ...audioListBlockers(sources, 'Recording', minSources),
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/rvc/batch-convert', payload));
  }

  return (
    <FormShell
      title="RVC: batch convert"
      description={`Converts several recordings to the same installed voice in one job (at least ${minSources} recording${minSources === 1 ? '' : 's'}). Result is a single zip \u2014 if any one file fails, the whole batch fails.`}
      endpoint="POST /v1/rvc/batch-convert"
      prefillJob={prefillJob}
      onClearPrefill={onClearPrefill}
      restoredDraft={restoredDraft}
      onReset={reset}
      rememberSettings={rememberSettings}
      blockers={blockers}
      submitting={submitting}
      error={error}
      lastJobId={lastJobId}
      submitLabel="Convert batch"
      onSubmit={handleSubmit}
    >
      <Section title="Recordings">
        <AudioFileList
          label={`Recordings (at least ${minSources})`}
          entries={sources}
          onChange={(next) => setField('sources', next)}
          minEntries={minSources}
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
