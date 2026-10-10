import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { assetBlocker, reusedAsset, useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Textarea from '../components/ui/Textarea.jsx';
import Input from '../components/ui/Input.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';

const MODEL = 'SynthesizeRequest';
const FORM_ID = 'breeze-tts-synthesize';

const DEFAULTS = {
  text: '',
  instruction: '',
  referenceAudio: null,
  referenceText: '',
  seed: 42,
  cfgScale: 1,
  partition: '',
};

// Everything except the text itself and the reference recording carries
// over as this form's defaults the next time it's opened fresh -- see
// storage.js.
const REMEMBER_KEYS = ['instruction', 'referenceText', 'seed', 'cfgScale', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    text: initialRequest.text ?? '',
    instruction: initialRequest.instruction ?? '',
    referenceAudio: reusedAsset(initialRequest.reference_audio_asset_id),
    referenceText: initialRequest.reference_text ?? '',
    seed: initialRequest.seed ?? 42,
    cfgScale: initialRequest.cfg_scale ?? 1,
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto SynthesizeRequest (services/breeze_tts/schemas.py). The mode
// is implied by which fields are filled in -- instruction only = Voice
// Design; reference recording + its transcript = Voice Clone; both =
// Voice Direction -- see the mode hint below. See TextToVideoForm.jsx for
// how initialRequest/useFormState/FormShell fit together.
export default function BreezeTtsForm({
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

  const { text, instruction, referenceAudio, referenceText, seed, cfgScale, partition } = values;

  const hasReference = Boolean(referenceAudio);
  const hasInstruction = instruction.trim() !== '';
  const mode = hasReference ? (hasInstruction ? 'Voice Direction' : 'Voice Clone') : 'Voice Design';

  const payload = {
    text,
    seed: seed === '' ? undefined : Number(seed),
    cfg_scale: cfgScale === '' ? undefined : Number(cfgScale),
  };
  if (hasInstruction) payload.instruction = instruction;
  if (referenceAudio?.status === 'ready') payload.reference_audio_asset_id = referenceAudio.assetId;
  if (referenceText.trim()) payload.reference_text = referenceText;
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [...missingRequired(schemas, MODEL, payload)];
  if (hasReference) {
    blockers.push(...assetBlocker(referenceAudio, 'the reference recording'));
    if (!referenceText.trim()) blockers.push('Enter the exact transcript of the reference recording.');
  } else if (referenceText.trim()) {
    blockers.push('Upload the reference recording, or clear its transcript.');
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/breeze-tts/synthesize', payload));
  }

  return (
    <FormShell
      title="Breeze TTS 2: text to speech"
      description={`Mode: ${mode}. Describe a voice with an instruction (Voice Design), add a reference recording to clone it, or both to clone and steer the delivery. English and Chinese. Weights are for research / non-commercial use only.`}
      endpoint="POST /v1/breeze-tts/synthesize"
      prefillJob={prefillJob}
      onClearPrefill={onClearPrefill}
      restoredDraft={restoredDraft}
      onReset={reset}
      rememberSettings={rememberSettings}
      blockers={blockers}
      submitting={submitting}
      error={error}
      lastJobId={lastJobId}
      submitLabel="Generate speech"
      onSubmit={handleSubmit}
    >
      <Section title="Text">
        <FieldLabel
          model={MODEL}
          field="text"
          text="text (vocal events: (laugh) (sigh) in English, [笑] [叹气] in Chinese)"
          required
        >
          <Textarea value={text} onChange={(event) => setField('text', event.target.value)} rows={4} />
        </FieldLabel>
      </Section>

      <Section title="Voice">
        <FieldLabel model={MODEL} field="instruction" text="instruction (voice description, or delivery direction with a reference)">
          <Textarea value={instruction} onChange={(event) => setField('instruction', event.target.value)} rows={2} />
        </FieldLabel>
        <AssetUpload
          model={MODEL}
          field="reference_audio_asset_id"
          label="reference recording (optional, to clone a voice)"
          accept="audio/*"
          value={referenceAudio}
          onChange={(next) => setField('referenceAudio', next)}
        />
        <FieldLabel model={MODEL} field="reference_text" text="reference_text (exact transcript of the reference recording)">
          <Textarea value={referenceText} onChange={(event) => setField('referenceText', event.target.value)} rows={2} />
        </FieldLabel>
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <FieldLabel model={MODEL} field="cfg_scale" text="cfg_scale (use 4 when an instruction is set)">
          <Input
            type="number"
            min="0.1"
            max="10"
            step="0.5"
            value={cfgScale}
            onChange={(event) => setField('cfgScale', event.target.value)}
            className="max-w-[8rem]"
          />
        </FieldLabel>
        <SeedField model={MODEL} value={seed} onChange={(value) => setField('seed', value)} />
        <PartitionField model={MODEL} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}
