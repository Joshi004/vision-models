import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Textarea from '../components/ui/Textarea.jsx';
import Input from '../components/ui/Input.jsx';
import SegmentedControl from '../components/ui/SegmentedControl.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';

const MODEL = 'TextToAudioRequest';
const FORM_ID = 'ltx-text-to-audio';

const DEFAULTS = {
  prompt: '',
  negativePrompt: '',
  seed: 10,
  lengthMode: 'default',
  durationSeconds: 5,
  numFrames: 121,
  frameRate: 24,
  partition: '',
};

const REMEMBER_KEYS = ['seed', 'lengthMode', 'durationSeconds', 'numFrames', 'frameRate', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  let lengthMode = 'default';
  if (initialRequest.duration_seconds != null) lengthMode = 'duration';
  else if (initialRequest.num_frames != null) lengthMode = 'frames';
  return {
    prompt: initialRequest.prompt ?? '',
    negativePrompt: initialRequest.negative_prompt ?? '',
    seed: initialRequest.seed ?? 10,
    lengthMode,
    durationSeconds: initialRequest.duration_seconds ?? 5,
    numFrames: initialRequest.num_frames ?? 121,
    frameRate: initialRequest.frame_rate ?? 24,
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto TextToAudioRequest (services/ltx/schemas.py) -- audio
// only, no video, so no shape/orientation fields; just a length toggle
// (mirrors VideoShapeFields' duration/frames toggle, but without height/
// width/orientation, which this endpoint doesn't accept at all). See
// TextToVideoForm.jsx for how initialRequest/useFormState/FormShell fit
// together.
export default function TextToAudioForm({
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

  const { prompt, negativePrompt, seed, lengthMode, durationSeconds, numFrames, frameRate, partition } = values;

  const payload = {
    prompt,
    seed: seed === '' ? undefined : Number(seed),
    frame_rate: frameRate === '' ? undefined : Number(frameRate),
  };
  if (negativePrompt.trim()) payload.negative_prompt = negativePrompt;
  if (lengthMode === 'duration') payload.duration_seconds = durationSeconds === '' ? undefined : Number(durationSeconds);
  if (lengthMode === 'frames') payload.num_frames = numFrames === '' ? undefined : Number(numFrames);
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [...missingRequired(schemas, MODEL, payload)];
  if (lengthMode === 'duration' && payload.duration_seconds === undefined) {
    blockers.push('duration_seconds is required when "Duration (seconds)" is selected.');
  }
  if (lengthMode === 'frames' && payload.num_frames === undefined) {
    blockers.push('num_frames is required when "Exact frame count" is selected.');
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/ltx/audio/generate', payload));
  }

  return (
    <FormShell
      title="Text to audio"
      description="Audio only, no video."
      endpoint="POST /v1/ltx/audio/generate"
      prefillJob={prefillJob}
      onClearPrefill={onClearPrefill}
      restoredDraft={restoredDraft}
      onReset={reset}
      rememberSettings={rememberSettings}
      blockers={blockers}
      submitting={submitting}
      error={error}
      lastJobId={lastJobId}
      submitLabel="Generate audio"
      onSubmit={handleSubmit}
    >
      <Section title="Prompt">
        <FieldLabel model={MODEL} field="prompt" text="prompt (describes the desired soundscape: sources, timing, character)">
          <Textarea value={prompt} onChange={(event) => setField('prompt', event.target.value)} rows={3} />
        </FieldLabel>
      </Section>

      <Section title="Settings">
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-muted">length</span>
          <SegmentedControl
            name="audioLengthMode"
            value={lengthMode}
            onChange={(value) => setField('lengthMode', value)}
            options={[
              { value: 'default', label: 'Default (~5s)' },
              { value: 'duration', label: 'Duration (s)' },
              { value: 'frames', label: 'Exact frames' },
            ]}
          />
        </div>
        {lengthMode === 'duration' && (
          <FieldLabel model={MODEL} field="duration_seconds" text="duration_seconds" required>
            <Input
              type="number"
              min="0"
              step="0.5"
              value={durationSeconds}
              onChange={(event) => setField('durationSeconds', event.target.value)}
            />
          </FieldLabel>
        )}
        {lengthMode === 'frames' && (
          <FieldLabel model={MODEL} field="num_frames" text="num_frames (9, 17, ... 8k+1)" required>
            <Input type="number" min="9" step="8" value={numFrames} onChange={(event) => setField('numFrames', event.target.value)} />
          </FieldLabel>
        )}
        <FieldLabel model={MODEL} field="frame_rate" text="frame_rate">
          <Input
            type="number"
            min="1"
            step="1"
            value={frameRate}
            onChange={(event) => setField('frameRate', event.target.value)}
            className="max-w-[8rem]"
          />
        </FieldLabel>
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
