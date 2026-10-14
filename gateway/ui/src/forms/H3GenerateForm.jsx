import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { assetBlocker, reusedAsset, useSubmitJob } from '../utils.js';
import { GENERATE_PROMPT_TEMPLATE, defaultH3Shape, h3ShapeBlockers, h3ShapeFromRequest, toH3ShapePayload } from '../h3.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Textarea from '../components/ui/Textarea.jsx';
import Input from '../components/ui/Input.jsx';
import Button from '../components/ui/Button.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';
import H3ShapeFields from '../components/H3ShapeFields.jsx';

const MODEL = 'H3GenerateRequest';
const FORM_ID = 'h3-generate';

const DEFAULTS = {
  prompt: '',
  firstFrame: null,
  lastFrame: null,
  steps: '',
  seed: 10,
  shape: defaultH3Shape(),
  partition: '',
};

// Only these carry over as this form's defaults the next time it's opened
// fresh -- see storage.js. The prompt and the two frames are specific to one
// generation, so they're left out.
const REMEMBER_KEYS = ['steps', 'seed', 'shape', 'partition'];

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    prompt: initialRequest.prompt ?? '',
    firstFrame: reusedAsset(initialRequest.first_frame_asset_id),
    lastFrame: reusedAsset(initialRequest.last_frame_asset_id),
    steps: initialRequest.num_inference_steps ?? '',
    seed: initialRequest.seed ?? 10,
    shape: h3ShapeFromRequest(initialRequest),
    partition: initialRequest.partition ?? '',
  };
}

// Maps 1:1 onto H3GenerateRequest (services/h3/schemas.py): a prompt, and
// optionally a first and/or last frame -- none is text-to-video. See
// TextToVideoForm.jsx for how initialRequest/useFormState/FormShell fit together.
export default function H3GenerateForm({ defaultPartition, onSubmitted, initialRequest = null, prefillJob = null, onClearPrefill }) {
  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(FORM_ID, {
    defaults: DEFAULTS,
    fromRequest: fromRequest(initialRequest),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${FORM_ID}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { prompt, firstFrame, lastFrame, steps, seed, shape, partition } = values;

  const payload = {
    prompt,
    first_frame_asset_id: firstFrame?.status === 'ready' ? firstFrame.assetId : undefined,
    last_frame_asset_id: lastFrame?.status === 'ready' ? lastFrame.assetId : undefined,
    seed: seed === '' ? undefined : Number(seed),
    num_inference_steps: steps === '' ? undefined : Number(steps),
    ...toH3ShapePayload(shape),
  };
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [
    ...missingRequired(schemas, MODEL, payload),
    // Both frames are optional: only a file that is still uploading blocks.
    ...(firstFrame ? assetBlocker(firstFrame, 'the first frame') : []),
    ...(lastFrame ? assetBlocker(lastFrame, 'the last frame') : []),
    ...h3ShapeBlockers(shape),
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON('/v1/h3/videos/generate', payload));
  }

  return (
    <FormShell
      title="Text / frame to video (MiniMax H3)"
      description="Video with generated stereo sound from a prompt, optionally starting from a first frame and/or ending on a last frame. Every job holds two GPUs and starts cold. The prompt reaches the model exactly as typed (nothing rewrites it) and is expected in H3's structured format -- see the MiniMax H3 section of /v1/guide."
      endpoint="POST /v1/h3/videos/generate"
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
      <Section title="Frames (optional)">
        <div className="grid gap-4 sm:grid-cols-2">
          <AssetUpload
            model={MODEL}
            field="first_frame_asset_id"
            label="first frame"
            accept="image/*"
            value={firstFrame}
            onChange={(next) => setField('firstFrame', next)}
          />
          <AssetUpload
            model={MODEL}
            field="last_frame_asset_id"
            label="last frame"
            accept="image/*"
            value={lastFrame}
            onChange={(next) => setField('lastFrame', next)}
          />
        </div>
      </Section>

      <Section title="Prompt">
        <FieldLabel model={MODEL} field="prompt" text="prompt">
          <Textarea
            value={prompt}
            onChange={(event) => setField('prompt', event.target.value)}
            rows={10}
            className="font-mono text-xs"
          />
        </FieldLabel>
        {prompt.trim() === '' && (
          <Button type="button" variant="secondary" size="sm" onClick={() => setField('prompt', GENERATE_PROMPT_TEMPLATE)} className="self-start">
            Insert prompt template
          </Button>
        )}
        {(firstFrame || lastFrame) && (
          <p className="text-xs text-muted">
            With a frame, start the prompt with the instruction line that places <code>&lt;Picture 1&gt;</code> (see the guide), then
            describe the motion that connects the frames.
          </p>
        )}
      </Section>

      <Section title="Settings">
        <H3ShapeFields
          model={MODEL}
          shape={shape}
          onChange={(next) => setField('shape', next)}
          defaultCanvasHint="H3's own 16:9 canvas (1344x768), or the first frame's aspect ratio when a first frame is given."
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
