import { postJSON } from '../api.js';
import { missingRequired, useSchemas } from '../schema.js';
import { assetBlocker, reusedAsset, useSubmitJob } from '../utils.js';
import { useFormState, usePersistedState } from '../storage.js';
import FormShell from '../components/FormShell.jsx';
import Section from '../components/ui/Section.jsx';
import CollapsibleSection from '../components/ui/CollapsibleSection.jsx';
import Textarea from '../components/ui/Textarea.jsx';
import Input from '../components/ui/Input.jsx';
import Select from '../components/ui/Select.jsx';
import SegmentedControl from '../components/ui/SegmentedControl.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import FieldHeading from '../components/FieldHeading.jsx';
import ReferenceImageList from '../components/ReferenceImageList.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';

// The two Qwen-Image-2.1 endpoints share almost everything, so one component
// serves both; `VARIANTS` holds the few differences. Each maps 1:1 onto its
// request model (services/qwen_image21/schemas.py).
const VARIANTS = {
  generate: {
    model: 'QwenImageGenerateRequest',
    formId: 'qwen-image21-generate',
    endpoint: '/v1/qwen-image21/images/generate',
    title: 'Qwen-Image-2.1: text to image',
    description:
      'Generate an image from a prompt. Fast uses the 8-step Turbo checkpoint, Quality the 40-step base checkpoint. For a transparent background, word the prompt as: "This is an RGBA image with transparency. <description>. The image has alpha channel and the background is transparent." Weights are for research / non-commercial use only.',
    submitLabel: 'Generate image',
    defaultAspectRatio: '1:1',
  },
  edit: {
    model: 'QwenImageEditRequest',
    formId: 'qwen-image21-edit',
    endpoint: '/v1/qwen-image21/images/edit',
    title: 'Qwen-Image-2.1: edit / compose images',
    description:
      'Edit one image, or compose a new one from up to 10 reference images. Describe the change in the prompt (refer to several images in order). Weights are for research / non-commercial use only.',
    submitLabel: 'Edit image',
    defaultAspectRatio: '',
  },
};

const ASPECT_RATIOS = [
  ['1:1', '1:1 (2048x2048)'],
  ['4:3', '4:3 (2400x1792)'],
  ['3:4', '3:4 (1792x2400)'],
  ['3:2', '3:2 (2528x1696)'],
  ['2:3', '2:3 (1696x2528)'],
  ['16:9', '16:9 (2752x1536)'],
  ['9:16', '9:16 (1536x2752)'],
];

const MAX_REFERENCE_IMAGES = 10;
const DEFAULT_QUALITY_STEPS = 40;

// Prompts and reference images are never remembered between sessions; the
// settings are -- see storage.js.
const REMEMBER_KEYS = ['mode', 'aspectRatio', 'steps', 'seed', 'partition'];

function makeDefaults(variant) {
  return {
    prompt: '',
    mode: 'fast',
    aspectRatio: variant.defaultAspectRatio,
    steps: '',
    images: [{ asset: null }],
    seed: 42,
    partition: '',
  };
}

function fromRequest(initialRequest, variant) {
  if (!initialRequest) return null;
  const ids = initialRequest.reference_image_asset_ids;
  return {
    prompt: initialRequest.prompt ?? '',
    mode: initialRequest.mode ?? 'fast',
    aspectRatio: initialRequest.aspect_ratio ?? variant.defaultAspectRatio,
    steps: initialRequest.num_inference_steps ?? '',
    images: Array.isArray(ids) && ids.length > 0 ? ids.map((id) => ({ asset: reusedAsset(id) })) : [{ asset: null }],
    seed: initialRequest.seed ?? 42,
    partition: initialRequest.partition ?? '',
  };
}

function QwenImageForm({
  variantId,
  defaultPartition,
  onSubmitted,
  initialRequest = null,
  prefillJob = null,
  onClearPrefill,
}) {
  const variant = VARIANTS[variantId];
  const isEdit = variantId === 'edit';
  const { model } = variant;

  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(variant.formId, {
    defaults: makeDefaults(variant),
    fromRequest: fromRequest(initialRequest, variant),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${variant.formId}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { prompt, mode, aspectRatio, steps, images, seed, partition } = values;

  const payload = {
    prompt,
    mode,
    seed: seed === '' ? undefined : Number(seed),
  };
  if (aspectRatio) payload.aspect_ratio = aspectRatio;
  if (mode === 'quality' && steps !== '') payload.num_inference_steps = Number(steps);
  if (isEdit) {
    payload.reference_image_asset_ids = images
      .filter((entry) => entry.asset?.status === 'ready')
      .map((entry) => entry.asset.assetId);
  }
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [...missingRequired(schemas, model, payload)];
  if (prompt.trim() === '' && !blockers.some((message) => message.startsWith('prompt'))) {
    blockers.push('prompt is required.');
  }
  if (isEdit) {
    images.forEach((entry, index) => {
      blockers.push(...assetBlocker(entry.asset, `reference image ${index + 1}`));
    });
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (blockers.length > 0) return;
    await submit(() => postJSON(variant.endpoint, payload));
  }

  return (
    <FormShell
      title={variant.title}
      description={variant.description}
      endpoint={`POST ${variant.endpoint}`}
      prefillJob={prefillJob}
      onClearPrefill={onClearPrefill}
      restoredDraft={restoredDraft}
      onReset={reset}
      rememberSettings={rememberSettings}
      blockers={blockers}
      submitting={submitting}
      error={error}
      lastJobId={lastJobId}
      submitLabel={variant.submitLabel}
      onSubmit={handleSubmit}
    >
      {isEdit && (
        <Section title="Reference images">
          <ReferenceImageList
            model={model}
            field="reference_image_asset_ids"
            images={images}
            onChange={(next) => setField('images', next)}
            maxImages={MAX_REFERENCE_IMAGES}
          />
        </Section>
      )}

      <Section title="Prompt">
        <FieldLabel model={model} field="prompt" text="prompt" required>
          <Textarea value={prompt} onChange={(event) => setField('prompt', event.target.value)} rows={6} />
        </FieldLabel>
      </Section>

      <Section title="Settings">
        <div className="flex flex-col gap-1.5">
          <FieldHeading model={model} field="mode" text="mode" />
          <SegmentedControl
            name={`${variant.formId}-mode`}
            value={mode}
            onChange={(value) => setField('mode', value)}
            options={[
              { value: 'fast', label: 'Fast (Turbo, 8 steps)' },
              { value: 'quality', label: 'Quality (base, slower)' },
            ]}
          />
        </div>
        <FieldLabel model={model} field="aspect_ratio" text={isEdit ? 'aspect_ratio (default: follow the last reference image)' : 'aspect_ratio'}>
          <Select value={aspectRatio} onChange={(event) => setField('aspectRatio', event.target.value)} className="max-w-[16rem]">
            {isEdit && <option value="">Follow reference image</option>}
            {ASPECT_RATIOS.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </Select>
        </FieldLabel>
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        {mode === 'quality' && (
          <FieldLabel model={model} field="num_inference_steps" text={`num_inference_steps (default ${DEFAULT_QUALITY_STEPS})`}>
            <Input
              type="number"
              min="1"
              max="100"
              step="1"
              placeholder={String(DEFAULT_QUALITY_STEPS)}
              value={steps}
              onChange={(event) => setField('steps', event.target.value)}
              className="max-w-[8rem]"
            />
          </FieldLabel>
        )}
        <SeedField model={model} value={seed} onChange={(value) => setField('seed', value)} />
        <PartitionField model={model} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}

export function QwenImageGenerateForm(props) {
  return <QwenImageForm variantId="generate" {...props} />;
}

export function QwenImageEditForm(props) {
  return <QwenImageForm variantId="edit" {...props} />;
}
