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
import FieldHeading from '../components/FieldHeading.jsx';
import LyricsField from '../components/LyricsField.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';

// The two YuE2 endpoints share most fields, so one component serves both; `VARIANTS` holds the differences.
// Each maps 1:1 onto its request model (services/yue2/schemas.py).
const VARIANTS = {
  generate: {
    model: 'Yue2GenerateRequest',
    formId: 'yue2-generate',
    endpoint: '/v1/yue2/songs/generate',
    title: 'YuE2: song from style and lyrics',
    description:
      'Generate a full song with sung vocals. The style line starts with the language, then genre, instruments, vocal character and tempo (there is no BPM field). Each request makes one candidate: try several seeds. Weights are CC BY-NC 4.0 (non-commercial).',
    submitLabel: 'Generate song',
    defaultCot: 'full',
    cotOptions: [
      { value: 'full', label: 'Full plan (melody + chords)' },
      { value: 'melody', label: 'Melody plan' },
      { value: 'off', label: 'No plan (faster, lower quality)' },
    ],
  },
  cover: {
    model: 'Yue2CoverRequest',
    formId: 'yue2-cover',
    endpoint: '/v1/yue2/songs/cover',
    title: 'YuE2: cover from a melody score',
    description:
      'Generate a song that follows an ABC melody score you supply, in a new style. The score needs the model\'s native format (voices Vocal and Ins; no chord symbols for a melody-only score), and the lyrics must follow its sections in order. Turning a recording into a score is not part of this API. Weights are CC BY-NC 4.0 (non-commercial).',
    submitLabel: 'Generate cover',
    defaultCot: 'melody',
    cotOptions: [
      { value: 'melody', label: 'Melody plan (recommended)' },
      { value: 'full', label: 'Full plan (fixed harmony)' },
    ],
  },
};

const LYRICS_TEMPLATE = '[Verse]\n\n[Chorus]\n\n[Verse]\n\n[Chorus]\n\n[Bridge]\n\n[Chorus]';
const STYLE_PLACEHOLDER = 'English, warm piano pop, expressive female voice, acoustic piano, light drums, 88 BPM';

// Style, lyrics and the score are never remembered between sessions; the settings are -- see storage.js.
const REMEMBER_KEYS = ['cot', 'seed', 'partition'];

function makeDefaults(variant) {
  return { style: '', lyrics: '', abc: '', cot: variant.defaultCot, cfgScale: '', seed: 831001, partition: '' };
}

function fromRequest(initialRequest, variant) {
  if (!initialRequest) return null;
  return {
    style: initialRequest.style ?? '',
    lyrics: initialRequest.lyrics ?? '',
    abc: initialRequest.abc ?? '',
    cot: initialRequest.cot ?? variant.defaultCot,
    cfgScale: initialRequest.cfg_scale ?? '',
    seed: initialRequest.seed ?? 831001,
    partition: initialRequest.partition ?? '',
  };
}

function Yue2Form({ variantId, defaultPartition, onSubmitted, initialRequest = null, prefillJob = null, onClearPrefill }) {
  const variant = VARIANTS[variantId];
  const { model } = variant;
  const isCover = variantId === 'cover';

  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(variant.formId, {
    defaults: makeDefaults(variant),
    fromRequest: fromRequest(initialRequest, variant),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${variant.formId}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const { style, lyrics, abc, cot, cfgScale, seed, partition } = values;

  const payload = {
    style,
    lyrics,
    cot,
    seed: seed === '' ? undefined : Number(seed),
  };
  if (isCover) payload.abc = abc;
  if (cfgScale !== '') payload.cfg_scale = Number(cfgScale);
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [...missingRequired(schemas, variant.model, payload)];
  for (const [field, value] of [['style', style], ['lyrics', lyrics], ...(isCover ? [['abc', abc]] : [])]) {
    if (value.trim() === '' && !blockers.some((message) => message.startsWith(field))) blockers.push(`${field} is required.`);
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
      <Section title="Style">
        <FieldLabel model={model} field="style" text="style" required>
          <Textarea value={style} onChange={(event) => setField('style', event.target.value)} rows={2} placeholder={STYLE_PLACEHOLDER} />
        </FieldLabel>
      </Section>

      <Section title="Lyrics">
        <LyricsField model={model} value={lyrics} onChange={(value) => setField('lyrics', value)} template={LYRICS_TEMPLATE} required />
      </Section>

      {isCover && (
        <Section title="Melody score (ABC)">
          <FieldLabel model={model} field="abc" text="abc" required>
            <Textarea value={abc} onChange={(event) => setField('abc', event.target.value)} rows={10} className="font-mono" />
          </FieldLabel>
        </Section>
      )}

      <Section title="Planning">
        <div className="flex flex-col gap-1.5">
          <FieldHeading model={model} field="cot" text="cot" />
          <SegmentedControl
            name={`${variant.formId}-cot`}
            value={cot}
            onChange={(value) => setField('cot', value)}
            options={variant.cotOptions}
          />
        </div>
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        <FieldLabel model={model} field="cfg_scale" text="cfg_scale (leave empty for the tuned default; upstream suggests trying 1.2)">
          <Input
            type="number"
            min="0"
            max="20"
            step="0.05"
            placeholder="default"
            value={cfgScale}
            onChange={(event) => setField('cfgScale', event.target.value)}
            className="max-w-[8rem]"
          />
        </FieldLabel>
        <SeedField model={model} value={seed} onChange={(value) => setField('seed', value)} />
        <PartitionField model={model} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}

export function Yue2GenerateForm(props) {
  return <Yue2Form variantId="generate" {...props} />;
}

export function Yue2CoverForm(props) {
  return <Yue2Form variantId="cover" {...props} />;
}
