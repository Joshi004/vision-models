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
import Switch from '../components/ui/Switch.jsx';
import FieldLabel from '../components/FieldLabel.jsx';
import AssetUpload from '../components/AssetUpload.jsx';
import LyricsField from '../components/LyricsField.jsx';
import SeedField from '../components/SeedField.jsx';
import PartitionField from '../components/PartitionField.jsx';

// The three ACE-Step 1.5 endpoints share most fields, so one component serves all of them; `VARIANTS`
// holds the differences. Each maps 1:1 onto its request model (services/ace_step15/schemas.py).
const VARIANTS = {
  generate: {
    model: 'AceStepGenerateRequest',
    formId: 'ace-step15-generate',
    endpoint: '/v1/ace-step15/music/generate',
    title: 'ACE-Step 1.5: text to music',
    description:
      'Generate a song with vocals, or an instrumental, from a caption and lyrics. Strong at many languages and exact tempo, key and length. Put tempo and key in their own fields, not in the caption. Fix the seed while tuning.',
    submitLabel: 'Generate music',
  },
  cover: {
    model: 'AceStepCoverRequest',
    formId: 'ace-step15-cover',
    endpoint: '/v1/ace-step15/music/cover',
    title: 'ACE-Step 1.5: cover',
    description:
      'Restyle an existing recording and keep its structure. The caption describes the new style; the output is as long as the source.',
    submitLabel: 'Create cover',
  },
  repaint: {
    model: 'AceStepRepaintRequest',
    formId: 'ace-step15-repaint',
    endpoint: '/v1/ace-step15/music/repaint',
    title: 'ACE-Step 1.5: repaint',
    description:
      'Regenerate one time window of a recording and keep the rest. The caption describes what the new section should contain.',
    submitLabel: 'Repaint section',
  },
};

const LYRICS_TEMPLATE = '[Intro]\n\n[Verse 1]\n\n[Pre-Chorus]\n\n[Chorus]\n\n[Verse 2]\n\n[Chorus]\n\n[Bridge]\n\n[Chorus]\n\n[Outro]';
const TIME_SIGNATURES = [
  ['', 'Let the model choose'],
  ['2', '2/4'],
  ['3', '3/4'],
  ['4', '4/4'],
  ['6', '6/8'],
];

// Captions, lyrics and source recordings are never remembered between sessions; the settings are -- see storage.js.
const REMEMBER_KEYS = ['vocalLanguage', 'steps', 'guidance', 'thinking', 'seed', 'partition'];

function makeDefaults() {
  return {
    caption: '',
    lyrics: '',
    instrumental: false,
    vocalLanguage: 'unknown',
    duration: '',
    bpm: '',
    keyscale: '',
    timeSignature: '',
    thinking: true,
    source: null,
    coverStrength: '1',
    repaintStart: '',
    repaintEnd: '',
    steps: '50',
    guidance: '7',
    seed: 42,
    partition: '',
  };
}

function fromRequest(initialRequest) {
  if (!initialRequest) return null;
  return {
    caption: initialRequest.caption ?? '',
    lyrics: initialRequest.lyrics ?? '',
    instrumental: initialRequest.instrumental ?? false,
    vocalLanguage: initialRequest.vocal_language ?? 'unknown',
    duration: initialRequest.duration ?? '',
    bpm: initialRequest.bpm ?? '',
    keyscale: initialRequest.keyscale ?? '',
    timeSignature: initialRequest.time_signature ?? '',
    thinking: initialRequest.thinking ?? true,
    source: reusedAsset(initialRequest.source_audio_asset_id),
    coverStrength: String(initialRequest.cover_strength ?? 1),
    repaintStart: initialRequest.repaint_start ?? '',
    repaintEnd: initialRequest.repaint_end ?? '',
    steps: String(initialRequest.num_inference_steps ?? 50),
    guidance: String(initialRequest.guidance_scale ?? 7),
    seed: initialRequest.seed ?? 42,
    partition: initialRequest.partition ?? '',
  };
}

function AceStepForm({ variantId, defaultPartition, onSubmitted, initialRequest = null, prefillJob = null, onClearPrefill }) {
  const variant = VARIANTS[variantId];
  const { model } = variant;
  const isGenerate = variantId === 'generate';
  const isCover = variantId === 'cover';
  const isRepaint = variantId === 'repaint';

  const { values, setField, rememberSettings, reset, restoredDraft } = useFormState(variant.formId, {
    defaults: makeDefaults(),
    fromRequest: fromRequest(initialRequest),
    rememberKeys: REMEMBER_KEYS,
  });
  const [advancedOpen, setAdvancedOpen] = usePersistedState(`advancedOpen:${variant.formId}`, false);

  const schemas = useSchemas();
  const { submit, submitting, error, lastJobId } = useSubmitJob(onSubmitted);

  const {
    caption, lyrics, instrumental, vocalLanguage, duration, bpm, keyscale, timeSignature, thinking,
    source, coverStrength, repaintStart, repaintEnd, steps, guidance, seed, partition,
  } = values;

  const payload = {
    caption,
    instrumental,
    vocal_language: vocalLanguage.trim() || 'unknown',
    seed: seed === '' ? undefined : Number(seed),
  };
  if (lyrics.trim()) payload.lyrics = lyrics;
  if (steps !== '') payload.num_inference_steps = Number(steps);
  if (guidance !== '') payload.guidance_scale = Number(guidance);
  if (isGenerate) {
    if (duration !== '') payload.duration = Number(duration);
    if (bpm !== '') payload.bpm = Number(bpm);
    if (keyscale.trim()) payload.keyscale = keyscale.trim();
    if (timeSignature) payload.time_signature = timeSignature;
    payload.thinking = thinking;
  } else {
    payload.source_audio_asset_id = source?.status === 'ready' ? source.assetId : undefined;
    if (isCover && coverStrength !== '') payload.cover_strength = Number(coverStrength);
    if (isRepaint) {
      if (repaintStart !== '') payload.repaint_start = Number(repaintStart);
      if (repaintEnd !== '') payload.repaint_end = Number(repaintEnd);
    }
  }
  if (partition.trim()) payload.partition = partition.trim();

  const blockers = [...missingRequired(schemas, model, payload, isGenerate ? [] : ['source_audio_asset_id'])];
  if (caption.trim() === '' && !blockers.some((message) => message.startsWith('caption'))) {
    blockers.push('caption is required.');
  }
  if (!isGenerate) blockers.push(...assetBlocker(source, 'the source recording'));
  if (isRepaint) {
    if (repaintStart !== '' && repaintEnd !== '' && Number(repaintEnd) <= Number(repaintStart)) {
      blockers.push('repaint_end must be greater than repaint_start.');
    }
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
      {!isGenerate && (
        <Section title="Source recording">
          <AssetUpload
            model={model}
            field="source_audio_asset_id"
            label="source recording"
            accept="audio/*"
            value={source}
            onChange={(next) => setField('source', next)}
            required
          />
        </Section>
      )}

      <Section title="Caption">
        <FieldLabel model={model} field="caption" text="caption" required>
          <Textarea value={caption} onChange={(event) => setField('caption', event.target.value)} rows={3} />
        </FieldLabel>
      </Section>

      <Section title="Lyrics">
        <LyricsField model={model} value={lyrics} onChange={(value) => setField('lyrics', value)} template={LYRICS_TEMPLATE} />
        <Switch checked={instrumental} onChange={(value) => setField('instrumental', value)} label="Instrumental (no vocals)" />
        <FieldLabel model={model} field="vocal_language" text="vocal_language (en, zh, ja, ... or unknown)">
          <Input value={vocalLanguage} onChange={(event) => setField('vocalLanguage', event.target.value)} className="max-w-[10rem]" />
        </FieldLabel>
      </Section>

      {isGenerate && (
        <Section title="Music settings">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <FieldLabel model={model} field="duration" text="duration (s, 10-600)">
              <Input type="number" min="10" max="600" step="1" placeholder="auto" value={duration} onChange={(event) => setField('duration', event.target.value)} />
            </FieldLabel>
            <FieldLabel model={model} field="bpm" text="bpm">
              <Input type="number" min="30" max="300" step="1" placeholder="auto" value={bpm} onChange={(event) => setField('bpm', event.target.value)} />
            </FieldLabel>
            <FieldLabel model={model} field="keyscale" text="keyscale">
              <Input placeholder="A minor" value={keyscale} onChange={(event) => setField('keyscale', event.target.value)} />
            </FieldLabel>
            <FieldLabel model={model} field="time_signature" text="time_signature">
              <Select value={timeSignature} onChange={(event) => setField('timeSignature', event.target.value)}>
                {TIME_SIGNATURES.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </Select>
            </FieldLabel>
          </div>
        </Section>
      )}

      {isCover && (
        <Section title="Cover">
          <FieldLabel model={model} field="cover_strength" text="cover_strength (1 = follow the source closely, 0.2 = loose style transfer)">
            <Input type="number" min="0" max="1" step="0.05" value={coverStrength} onChange={(event) => setField('coverStrength', event.target.value)} className="max-w-[8rem]" />
          </FieldLabel>
        </Section>
      )}

      {isRepaint && (
        <Section title="Section to regenerate">
          <div className="grid grid-cols-2 gap-3">
            <FieldLabel model={model} field="repaint_start" text="repaint_start (s)" required>
              <Input type="number" min="0" step="0.1" value={repaintStart} onChange={(event) => setField('repaintStart', event.target.value)} />
            </FieldLabel>
            <FieldLabel model={model} field="repaint_end" text="repaint_end (s, empty = to the end)">
              <Input type="number" min="0" step="0.1" value={repaintEnd} onChange={(event) => setField('repaintEnd', event.target.value)} />
            </FieldLabel>
          </div>
        </Section>
      )}

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={setAdvancedOpen}>
        {isGenerate && (
          <Switch checked={thinking} onChange={(value) => setField('thinking', value)} label="Thinking (let the language model plan BPM, key and duration first)" />
        )}
        <div className="grid grid-cols-2 gap-3">
          <FieldLabel model={model} field="num_inference_steps" text="num_inference_steps (default 50)">
            <Input type="number" min="1" max="100" step="1" value={steps} onChange={(event) => setField('steps', event.target.value)} />
          </FieldLabel>
          <FieldLabel model={model} field="guidance_scale" text="guidance_scale (default 7)">
            <Input type="number" min="0" max="30" step="0.5" value={guidance} onChange={(event) => setField('guidance', event.target.value)} />
          </FieldLabel>
        </div>
        <SeedField model={model} value={seed} onChange={(value) => setField('seed', value)} />
        <PartitionField model={model} value={partition} onChange={(value) => setField('partition', value)} defaultPartition={defaultPartition} />
      </CollapsibleSection>
    </FormShell>
  );
}

export function AceStepGenerateForm(props) {
  return <AceStepForm variantId="generate" {...props} />;
}

export function AceStepCoverForm(props) {
  return <AceStepForm variantId="cover" {...props} />;
}

export function AceStepRepaintForm(props) {
  return <AceStepForm variantId="repaint" {...props} />;
}
