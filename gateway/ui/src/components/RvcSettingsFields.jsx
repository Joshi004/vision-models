import { useEffect, useState } from 'react';
import { listVoices } from '../api.js';
import FieldLabel from './FieldLabel.jsx';
import PartitionField from './PartitionField.jsx';
import Section from './ui/Section.jsx';
import CollapsibleSection from './ui/CollapsibleSection.jsx';
import Input from './ui/Input.jsx';
import Select from './ui/Select.jsx';

// export_format's own full option list -- identical on both
// POST /v1/rvc/convert (ConvertRequest) and POST /v1/rvc/batch-convert
// (BatchConvertRequest), since both inherit it from the same base
// (services/rvc/schemas.py's ConversionSettings). No M4A on either: it's
// never been offered on batch-convert (Applio's WAV-to-M4A step can fail
// silently on this deployment, with no M4A encoder in the installed audio
// library), and was removed from /convert for the same reason -- it used
// to silently return a file named .m4a that actually contained plain WAV
// bytes. Exported so both RvcConvertForm.jsx and RvcBatchForm.jsx can pass
// it explicitly (readable at each call site, rather than relying on this
// being the default).
export const EXPORT_FORMATS = ['WAV', 'MP3', 'FLAC', 'OGG'];

// The voice dropdown plus every other RVC conversion setting (pitch/
// f0_method/index_rate/protect/export_format) -- shared by both
// POST /v1/rvc/convert (RvcConvertForm.jsx) and POST /v1/rvc/batch-convert
// (RvcBatchForm.jsx), since both request schemas inherit these exact same
// fields from one base (services/rvc/schemas.py's ConversionSettings).
// Kept here as a single copy so the two forms can't slowly drift apart on
// fields/defaults/help text they're actually required to share.
//
// `model` selects which concrete request schema (ConvertRequest or
// BatchConvertRequest) this component's field docs/required-markers are
// read from (see schema.js) -- both currently document these fields
// identically, but read live per-model anyway rather than assumed, in
// case that ever changes.
//
// `values`/`setField` are the parent form's own useFormState state/setter
// (see storage.js) -- this component calls setField('voice', ...) etc.
// directly on the parent's own state, exactly like each field's inline
// handler did before this was extracted from RvcConvertForm.jsx, so
// neither form's stored draft/settings shape had to change to adopt this.
//
// `advancedOpen`/`onAdvancedToggle` are the parent's own
// usePersistedState pair for its "Advanced" section (see
// CollapsibleSection.jsx) -- passed through rather than owned here so each
// form keeps its own independent open/closed state (and its own
// `advancedOpen:<FORM_ID>` storage key), the same as every other form's
// Advanced section already works.
//
// `partition` is included here too, not left to each parent form --
// PartitionOptionMixin is the other half of ConversionSettings
// (services/rvc/schemas.py: `ConversionSettings(PartitionOptionMixin)`),
// so it's exactly as shared between ConvertRequest/BatchConvertRequest as
// voice/pitch/f0_method/index_rate/protect/export_format are.
// `defaultPartition` is passed straight through to PartitionField (see
// that component -- it only affects the input's placeholder text).
export default function RvcSettingsFields({
  model,
  values: { voice, pitch, f0Method, indexRate, protect, exportFormat, partition },
  setField,
  advancedOpen,
  onAdvancedToggle,
  defaultPartition,
  exportFormats = EXPORT_FORMATS,
}) {
  const [voices, setVoices] = useState([]);
  const [voicesError, setVoicesError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    listVoices()
      .then((result) => {
        if (!cancelled) setVoices(result.voices);
      })
      .catch((err) => {
        if (!cancelled) setVoicesError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Nothing chosen yet (a fresh form, or a restored draft from before any
  // voices existed) -- default to the first one once the list loads,
  // rather than leaving the dropdown on an empty option.
  useEffect(() => {
    if (!voice && voices.length > 0) setField('voice', voices[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voices]);

  return (
    <>
      <Section title="Voice">
        <FieldLabel model={model} field="voice" text="voice">
          <Select value={voice} onChange={(event) => setField('voice', event.target.value)} disabled={voices.length === 0}>
            {voices.length === 0 && <option value="">{voicesError ? 'Could not load voices' : 'Loading\u2026'}</option>}
            {voices.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </FieldLabel>
        {voicesError && (
          <p className="text-xs text-danger">
            Could not load the voice list ({voicesError}). Reload the page once the API is reachable.
          </p>
        )}
      </Section>

      <CollapsibleSection title="Advanced" open={advancedOpen} onToggle={onAdvancedToggle}>
        <FieldLabel model={model} field="pitch" text="pitch (semitones)">
          <Input
            type="number"
            min={-24}
            max={24}
            step={1}
            value={pitch}
            onChange={(event) => setField('pitch', event.target.value === '' ? '' : Number(event.target.value))}
          />
        </FieldLabel>

        <FieldLabel model={model} field="f0_method" text="f0_method (pitch extraction)">
          <Select value={f0Method} onChange={(event) => setField('f0Method', event.target.value)}>
            <option value="rmvpe">rmvpe (default \u2014 fast, high accuracy)</option>
            <option value="crepe">crepe (slower, most accurate)</option>
            <option value="crepe-tiny">crepe-tiny (fastest, least accurate)</option>
            <option value="fcpe">fcpe (fast, modern alternative)</option>
          </Select>
        </FieldLabel>

        <FieldLabel model={model} field="index_rate" text="index_rate">
          <Input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={indexRate}
            onChange={(event) => setField('indexRate', event.target.value === '' ? '' : Number(event.target.value))}
          />
        </FieldLabel>

        <FieldLabel model={model} field="protect" text="protect">
          <Input
            type="number"
            min={0}
            max={0.5}
            step={0.01}
            value={protect}
            onChange={(event) => setField('protect', event.target.value === '' ? '' : Number(event.target.value))}
          />
        </FieldLabel>

        <FieldLabel model={model} field="export_format" text="export_format">
          <Select value={exportFormat} onChange={(event) => setField('exportFormat', event.target.value)}>
            {exportFormats.map((format) => (
              <option key={format} value={format}>
                {format}
              </option>
            ))}
          </Select>
        </FieldLabel>

        <PartitionField
          model={model}
          value={partition}
          onChange={(value) => setField('partition', value)}
          defaultPartition={defaultPartition}
        />
      </CollapsibleSection>
    </>
  );
}
