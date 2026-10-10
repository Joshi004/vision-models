import { emptyImageEntry } from '../utils.js';
import AssetUpload from './AssetUpload.jsx';
import FieldLabel from './FieldLabel.jsx';
import Input from './ui/Input.jsx';
import Button from './ui/Button.jsx';
import { XIcon } from './icons.jsx';

// Repeatable rows of {asset, frame_idx, strength, crf} -- the `images`
// field on the text-to-video/audio-to-video endpoints, or the `keyframes`
// field on keyframe-interpolation (see services/ltx/schemas.py's
// ImageConditioning model, which every row's frame_idx/strength/crf field
// help is read from). `minEntries` only affects when the "Remove" button
// is offered (a soft UI guard); the parent form computes the real submit
// blockers from these entries via utils.js's imageListBlockers, and the
// API itself is still what actually enforces the minimum either way.
//
// `onChange` is always called with an updater function, `(previousEntries)
// => nextEntries`, never a plain array -- so it composes safely with
// useFormState's setField (see storage.js), which applies the update
// against the latest pending value rather than whatever `entries` this
// component's last render happened to close over. That matters here
// specifically because two rows can be mid-upload at the same time, each
// with its own AssetUpload progress callbacks firing independently: with a
// plain-array onChange, the second row's callback could overwrite the
// first row's just-applied change if both fire before either render
// commits.
export default function ImageConditioningList({ label, entries, onChange, minEntries = 0 }) {
  function updateEntry(index, patch) {
    onChange((previous) => previous.map((entry, i) => (i === index ? { ...entry, ...patch } : entry)));
  }

  function addEntry() {
    onChange((previous) => [...previous, emptyImageEntry()]);
  }

  function removeEntry(index) {
    onChange((previous) => previous.filter((_, i) => i !== index));
  }

  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted">{label}</p>
      {entries.map((entry, index) => (
        <div key={index} className="flex flex-wrap items-end gap-3 rounded-lg border border-border p-3">
          <div className="min-w-[12rem] flex-1">
            <AssetUpload
              model="ImageConditioning"
              field="asset_id"
              label={`Image ${index + 1}`}
              accept="image/*"
              value={entry.asset}
              onChange={(asset) => updateEntry(index, { asset })}
              required
            />
          </div>
          <div className="w-24">
            <FieldLabel model="ImageConditioning" field="frame_idx" text="frame_idx">
              <Input
                type="number"
                min="0"
                value={entry.frameIdx}
                onChange={(event) => updateEntry(index, { frameIdx: event.target.value })}
              />
            </FieldLabel>
          </div>
          <div className="w-24">
            <FieldLabel model="ImageConditioning" field="strength" text="strength">
              <Input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={entry.strength}
                onChange={(event) => updateEntry(index, { strength: event.target.value })}
              />
            </FieldLabel>
          </div>
          <div className="w-24">
            <FieldLabel model="ImageConditioning" field="crf" text="crf">
              <Input
                type="number"
                min="0"
                value={entry.crf}
                onChange={(event) => updateEntry(index, { crf: event.target.value })}
              />
            </FieldLabel>
          </div>
          {entries.length > minEntries && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => removeEntry(index)}
              aria-label={`Remove image ${index + 1}`}
            >
              <XIcon className="h-3.5 w-3.5" />
              Remove
            </Button>
          )}
        </div>
      ))}
      <Button type="button" variant="secondary" size="sm" onClick={addEntry} className="self-start">
        Add image
      </Button>
    </div>
  );
}
