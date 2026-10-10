import { emptyAudioEntry } from '../utils.js';
import AssetUpload from './AssetUpload.jsx';
import Button from './ui/Button.jsx';
import { XIcon } from './icons.jsx';

// Repeatable rows of { asset } -- BatchConvertRequest's own `sources`
// field (services/rvc/schemas.py's BatchSource: just one asset_id per
// entry). Modeled on ImageConditioningList.jsx, just without that
// component's per-row frame_idx/strength/crf fields -- a batch source is
// nothing but a file. `minEntries` only affects when the "Remove" button
// is offered (a soft UI guard); the parent form computes the real submit
// blockers from these entries via utils.js's audioListBlockers, and the
// API itself is still what actually enforces the minimum either way.
//
// `onChange` is always called with an updater function, `(previousEntries)
// => nextEntries`, never a plain array -- same reasoning as
// ImageConditioningList's own onChange contract (see that file's comment):
// several rows can be mid-upload at once, each with its own AssetUpload
// progress callback firing independently, and only an updater function
// composes safely against the latest pending value rather than whatever
// `entries` this component's last render happened to close over.
export default function AudioFileList({ label, entries, onChange, minEntries = 0 }) {
  function updateEntry(index, patch) {
    onChange((previous) => previous.map((entry, i) => (i === index ? { ...entry, ...patch } : entry)));
  }

  function addEntry() {
    onChange((previous) => [...previous, emptyAudioEntry()]);
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
              model="BatchSource"
              field="asset_id"
              label={`Recording ${index + 1}`}
              accept="audio/*"
              value={entry.asset}
              onChange={(asset) => updateEntry(index, { asset })}
              required
            />
          </div>
          {entries.length > minEntries && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => removeEntry(index)}
              aria-label={`Remove recording ${index + 1}`}
            >
              <XIcon className="h-3.5 w-3.5" />
              Remove
            </Button>
          )}
        </div>
      ))}
      <Button type="button" variant="secondary" size="sm" onClick={addEntry} className="self-start">
        Add recording
      </Button>
    </div>
  );
}
