import AssetUpload from './AssetUpload.jsx';
import Button from './ui/Button.jsx';
import { XIcon } from './icons.jsx';

// A repeatable list of image uploads -- the `reference_image_asset_ids`
// field on Qwen-Image-2.1's edit endpoint. `images` is an array of
// `{ asset }` entries, where `asset` is AssetUpload's own value (null /
// uploading / ready, see AssetUpload.jsx); the parent form turns the ready
// ones into asset ids. The `{ asset }` wrapper (same as ImageConditioningList's
// entries) is what storage.js's clearStaleUploads knows how to clean up
// after a page reload.
//
// `onChange` is always called with an updater function, `(previous) =>
// next`, never a plain array, for the same reason as ImageConditioningList:
// several rows can be mid-upload at once, and each row's progress callbacks
// must apply to the latest list rather than whatever this render closed over.
//
// `itemLabel`, `accept` and `addLabel` default to the image wording above; the
// MiniMax H3 audio-to-video form reuses this same list for its reference
// videos (`accept="video/*"`) and reference audio (`accept="audio/*,video/*"`),
// which are `{ asset }` entries just the same.
//
// `renderExtra(entry, index)`, when given, is drawn under a row's file
// picker for per-row options that live next to the file (the H3 form's "use
// this video's soundtrack" checkbox). Such a field is stored on the entry
// itself, beside `asset`, which is why replacing a row's file keeps the
// rest of the entry.
export default function ReferenceImageList({
  model,
  field,
  images,
  onChange,
  minImages = 1,
  maxImages = 10,
  itemLabel = 'Reference image',
  accept = 'image/*',
  addLabel = 'Add reference image',
  renderExtra = null,
}) {
  function updateImage(index, asset) {
    onChange((previous) => previous.map((item, i) => (i === index ? { ...item, asset } : item)));
  }

  function addImage() {
    onChange((previous) => [...previous, { asset: null }]);
  }

  function removeImage(index) {
    onChange((previous) => previous.filter((_, i) => i !== index));
  }

  return (
    <div className="flex flex-col gap-3">
      {images.map((entry, index) => (
        <div key={index} className="flex flex-wrap items-end gap-3 rounded-lg border border-border p-3">
          <div className="min-w-[12rem] flex-1">
            <AssetUpload
              model={model}
              field={field}
              label={`${itemLabel} ${index + 1}`}
              accept={accept}
              value={entry.asset}
              onChange={(next) => updateImage(index, next)}
              required
            />
            {renderExtra && renderExtra(entry, index)}
          </div>
          {images.length > minImages && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => removeImage(index)}
              aria-label={`Remove ${itemLabel.toLowerCase()} ${index + 1}`}
            >
              <XIcon className="h-3.5 w-3.5" />
              Remove
            </Button>
          )}
        </div>
      ))}
      {images.length < maxImages && (
        <Button type="button" variant="secondary" size="sm" onClick={addImage} className="self-start">
          {addLabel}
        </Button>
      )}
    </div>
  );
}
