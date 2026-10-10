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
export default function ReferenceImageList({ model, field, images, onChange, minImages = 1, maxImages = 10 }) {
  function updateImage(index, asset) {
    onChange((previous) => previous.map((item, i) => (i === index ? { asset } : item)));
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
              label={`Reference image ${index + 1}`}
              accept="image/*"
              value={entry.asset}
              onChange={(next) => updateImage(index, next)}
              required
            />
          </div>
          {images.length > minImages && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => removeImage(index)}
              aria-label={`Remove reference image ${index + 1}`}
            >
              <XIcon className="h-3.5 w-3.5" />
              Remove
            </Button>
          )}
        </div>
      ))}
      {images.length < maxImages && (
        <Button type="button" variant="secondary" size="sm" onClick={addImage} className="self-start">
          Add reference image
        </Button>
      )}
    </div>
  );
}
