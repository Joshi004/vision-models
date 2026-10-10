import FieldLabel from './FieldLabel.jsx';
import Input from './ui/Input.jsx';
import Select from './ui/Select.jsx';
import SegmentedControl from './ui/SegmentedControl.jsx';

// Mirrors VideoShapeMixin (services/ltx/schemas.py): exactly one of
// orientation or explicit height/width, and exactly one of duration_seconds
// or num_frames. Modeled here as two independent toggles instead of six
// loose fields, so the UI can't easily build the invalid "both"
// combination the API would otherwise reject with a 422.
//
// `model` selects which concrete request schema (TextToVideoRequest,
// KeyframeInterpolationRequest, or AudioToVideoRequest) these fields'
// hover descriptions come from -- each declares them separately (even
// though the text is often identical), so this can't be hardcoded to one.
//
// `allowAutoLength` adds a fourth length choice, "Auto (from prompt)" --
// only meaningful for a backend whose request has an `auto_duration` flag
// (LTX-2.5, see Ltx25TextToVideoForm.jsx). Choosing it sets
// shape.lengthMode to 'auto', which toShapePayload (utils.js) turns into no
// length fields at all; the form itself is what sends `auto_duration: true`.
export default function VideoShapeFields({ model, shape, onChange, allowAutoLength = false }) {
  function patch(fields) {
    onChange({ ...shape, ...fields });
  }

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-border p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted">Shape &amp; length</p>

      <div className="flex flex-col gap-1.5">
        <SegmentedControl
          name="shapeMode"
          value={shape.shapeMode}
          onChange={(value) => patch({ shapeMode: value })}
          options={[
            { value: 'orientation', label: 'Orientation' },
            { value: 'custom', label: 'Custom height/width' },
          ]}
        />
      </div>

      {shape.shapeMode === 'orientation' ? (
        <FieldLabel model={model} field="orientation" text="orientation">
          <Select value={shape.orientation} onChange={(event) => patch({ orientation: event.target.value })}>
            <option value="landscape">landscape (1920x1088)</option>
            <option value="portrait">portrait (1088x1920)</option>
          </Select>
        </FieldLabel>
      ) : (
        <div className="grid grid-cols-2 gap-3">
          <FieldLabel model={model} field="height" text="height (multiple of 64)" required>
            <Input type="number" step="64" min="64" value={shape.height} onChange={(event) => patch({ height: event.target.value })} />
          </FieldLabel>
          <FieldLabel model={model} field="width" text="width (multiple of 64)" required>
            <Input type="number" step="64" min="64" value={shape.width} onChange={(event) => patch({ width: event.target.value })} />
          </FieldLabel>
        </div>
      )}

      <SegmentedControl
        name="lengthMode"
        value={shape.lengthMode}
        onChange={(value) => patch({ lengthMode: value })}
        options={[
          { value: 'default', label: 'Default (~5s)' },
          { value: 'duration', label: 'Duration (s)' },
          { value: 'frames', label: 'Exact frames' },
          ...(allowAutoLength ? [{ value: 'auto', label: 'Auto (from prompt)' }] : []),
        ]}
      />

      {shape.lengthMode === 'duration' && (
        <FieldLabel model={model} field="duration_seconds" text="duration_seconds" required>
          <Input
            type="number"
            min="0"
            step="0.5"
            value={shape.durationSeconds}
            onChange={(event) => patch({ durationSeconds: event.target.value })}
          />
        </FieldLabel>
      )}
      {shape.lengthMode === 'frames' && (
        <FieldLabel model={model} field="num_frames" text="num_frames (9, 17, ... 8k+1)" required>
          <Input
            type="number"
            min="9"
            step="8"
            value={shape.numFrames}
            onChange={(event) => patch({ numFrames: event.target.value })}
          />
        </FieldLabel>
      )}

      <FieldLabel model={model} field="frame_rate" text="frame_rate">
        <Input
          type="number"
          min="1"
          step="1"
          value={shape.frameRate}
          onChange={(event) => patch({ frameRate: event.target.value })}
          className="max-w-[8rem]"
        />
      </FieldLabel>
    </div>
  );
}
