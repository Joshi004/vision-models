import FieldLabel from './FieldLabel.jsx';
import Input from './ui/Input.jsx';
import Select from './ui/Select.jsx';
import SegmentedControl from './ui/SegmentedControl.jsx';
import { H3_DOCUMENTED_RANGE, H3_MAX_FRAMES, H3_MAX_SECONDS, H3_MAX_SIDE, H3_MIN_FRAMES, H3_MIN_SECONDS, H3_MIN_SIDE, H3_SIZE_MULTIPLE } from '../h3.js';

// Canvas and length fields for the MiniMax H3 forms; the state shape and its
// mapping to the API live in h3.js. Mirrors H3ShapeMixin
// (services/h3/schemas.py): at most one of orientation or height/width, and at
// most one of duration_seconds or num_frames -- modeled as two independent
// toggles so the UI cannot build the invalid "both" combination.
//
// `model` selects which request schema (H3GenerateRequest or
// H3AudioToVideoRequest) the hover descriptions come from.
// `defaultCanvasHint` says what "Model default" means for that form.
// `allowMatchAudio` adds a length choice for audio-to-video, "Match audio",
// which the form itself sends as `match_audio_duration: true`.
export default function H3ShapeFields({ model, shape, onChange, defaultCanvasHint, allowMatchAudio = false }) {
  function patch(fields) {
    onChange({ ...shape, ...fields });
  }

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-border p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted">Shape &amp; length</p>

      <SegmentedControl
        name="h3ShapeMode"
        value={shape.shapeMode}
        onChange={(value) => patch({ shapeMode: value })}
        options={[
          { value: 'default', label: 'Model default' },
          { value: 'orientation', label: 'Orientation' },
          { value: 'custom', label: 'Custom height/width' },
        ]}
      />

      {shape.shapeMode === 'default' && <p className="text-xs text-muted">{defaultCanvasHint}</p>}

      {shape.shapeMode === 'orientation' && (
        <FieldLabel model={model} field="orientation" text="orientation">
          <Select value={shape.orientation} onChange={(event) => patch({ orientation: event.target.value })}>
            <option value="landscape">landscape (1344x768)</option>
            <option value="portrait">portrait (768x1344)</option>
          </Select>
        </FieldLabel>
      )}

      {shape.shapeMode === 'custom' && (
        <div className="grid grid-cols-2 gap-3">
          <FieldLabel model={model} field="height" text={`height (multiple of ${H3_SIZE_MULTIPLE})`} required>
            <Input
              type="number"
              step={H3_SIZE_MULTIPLE}
              min={H3_MIN_SIDE}
              max={H3_MAX_SIDE}
              value={shape.height}
              onChange={(event) => patch({ height: event.target.value })}
            />
          </FieldLabel>
          <FieldLabel model={model} field="width" text={`width (multiple of ${H3_SIZE_MULTIPLE})`} required>
            <Input
              type="number"
              step={H3_SIZE_MULTIPLE}
              min={H3_MIN_SIDE}
              max={H3_MAX_SIDE}
              value={shape.width}
              onChange={(event) => patch({ width: event.target.value })}
            />
          </FieldLabel>
        </div>
      )}

      <SegmentedControl
        name="h3LengthMode"
        value={shape.lengthMode}
        onChange={(value) => patch({ lengthMode: value })}
        options={[
          { value: 'default', label: 'Default (5.17 s)' },
          { value: 'duration', label: 'Duration (s)' },
          { value: 'frames', label: 'Exact frames' },
          ...(allowMatchAudio ? [{ value: 'match', label: 'Match audio' }] : []),
        ]}
      />

      {shape.lengthMode === 'duration' && (
        <FieldLabel model={model} field="duration_seconds" text={`duration_seconds (${H3_MIN_SECONDS}-${H3_MAX_SECONDS})`} required>
          <Input
            type="number"
            min={H3_MIN_SECONDS}
            max={H3_MAX_SECONDS}
            step="0.1"
            value={shape.durationSeconds}
            onChange={(event) => patch({ durationSeconds: event.target.value })}
          />
        </FieldLabel>
      )}
      {shape.lengthMode === 'frames' && (
        <FieldLabel model={model} field="num_frames" text={`num_frames at 24 fps (${H3_MIN_FRAMES}-${H3_MAX_FRAMES}, rounded up to 17n+5)`} required>
          <Input
            type="number"
            min={H3_MIN_FRAMES}
            max={H3_MAX_FRAMES}
            step="1"
            value={shape.numFrames}
            onChange={(event) => patch({ numFrames: event.target.value })}
          />
        </FieldLabel>
      )}
      {shape.lengthMode === 'match' && (
        <p className="text-xs text-muted">The video is as long as the audio reference, rounded up to the next 17n+5 frames (no minimum).</p>
      )}
      <p className="text-xs text-muted">
        Fixed 24 fps. Lengths are rounded up to the next 17n+5 frames. The model is documented for {H3_DOCUMENTED_RANGE} (124-345
        frames); anything shorter or longer is experimental and may look wrong, fail, or run out of GPU memory.
      </p>
    </div>
  );
}
