// Helpers shared by the two MiniMax H3 forms (forms/H3GenerateForm.jsx and
// forms/H3AudioToVideoForm.jsx): the canvas/length state, how it maps to and
// from the API's fields, the UI-only blockers, and prompt templates.
//
// H3's shape rules differ from LTX's (utils.js's defaultShape/toShapePayload),
// so they are kept apart: no frame_rate (fixed 24 fps), sizes are multiples of
// 32, lengths are rounded up onto a 17*n+5 frame grid (5-15 s is the documented, no longer enforced, range), and leaving the canvas alone is
// a real choice ("model default": H3's own 16:9, or the first frame's aspect
// ratio) -- see H3ShapeMixin in services/h3/schemas.py, which these mirror.

export const H3_MIN_SIDE = 256;
export const H3_MAX_SIDE = 1536;
export const H3_SIZE_MULTIPLE = 32;
// The model is documented for 5-15 s (124-345 frames); anything outside that is
// experimental, and the API no longer enforces it. The UI minimums are just the
// smallest sensible inputs (the API takes any duration above 0 and any frame
// count from 1). The maximums mirror the API's default ceiling (H3_MAX_SECONDS,
// 60 s); an operator who changes it also has to change these.
export const H3_MIN_SECONDS = 0.1;
export const H3_MAX_SECONDS = 60;
export const H3_MIN_FRAMES = 1;
export const H3_MAX_FRAMES = 1440;
export const H3_DOCUMENTED_RANGE = '5-15 s';

export function defaultH3Shape() {
  return {
    // 'default' sends no canvas fields, 'orientation' sends orientation,
    // 'custom' sends height + width.
    shapeMode: 'default',
    orientation: 'landscape',
    height: '',
    width: '',
    // 'default' sends no length fields, 'duration' / 'frames' send that field,
    // 'match' (audio-to-video only) is sent by the form as match_audio_duration.
    lengthMode: 'default',
    durationSeconds: 6,
    numFrames: 124,
  };
}

// A blank field is left out of the payload entirely (JSON.stringify drops
// undefined-valued keys) so the API's own default applies.
export function toH3ShapePayload(shape) {
  const payload = {};
  if (shape.shapeMode === 'orientation') {
    payload.orientation = shape.orientation;
  } else if (shape.shapeMode === 'custom') {
    payload.height = shape.height === '' ? undefined : Number(shape.height);
    payload.width = shape.width === '' ? undefined : Number(shape.width);
  }
  if (shape.lengthMode === 'duration') {
    payload.duration_seconds = shape.durationSeconds === '' ? undefined : Number(shape.durationSeconds);
  } else if (shape.lengthMode === 'frames') {
    payload.num_frames = shape.numFrames === '' ? undefined : Number(shape.numFrames);
  }
  return payload;
}

// The inverse of toH3ShapePayload, for Retry (see App.jsx).
export function h3ShapeFromRequest(request) {
  const shape = defaultH3Shape();
  if (request.height != null && request.width != null) {
    shape.shapeMode = 'custom';
    shape.height = request.height;
    shape.width = request.width;
  } else if (request.orientation) {
    shape.shapeMode = 'orientation';
    shape.orientation = request.orientation;
  }
  if (request.match_audio_duration) {
    shape.lengthMode = 'match';
  } else if (request.duration_seconds != null) {
    shape.lengthMode = 'duration';
    shape.durationSeconds = request.duration_seconds;
  } else if (request.num_frames != null) {
    shape.lengthMode = 'frames';
    shape.numFrames = request.num_frames;
  }
  return shape;
}

function outOfRange(value, min, max) {
  const number = Number(value);
  return !Number.isFinite(number) || number < min || number > max;
}

// Submit-blocker messages for the rules the API would otherwise reject with a 422.
export function h3ShapeBlockers(shape) {
  const messages = [];
  if (shape.shapeMode === 'custom') {
    for (const [name, value] of [
      ['height', shape.height],
      ['width', shape.width],
    ]) {
      if (value === '') {
        messages.push(`${name} is required when using custom height/width.`);
      } else if (outOfRange(value, H3_MIN_SIDE, H3_MAX_SIDE) || Number(value) % H3_SIZE_MULTIPLE !== 0) {
        messages.push(`${name} must be a multiple of ${H3_SIZE_MULTIPLE} between ${H3_MIN_SIDE} and ${H3_MAX_SIDE}.`);
      }
    }
  }
  if (shape.lengthMode === 'duration' && outOfRange(shape.durationSeconds, H3_MIN_SECONDS, H3_MAX_SECONDS)) {
    messages.push(`duration_seconds must be between ${H3_MIN_SECONDS} and ${H3_MAX_SECONDS}.`);
  }
  if (shape.lengthMode === 'frames' && outOfRange(shape.numFrames, H3_MIN_FRAMES, H3_MAX_FRAMES)) {
    messages.push(`num_frames must be between ${H3_MIN_FRAMES} and ${H3_MAX_FRAMES}.`);
  }
  return messages;
}

// Starting points for the prompt box -- the structured formats from MiniMax's
// own prompt-writing guides (see gateway/services/h3/GUIDE.md). Nothing rewrites
// a prompt on the server, so what is typed here is exactly what the model reads.
export const GENERATE_PROMPT_TEMPLATE = `integrated_multimodal_description: [Shot 1] Live-action, cinematic, <shot size and composition>. <Who or what is in frame, and what they do.> The camera <motion type> with small amplitude at slow speed. <The speaker, described once, (S1)> says: <d>[English] <the words, exactly as they should be spoken.></d>

overall_soundscape: <Ambience, physical sounds and non-verbal human sounds. No dialogue here.>

non_diegetic_music: <Instrumentation, tempo and dynamics of a score the characters cannot hear, or N/A.>`;

export const REFERENCE_PROMPT_TEMPLATE = `subject_definitions:
<Subject 1> is <the person in <Picture 1>, with their key visible features>.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] <One short sentence on what the video shows and how each reference is used.>

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - <what is kept>.
<Audio 1>: reference - the target speaker follows <Audio 1>'s voice timbre and delivery without copying the original signal.

detailed_description:
<One or two sentences on the overall style.>
[Shot 1] <Composition, action and camera.> <Subject 1> (S1) says, using the voice timbre referenced from <Audio 1>, <d>[English] <the words.></d>

overall_soundscape:
<Ambience and physical sounds.>

non_diegetic_music:
N/A`;
