## Wan-Animate v1 -- character replacement in existing video

**What this is for:** swapping the person in an existing video for a reference character, while
keeping that video's background, camera motion, and lighting -- "replace" mode, in the underlying
model's (Wan2.2-Animate-14B) own terminology. This is editing of *existing* footage, not generating
a new scene from a text prompt -- there is no `prompt` field on this endpoint at all, and the
official model's own authors advise against trying to add one.

**Use this instead of LTX-2.3 when** you already have a real video and want to replace who's in it,
not generate a new clip from scratch.

**Inputs:** a driving video (`video_asset_id`) and a single reference photo of the character to
swap in (`image_asset_id`), both uploaded first via `POST /v1/uploads`.

**Known hard limits, from the official model's own documentation, not just this API's own
observation:**
- **Single-person video only.** The preprocessing step that extracts the person's mask from the
  driving video is documented as designed for single-person footage; a video with more than one
  person on screen may produce an incorrect mask, a garbled swap, or an outright failure.
- **Body-proportion mismatches between the reference character and the original person can produce
  visible artifacts** -- the pose skeleton driving the swap doesn't reshape to fit a very different
  body shape.
- Cost scales with the driving video's length -- the model processes it in short segments rather
  than all at once, so a longer driving video takes proportionally longer, not a fixed amount of
  time regardless of length.

**Timing:** a real measured run on this deployment (a 6.8s, 1280x720 driving video) took ~25 minutes
of wall-clock time end to end -- long enough that this endpoint's job spends meaningfully longer
exposed to a pre-emptible partition's pre-emption risk than a typical short-video request on another
backend does (a job killed mid-run by higher-priority cluster work loses the whole run, not just the
remainder, and surfaces as `status: "failed"`). Consider setting `partition` to a non-pre-emptible
one (see `GET /v1/partitions`) if that risk isn't acceptable for a given job.

**Live progress:** unlike most backends here, a running job on this endpoint reports fine-grained
`progress` (current stage, which clip/step, a rough time-remaining estimate) -- see the main guide's
"Jobs and timing" section and this endpoint's own schema for the `progress` field's shape.

**Output:** MP4, H.264 video. Resolution/frame rate follow the source video's own shape (resized so
its pixel area roughly matches 1280x720, aspect ratio preserved) at 30 fps -- there are no
`height`/`width`/`orientation` fields on this endpoint. The model itself only ever produces silent
video; the source video's original audio is muxed back on afterward unless `keep_audio` is set to
`false`.