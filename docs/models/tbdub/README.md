# TBDub (lip sync)

TBDub ([TaoLiveAIGC/TBDub](https://github.com/TaoLiveAIGC/TBDub), Apache-2.0, released 2026-09) is an open-weight
visual-dubbing model: given an existing video and an audio track, it redraws the speaker's mouth to match the audio.
This project runs its **30-step Teacher** (the highest-quality variant; a distilled 2-step Student also exists and is
about 14x faster in the authors' benchmark, but is not wired in). Why it was chosen over LatentSync, KeySync, Lip
Forcing and X-Dub: `docs/research/LIP_SYNC_RESEARCH.md` section 10.

It is used two ways:

- `lip_sync: true` on `POST /v1/ltx/videos/audio-to-video` and `POST /v1/ltx25/videos/audio-to-video`: the LTX video is
  generated, then lip-synced, as two steps of **one Slurm job on one GPU**.
- `POST /v1/tbdub/videos/lip-sync`: its own job, for an uploaded video and audio, or for a finished LTX
  audio-to-video job (`source_job_id`).

## How it works (read from its code)

1. MediaPipe finds the face in every frame; the face box is smoothed over time.
2. The face is cropped to a square and resized to **512x512**.
3. The audio is turned into speech features (HuBERT) at 16 kHz.
4. A video diffusion model (Wan2.2-5B base) takes the cropped original face video and the audio features and
   produces a new 512x512 face video: 30 denoising steps per 77-frame chunk, 5 frames of motion carried between
   chunks, so clips can be any length.
5. The result is colour-corrected to the original, feathered into the frame, and pasted back.

**There is no prompt.** The code passes an empty text prompt and a precomputed "empty" embedding; the text pathway
exists only because the Wan backbone expects one. The controls are:

- `audio_cfg_scale` ("follow the voice"), default 6. Higher: the mouth follows the audio more strongly.
- `ref_cfg_scale` ("stay like the original"), default 2. Higher: skin, teeth and face shape stay closer to the
  input, and the original mouth's own motion is copied more.
- the seed, and the step count (fixed at 30 here).

(How the two scales work: at each step the model makes three guesses of what the face should look like: with
nothing, with the original video, and with the original video plus the audio. `ref_cfg_scale` is how far to push
from the first guess toward the second; `audio_cfg_scale` is how far to push from the second toward the third. 1.0
means no extra push. These are also why the 30-step version is about 14x slower than the 2-step one: it runs all
three guesses at every step.) The shipped `config.json` uses 2.0 and 6.0, the same as the code's defaults; the
paper's benchmark used 2.5 and 10.

**What it changes.** It re-renders the whole face crop, but it was trained on pairs of videos that differ only in
speech motion, so it changes lips, teeth, jaw and the cheeks and chin around them, and copies eyes, eyebrows, head
pose and expression from the input. Consequences:

- Eyebrows, laughter, nods and other reactions have to be in the input video already. For LTX, describe them in the
  prompt.
- If the input holds one strong expression (for example a broad smile), the redrawn mouth stays close to it: the
  mouth is cleaner and follows the audio, but articulation can be subtle (see "Measured" below).
- A laugh in the *audio* is likely to come out as ordinary talking mouth shapes (inferred from the training, not tested).

## Setup

```bash
scripts/build_env.sh tbdub            # Python 3.10, torch 2.9.1 (CUDA 12.8 build), numpy 1.26, mediapipe 0.10.21, ...
models/tbdub/download_weights.sh      # ~15 GB into $VISION_DATA_DIR/weights/tbdub, none of it gated
```

It has its own environment because its dependencies conflict with the LTX ones (numpy 1.x against 2.x, transformers
5.1 against 5.14, Python 3.10 against 3.12, mediapipe and jax). The upstream code is the pinned submodule
`third_party/TBDub`. The gateway side is `gateway/services/tbdub/`; the Slurm-side wrapper is
`models/tbdub/runner/run_lipsync.py`.

## How a combined job runs

`gateway/common/run_pipeline_job.py` (standard library only) runs the LTX command, renames its video to
`lipsync_base.mp4`, and then runs the command in `post_step.json` (the lip-sync wrapper, with TBDub's own interpreter) as a
separate process. LTX has exited by then, so its GPU memory is free. If lip sync fails, the base video is copied to
`output.mp4` and `lip_sync.json` records the reason; the job still succeeds (`lip_sync.applied: false` in the job
status). The un-synced base video stays in the job folder in both cases, so every file is deleted with the job
(`DELETE /v1/jobs/{job_id}/purge`, `purge_all_data.sh`, retention).

## Things the wrapper does because TBDub does not

| TBDub's behavior | What `run_lipsync.py` does |
|---|---|
| Treats every frame as 1/25 s and never checks the input frame rate | Re-encodes any other frame rate to 25 fps first, keeping the duration. LTX jobs with `lip_sync` are generated at 25 fps so this is a no-op for them. |
| Fills a video shorter than the audio by playing it backwards | Cuts the audio to the video's exact length (silence-padded by half a frame, so rounding cannot drop a frame). |
| Calls a bare `ffmpeg` command | Puts a directory with an `ffmpeg` link to the `imageio-ffmpeg` binary first on `PATH` (the only ffmpeg on this cluster). |
| Muxes the audio as 16 kHz mono AAC | Re-muxes the cut audio at its original sample rate and channels, copying TBDub's video stream. |
| Reports a failure as the last line of a chained traceback | Prints the root-cause exception as the final `ERROR:` line, which is what the job status shows. |

## Limits

- **One speaker, face visible.** One face is tracked. If no face is found for more than about 0.4 s (10 frames at
  25 fps) it fails with "MediaPipe face track has a gap longer than 10 frames". Seen for real: an LTX-2.5 clip in
  which the speaker held a product up until it covered her mouth, as the camera pushed in. Inside an LTX job this is
  the fallback path above.
- **512x512 face crop**, pasted back. On a very tight close-up the mouth can look slightly softer than the rest of
  the frame.
- Output is **25 fps**, H.264 at CRF 10: the file is several times larger than a typical LTX output (about 13 MB for
  5 s at 1088x1920).
- Validated by the authors on English, Chinese, Japanese, Korean and Russian with an English-trained audio encoder.
  Other languages (for example Hindi) are untested.
- The authors' own numbers are the only published quality comparison; see the research doc for the caveat.

## Measured on this cluster (2026-10-12, H100, portrait 1088x1920, 5 s clip)

| | Result |
|---|---|
| Standalone job, default settings | 3m07s-3m13s per job (about 2 min loading and face tracking, plus 2 chunks x ~34 s of denoising at ~1.13 s per step) |
| Peak GPU memory | ~29 GB (sampled every 3 s inside the job) |
| Peak host RAM | ~16 GB (standalone); the combined LTX-2.5 job peaked at ~68 GB, mostly LTX |
| Lip-sync step inside an LTX job | 181 s (LTX-2.5) and 189 s (LTX-2.3) |
| Scaling | about 12 s of GPU time per second of video, plus the fixed ~2 min |
| Output size | ~13 MB for 5 s (near-lossless, CRF 10), against 2-5 MB for the un-synced video |

Quality, from tracking the mouth opening with MediaPipe landmarks and from side-by-side mouth crops of the un-synced
and the synced video (the files are kept in `/home/naresh/vision-data-lipsync-test/review/`):

- During silences the mouth now stays still (the original LTX-2.5 mouth jittered open), and it opens with the louder
  bursts of speech.
- On clips where the speaker holds a broad smile (it came from the reference photo), the articulation stays subtle
  except in the strongest syllables, because TBDub keeps the source expression.
- `audio_cfg_scale` 6 to 10: no visible difference on this clip. `ref_cfg_scale` 2 to 1: visibly more mouth change
  (lips closing in a pause, opening wider on a burst), at a small cost in staying like the original.
- So the base video matters: ask LTX for a talking or neutral expression rather than a held smile, and describe
  any wanted reactions (eyebrows, a laugh) in its prompt.
