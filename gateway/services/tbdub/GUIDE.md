## TBDub lip sync

**What this is for:** making the mouth of the speaker in an existing video match an audio track
exactly. The audio is never regenerated: your exact audio is muxed into the result at its original
quality. The model is TBDub's 30-step Teacher, an open-weight visual-dubbing model.

**Two ways to get lip sync**
1. **As part of an LTX audio-to-video request** -- add `lip_sync: true` to
   `POST /v1/ltx/videos/audio-to-video` (LTX-2.3) or `POST /v1/ltx25/videos/audio-to-video` (LTX-2.5).
   One request, one job, no second upload: the video is generated, then its mouth is re-drawn to the
   same audio. Choose the LTX version first; lip sync is applied on top of whichever you pick.
2. **On its own** -- `POST /v1/tbdub/videos/lip-sync`, for a video and audio you upload, or for an LTX
   audio-to-video job that already ran (`source_job_id`).

**When lip sync fails inside an LTX job** (for example no face could be tracked), the job does **not**
fail: it still reports `succeeded` and returns the video **without** lip sync. The job's `lip_sync`
field tells you: `lip_sync.applied` is `false` and `lip_sync.error` says why. Always check
`lip_sync.applied` on a succeeded job that asked for lip sync. To retry only the sync (no new LTX
generation), call `POST /v1/tbdub/videos/lip-sync` with `{"source_job_id": "<that job's id>"}` --
optionally with a different `seed`, or a higher `audio_cfg_scale` if the mouth under-articulated.

**Frame rate.** TBDub works at 25 fps. With `lip_sync: true` an LTX request is generated at 25 fps
(`frame_rate` is overridden, and an explicit `num_frames` is converted to the same *duration*, not the
same frame count); the stored request shows the values actually used. The standalone endpoint accepts
any frame rate and converts to 25 fps. A lip-synced result is always 25 fps.

**What it changes.** Only what speech moves: lips, teeth, jaw and the skin around them. Eyes, eyebrows,
head motion and the overall expression stay as they are in the video, so a laugh, a raised eyebrow or a
nod has to be in the video already -- for LTX, describe it in the prompt (describe what is *seen*; the
dialogue itself does not need to be quoted). A laugh in the *audio* is likely to come out as ordinary
talking mouth shapes.

**What the video must look like.** One speaker with a visible face. If no face is found for more than
about 0.4 s the sync fails (inside an LTX job: see above). The face is processed as a 512x512 crop and
pasted back, so on a very tight close-up the mouth can look slightly softer than the rest of the frame.

**Deleting.** Everything a lip-sync step writes lives in its job's own folder, so
`DELETE /v1/jobs/{job_id}/purge` removes all of it. An LTX job with lip sync is one job (one history
entry, deleted as one), and keeps its un-synced video in that folder until then. A standalone
lip-sync job is always a separate job, deleted on its own -- even one started from `source_job_id`,
which copies the source's files, so purging either job leaves the other intact.

**Time.** Measured on this cluster: lip sync adds about 3 minutes to an LTX job for a 5 s clip (about 2
minutes of loading and face tracking, plus roughly 12 s of GPU time per second of video), so a 10 s clip adds
about 5 minutes. It runs on the same GPU right after the LTX step, so there is no extra queue wait. A
standalone lip-sync job takes the same time for the same clip. The sync step is stopped after at most 30 minutes
(the job's own time limit is extended by that much), and a stopped sync is a failed sync: you still get the video.

**Results are large.** The lip-synced file is encoded near-losslessly (about 13 MB for a 5 s portrait 1080p
clip, several times the size of the un-synced video).

**Responsible use:** only sync faces and recordings you have the right to use, and disclose generated
media as such. This API does not enforce usage restrictions itself.
