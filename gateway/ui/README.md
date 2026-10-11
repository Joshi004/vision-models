# model-api web UI

A small React + Vite app that calls the model-api server's own `/v1/...`
JSON API -- plain `fetch()`, no separate backend, no new validation logic.
Every field/default it shows comes straight from that recipe's Pydantic
model under `../services/<name>/schemas.py`; this UI never invents behavior
the API doesn't already have.

## Build (production)

```bash
cd model-api/ui
npm install
npm run build
```

This produces `dist/`, which `server.py` mounts at `/ui/` automatically
*if the directory exists* -- no server code change needed after a rebuild,
just restart the API process (or leave it running if you're editing
`ui/`'s own dev server instead, see below). If `dist/` doesn't exist yet,
the API still starts fine; `/ui` is simply disabled and a warning is logged.

After changing anything under `ui/src/`, re-run `npm run build` and restart
the API process to pick up the new `dist/`.

## Develop (hot reload against a running API)

```bash
cd model-api/ui
npm install
npm run dev
```

Starts Vite's own dev server (default `http://localhost:5173`), proxying
`/v1/...` requests to `http://localhost:8012` (see `vite.config.js`) --
point your browser at the Vite dev server directly while iterating, not at
the API's own `/ui` path.

## Layout

- `src/api.js` -- the only place that calls `fetch()`. Turns FastAPI's
  error bodies into a plain message string. No other file talks to the
  network directly.
- `src/utils.js` -- small shared helpers (image-conditioning list
  handling, the orientation/height-width and duration/frames toggles, the
  submit-a-job state machine, submit-blocker messages) reused across forms.
- `src/schema.js` -- loads and reads `GET /openapi.json` (fetched once in
  `App.jsx`). Every field's hover description and required-asterisk comes
  from here (see `components/FieldLabel.jsx`), straight from the same
  Pydantic models the API itself validates against -- never hand-copied,
  so neither can drift out of sync with the API. If that fetch fails,
  forms still work; they just show no tooltips/asterisks and rely on the
  API's own 422 messages instead.
- `src/components/` -- shared UI pieces (field labels/tooltips, progress
  bars, file upload, the jobs panel, result preview, etc.).
- `src/components/JobHistory.jsx` -- the full, filterable job history (see
  `src/components/JobRow.jsx` for one job's row, shared with the smaller
  `JobsPanel.jsx` strip shown under every form), with permanent delete via
  `DELETE /v1/jobs/{job_id}/purge`.
- Incognito mode: the switch in `components/Header.jsx` (state in `src/App.jsx`,
  remembered across visits) makes `postJSON` in `src/api.js` add
  `incognito: true` to every job submission -- the one place all forms submit
  through, so no form needs to know about it (`src/incognito.js` holds the
  current value). Jobs are otherwise kept in full until purged or
  `job_retention_days` pass. An incognito job (`job.incognito` in the jobs
  list, shown as a badge by `JobRow.jsx`) has its data deleted by the server:
  its result is a one-time download -- `components/ResultPreview.jsx` says so
  before loading it, and the browser's own copy (IndexedDB, see
  `resultCache.js`) is what remains. Retry is hidden for a job whose data was
  deleted (its `request` is `{}`), and clears reused upload ids from the
  pre-filled form of an incognito job, since its uploads are deleted when it
  ends (see `src/App.jsx`'s `withoutAssetIds`).
- `src/forms/` -- one form per generation recipe, each mapping 1:1 onto a
  request schema in `services/ltx/schemas.py`, `services/ltx25/schemas.py`,
  `services/wan_animate/schemas.py`, `services/rvc/schemas.py`, or
  `services/parakeet/schemas.py`. Where an LTX-2.5 recipe has the same fields
  as LTX-2.3's, the 2.3 form is reused through the recipe's `formProps` in
  `src/App.jsx` instead of copied: `RetakeForm.jsx` for retake, and
  `AudioToVideoForm.jsx` for audio-to-video (plus a "Match audio" length
  choice, `matchAudioLength`). The simplest
  one is `TranscribeForm.jsx` (Parakeet transcription): one file upload plus
  an optional partition override, nothing else -- its result is a JSON
  transcript, rendered by `components/TranscriptView.jsx` (full text with a
  copy button, a scrollable segment list, and a collapsible word-level list)
  in place of the usual `<audio>`/`<video>` element -- see
  `components/ResultPreview.jsx`'s `isTranscriptPipeline` branch.
