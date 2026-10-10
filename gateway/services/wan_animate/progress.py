"""Live progress for a running Wan-Animate v1 (replace mode) job, read
straight from its own run.log -- see common/run_pipeline_job.py for how
that file is written (the full stdout+stderr of run_replace.py and
everything it shells out to, captured verbatim; not something built for
this purpose).

Nothing here is authoritative: it's a best-effort reading of another
project's own log output (ModelService_Wan-Animate-2/v1/run_replace.py, and
the Wan2.2 repo's generate.py underneath it), not a documented interface
those scripts promise to keep stable. If their log format ever changes,
read_progress degrades gracefully (missing fields go to None, or the whole
call returns None) rather than breaking GET /v1/jobs.

Zero third-party dependencies (stdlib only), same reasoning as
services/wan_animate/config.py: this is imported by server.py directly, not
run on a compute node, but there's no reason for it to need fastapi/torch
either.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from common.schemas import JobProgress, JobStage

# The stages run_replace.py's own _run_step() always prints, in order (see
# that script). mux_audio only actually runs when the request asked to
# keep audio (see build_replace_args in dispatch.py, and ReplaceRequest's
# own keep_audio field); when it doesn't, run_replace.py renames the
# silent render directly instead, with no stage marker printed for that
# step at all.
_ALWAYS_STAGES = ("preprocess", "generate")
_AUDIO_STAGE = "mux_audio"

_STAGE_STARTING = re.compile(r"^=== (\w+): starting ===$", re.MULTILINE)
_STAGE_FINISHED = re.compile(r"^=== (\w+): finished in ([\d.]+)s ===$", re.MULTILINE)

# Wan2.2's own hardcoded values for this deployment (see run_replace.py's
# own CLIP_LEN/REFERT_NUM constants, and dispatch.py's build_replace_args,
# which never overrides either from the request) -- ReplaceRequest has no
# field for either, so there's nowhere else to read them from. If
# run_replace.py's own constants ever change, this drifts out of sync with
# it and clip_count comes out wrong; stage tracking below is unaffected
# either way, since it doesn't depend on these.
_CLIP_LEN_FRAMES = 77
_REFERT_NUM = 1

_REAL_TARGET_FRAMES = re.compile(r"real frames:\s*(\d+)\s*target frames:\s*(\d+)")
# tqdm's standard bar text, e.g. "45%|####      | 9/20 [02:56<03:34, 19.53s/it]"
# -- matched anywhere in the text, not per-line: tqdm separates its own
# updates with a bare \r, not \n, when its output isn't a real terminal
# (confirmed against a real run.log), so naive line-splitting would merge
# many updates into what looks like one giant "line".
_TQDM_BAR = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]")
_RATE_S_PER_IT = re.compile(r"([\d.]+)\s*s/it")
_RATE_IT_PER_S = re.compile(r"([\d.]+)\s*it/s")


def _stage_list(text: str, keep_audio: bool) -> tuple[list[JobStage], str | None]:
    names = (*_ALWAYS_STAGES, _AUDIO_STAGE) if keep_audio else _ALWAYS_STAGES
    finished = dict(_STAGE_FINISHED.findall(text))  # name -> "146.4" (seconds, as str)
    started = set(_STAGE_STARTING.findall(text))  # single capture group -> plain strings, not tuples
    stages: list[JobStage] = []
    current: str | None = None
    for name in names:
        if name in finished:
            stages.append(JobStage(name=name, state="done", seconds=float(finished[name])))
        elif name in started:
            stages.append(JobStage(name=name, state="running", seconds=None))
            current = name
        else:
            stages.append(JobStage(name=name, state="pending", seconds=None))
    return stages, current


def _seconds_per_iteration(bracket_text: str) -> float | None:
    match = _RATE_S_PER_IT.search(bracket_text)
    if match:
        return float(match.group(1))
    match = _RATE_IT_PER_S.search(bracket_text)
    if match and float(match.group(1)) > 0:
        return 1.0 / float(match.group(1))
    return None


def _clip_progress(generate_text: str) -> dict[str, Any]:
    """clip/clip_count/step/step_count/eta_seconds for the `generate` stage
    -- {} if the log doesn't (yet) have enough to say anything (e.g. still
    loading checkpoints, before the first denoising step even starts)."""
    frames_match = _REAL_TARGET_FRAMES.search(generate_text)
    if not frames_match:
        return {}
    target_frames = int(frames_match.group(2))
    clip_count = (target_frames - _REFERT_NUM) // (_CLIP_LEN_FRAMES - _REFERT_NUM)
    if clip_count <= 0:
        return {}

    # Only look at bar text *after* the frames line -- excludes the earlier
    # "Loading checkpoint shards" bars, which use the same tqdm format but
    # aren't a denoising step.
    bars = list(_TQDM_BAR.finditer(generate_text[frames_match.end() :]))
    if not bars:
        return {"clip_count": clip_count}

    # Each clip's bar starts fresh at "0/step_count" -- count how many
    # such resets have happened so far to know which clip we're on.
    clip = max(sum(1 for bar in bars if bar.group(2) == "0"), 1)
    last = bars[-1]
    step, step_count = int(last.group(2)), int(last.group(3))

    seconds_per_it = None
    for bar in reversed(bars):
        seconds_per_it = _seconds_per_iteration(bar.group(4))
        if seconds_per_it is not None:
            break

    eta_seconds = None
    if seconds_per_it is not None:
        remaining_this_clip = max(step_count - step, 0)
        remaining_clips = max(clip_count - clip, 0)
        eta_seconds = seconds_per_it * (remaining_this_clip + remaining_clips * step_count)

    return {
        "clip": clip,
        "clip_count": clip_count,
        "step": step,
        "step_count": step_count,
        "eta_seconds": eta_seconds,
    }


def parse_progress(text: str, keep_audio: bool) -> JobProgress:
    """Pure parsing of one job's run.log text (or any prefix of it, e.g. a
    partially-written file mid-run) into a JobProgress. Deliberately
    tolerant of unexpected input in the clip-progress portion (wrapped in
    its own try/except below) -- a field it can't determine is left at its
    default (None), never guessed at. Stage tracking alone (regex findall
    over the whole text) has no realistic failure mode, so it isn't
    separately guarded; read_progress below is the actual safety net for
    anything this function doesn't anticipate."""
    stages, current_stage = _stage_list(text, keep_audio)

    clip_fields: dict[str, Any] = {}
    if current_stage == "generate":
        try:
            clip_fields = _clip_progress(text)
        except (ValueError, ArithmeticError):
            clip_fields = {}

    return JobProgress(stages=stages, current_stage=current_stage, **clip_fields)


def read_progress(job_dir: Path, request: dict[str, Any]) -> JobProgress | None:
    """For a job that's currently queued or running -- see server.py's
    _row_to_status, the only caller, which already only calls this for a
    queued/running wan-animate:replace row. Returns None on anything
    unexpected (run.log not written yet because the job is still queued,
    a permission hiccup, a shared-filesystem visibility lag, ...) so a bad
    read can never break GET /v1/jobs; the caller just shows no live
    progress for that poll, same as if this function didn't exist."""
    log_path = job_dir / "run.log"
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    try:
        return parse_progress(text, bool(request.get("keep_audio", True)))
    except Exception:  # best-effort by design -- see module docstring
        return None
