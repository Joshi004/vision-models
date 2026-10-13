"""Builds TBDub lip-sync invocations and submits them as Slurm jobs.

Two uses of the same runner (models/tbdub/runner/run_lipsync.py):

  * `dispatch_lip_sync` -- the standalone `POST /v1/tbdub/videos/lip-sync` job.
  * `build_post_step` -- the second step of an LTX audio-to-video job that was
    submitted with `lip_sync: true`. The LTX dispatch modules write it to the
    job's `post_step.json`; common/run_pipeline_job.py runs it after the LTX
    step has exited, as a separate process with TBDub's own interpreter.

Every file a lip-sync job touches lives inside that job's own folder
(jobs/<job_id>/), so DELETE /v1/jobs/{job_id}/purge, purge_all_data.sh and the
retention sweep delete all of it with no lip-sync-specific code. A standalone
job started from `source_job_id` copies the source's video and audio into its
own folder, so the two jobs are independent: purging one never affects the
other.

See services/ltx/dispatch.py for the fuller rationale behind the submission
plumbing mirrored below (capacity check, partial-path rename convention).
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path

from common import config as common_config
from common import job_store, retention, storage
from common.slurm import resolve_partition
from services.tbdub import config
from services.tbdub.schemas import LipSyncRequest


class DispatchError(RuntimeError):
    """Could not submit the job (sbatch failure, ...)."""


class CapacityError(RuntimeError):
    """Too many jobs already in flight (see common.config.MAX_INFLIGHT_JOBS)."""


class InputError(RuntimeError):
    """The request names a source job that cannot be used (wrong kind of job)
    or an input that can no longer be found -> 400."""


class NotFoundError(RuntimeError):
    """`source_job_id` does not exist (never did, or was deleted) -> 404."""


class ConflictError(RuntimeError):
    """The source job exists but is not in a state that can be re-synced
    (not finished, failed, or its data was removed) -> 409."""


def _check_capacity() -> None:
    if (
        common_config.MAX_INFLIGHT_JOBS is not None
        and job_store.count_inflight_jobs() >= common_config.MAX_INFLIGHT_JOBS
    ):
        raise CapacityError(
            f"At capacity: {common_config.MAX_INFLIGHT_JOBS} job(s) already in flight. Try again shortly."
        )


# ---------------------------------------------------------------------------
# The lip-sync command itself -- shared by both uses above.
# ---------------------------------------------------------------------------


def build_lip_sync_argv(
    *,
    video_path: Path,
    audio_path: Path,
    audio_start_time: float,
    output_path: Path,
    work_dir: Path,
    seed: int,
    audio_cfg_scale: float | None = None,
    ref_cfg_scale: float | None = None,
) -> list[str]:
    """The complete argv (interpreter first) for one run_lipsync.py run."""
    argv = [
        str(config.TBDUB_PYTHON), str(config.RUN_LIPSYNC_SCRIPT),
        "--video", str(video_path),
        "--audio", str(audio_path),
        "--audio-start-time", str(audio_start_time),
        "--output-path", str(output_path),
        "--work-dir", str(work_dir),
        "--tbdub-repo", str(config.TBDUB_REPO),
        "--checkpoint-dir", str(config.WEIGHTS_DIR),
        "--seed", str(seed),
    ]
    if audio_cfg_scale is not None:
        argv += ["--audio-cfg-scale", str(audio_cfg_scale)]
    if ref_cfg_scale is not None:
        argv += ["--ref-cfg-scale", str(ref_cfg_scale)]
    return argv


def build_post_step(job_dir: Path, audio_path: Path, audio_start_time: float, seed: int) -> dict:
    """The `post_step.json` content that makes an LTX audio-to-video job run
    lip sync after its own step (see run_pipeline_job.py::_run_lip_sync_step).
    The un-synced LTX video is kept as BASE_VIDEO_NAME in the job folder."""
    base_path = job_dir / config.BASE_VIDEO_NAME
    result_path = job_dir / "output.lipsync.partial.mp4"
    return {
        "argv": build_lip_sync_argv(
            video_path=base_path,
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            output_path=result_path,
            work_dir=job_dir,
            seed=seed,
        ),
        "input_path": str(base_path),
        "output_path": str(result_path),
        "timeout_seconds": config.LIPSYNC_EXTRA_SECONDS,
    }


def post_step_for_audio_to_video(req, job_dir: Path) -> dict | None:
    """`build_post_step` for an LTX audio-to-video request (2.3 or 2.5), or
    None if it did not ask for lip sync. Raises KeyError/FileNotFoundError
    for an unknown audio asset, like the rest of request validation."""
    if not req.lip_sync:
        return None
    audio_path = storage.resolve_asset_path(req.audio_asset_id)
    return build_post_step(job_dir, audio_path, req.audio_start_time, req.seed)


# ---------------------------------------------------------------------------
# Resolving what to sync.
# ---------------------------------------------------------------------------


def _resolve_source_job(source_job_id: str) -> tuple[Path, Path, float]:
    """(video path, audio path, audio_start_time) of a succeeded LTX
    audio-to-video job, checked in a fixed order so each failure says
    what to do about it. Nothing is created or copied here."""
    row = job_store.get_job(source_job_id)
    if row is None:
        raise NotFoundError("Unknown source_job_id (it may have been deleted).")
    if row["pipeline"] not in config.SOURCE_PIPELINES:
        raise InputError(
            f"source_job_id must be an audio-to-video job ({', '.join(sorted(config.SOURCE_PIPELINES))}); "
            f"this one is {row['pipeline']}."
        )
    if row["status"] != "succeeded":
        raise ConflictError(f"The source job has not succeeded (status: {row['status']}).")
    if job_store.is_scrubbed(row):
        raise ConflictError(
            "The source job's data was removed (it was an incognito job), so it cannot be re-synced. "
            "Upload the video and audio instead."
        )

    job_dir = Path(row["job_dir"])
    if job_dir.resolve() != (common_config.JOBS_DIR / row["job_id"]).resolve():
        raise ConflictError("The source job's folder is not where it should be; refusing to read from it.")
    # Prefer the un-synced base video; a job that never asked for lip sync only has its result.
    video_path = job_dir / config.BASE_VIDEO_NAME
    if not video_path.is_file():
        video_path = Path(row["output_path"]) if row["output_path"] else video_path
    if not video_path.is_file():
        raise ConflictError("The source job's video is no longer on disk. Upload the video and audio instead.")

    request = json.loads(row["request_json"])
    try:
        audio_path = storage.resolve_asset_path(request["audio_asset_id"])
    except (KeyError, FileNotFoundError) as error:
        raise ConflictError(
            f"The source job's audio upload is no longer available ({error}). Upload the video and audio instead."
        ) from error
    return video_path, audio_path, float(request.get("audio_start_time") or 0.0)


# ---------------------------------------------------------------------------
# Submission plumbing for the standalone endpoint.
# ---------------------------------------------------------------------------


def _submit_slurm_job(job_id: str, job_dir: Path, argv: list[str], output_path: Path, partition: str, incognito: bool) -> str:
    """Write args.json + submit via sbatch (non-blocking). Returns the Slurm job id."""
    partial_path = job_dir / f"output.partial{output_path.suffix}"
    args = list(argv)
    args[args.index("--output-path") + 1] = str(partial_path)

    (job_dir / "args.json").write_text(json.dumps(args))
    (job_dir / "output_path.txt").write_text(str(output_path))

    wrap_argv = ["python3", str(config.RUN_PIPELINE_JOB_SCRIPT), str(job_dir)]
    wrap_str = "HF_HUB_OFFLINE=1 " + shlex.join(wrap_argv)
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        "--gres=gpu:1",
        f"--cpus-per-task={config.CPUS_PER_GPU}",
        f"--mem-per-gpu={config.MEM_PER_GPU}",
        f"--time={config.JOB_TIME_LIMIT}",
        f"--job-name={common_config.SLURM_JOB_NAME_PREFIX}tbdub-api-{job_id[:8]}",
        f"--output={retention.slurm_log_path(job_dir, incognito)}",
        f"--wrap={wrap_str}",
    ]
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DispatchError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip().split(";")[0]  # --parsable prints "<id>" or "<id>;<cluster>"


def dispatch_lip_sync(req: LipSyncRequest) -> str:
    """Validate capacity + inputs, then create the job directory/DB row/Slurm
    job, in that order, so an invalid request never leaves debris behind."""
    _check_capacity()
    partition = resolve_partition(req.partition)

    # Everything that can reject the request happens before anything is created.
    if req.source_job_id is not None:
        video_source, audio_source, audio_start_time = _resolve_source_job(req.source_job_id)
    else:
        video_source = storage.resolve_asset_path(req.video_asset_id)  # raises KeyError/FileNotFoundError
        audio_source = storage.resolve_asset_path(req.audio_asset_id)
        audio_start_time = req.audio_start_time or 0.0

    job_id = job_store.new_id()
    job_dir = common_config.JOBS_DIR / job_id
    output_path = job_dir / "output.mp4"
    job_dir.mkdir(parents=True)

    if req.source_job_id is not None:
        # Copy, don't link: this job must stay valid if the source job is purged.
        try:
            video_path = job_dir / f"source_video{video_source.suffix}"
            audio_path = job_dir / f"source_audio{audio_source.suffix}"
            shutil.copy2(video_source, video_path)
            shutil.copy2(audio_source, audio_path)
        except OSError as error:
            shutil.rmtree(job_dir, ignore_errors=True)  # the folder created just above, nothing else
            raise DispatchError(f"Could not copy the source job's files: {error}") from error
    else:
        video_path, audio_path = video_source, audio_source

    argv = build_lip_sync_argv(
        video_path=video_path,
        audio_path=audio_path,
        audio_start_time=audio_start_time,
        output_path=output_path,
        work_dir=job_dir,
        seed=req.seed,
        audio_cfg_scale=req.audio_cfg_scale,
        ref_cfg_scale=req.ref_cfg_scale,
    )
    job_store.create_job(job_id, "tbdub:lip-sync", job_dir, req.model_dump(mode="json"), partition, incognito=req.incognito)
    try:
        slurm_job_id = _submit_slurm_job(job_id, job_dir, argv, output_path, partition, req.incognito)
    except DispatchError:
        job_store.mark_failed(job_id, "Could not submit the Slurm job.")
        raise
    job_store.set_slurm_job_id(job_id, slurm_job_id)
    return job_id
