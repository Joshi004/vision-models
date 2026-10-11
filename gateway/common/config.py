"""Shared configuration for model-api: settings and paths that are not
specific to any one backend pipeline (LTX-2.3, and whatever else joins it
later).

Per-backend configuration (model checkpoints, Slurm resource shape, project
paths) lives in services/<name>/config.py instead -- see that module's
docstring for why the split is there. This module owns everything the
job-tracking/upload machinery in this directory needs, since that
machinery is shared by every backend.

Zero third-party dependencies (stdlib only) so it can be imported both by
the FastAPI app (running in venv/) and by the small per-job wrapper script
that Slurm executes directly (common/run_pipeline_job.py), which should not
need fastapi/uvicorn installed at all.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
COMMON_DIR = Path(__file__).resolve().parent
MODEL_API_DIR = COMMON_DIR.parent

# The git repository root (the folder that holds gateway/, models/ and
# third_party/).
REPO_ROOT = MODEL_API_DIR.parent

# Everything that is NOT code lives outside the repository, under one data
# directory: built Python environments, model weights, jobs and the job
# database. Override with VISION_DATA_DIR; the default is a sibling folder of
# the repository. It must be on storage that every Slurm compute node can see.
DATA_DIR = Path(os.environ.get("VISION_DATA_DIR", str(REPO_ROOT.parent / "vision-data")))
ENVS_DIR = DATA_DIR / "envs"
WEIGHTS_DIR = DATA_DIR / "weights"

# Job workspace: one subdirectory per job_id (args, logs, and ultimately the
# output file itself), regardless of which backend created it -- job_store's
# schema already tracks jobs from any backend in one table, so this is a
# single shared tree rather than one per backend.
JOBS_DIR = DATA_DIR / "jobs"
UPLOADS_DIR = JOBS_DIR / "_uploads"

# Built web UI (model-api/ui/), if it's been built -- see server.py, which
# mounts this at /ui only when the directory actually exists, so the API
# still starts fine even before anyone's run `npm run build` there.
UI_DIST_DIR = MODEL_API_DIR / "ui" / "dist"

DB_PATH = DATA_DIR / "jobs.db"

# ---------------------------------------------------------------------------
# API server settings
# ---------------------------------------------------------------------------
API_HOST = os.environ.get("MODEL_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("MODEL_API_PORT", "8012"))
POLL_INTERVAL_SECONDS = float(os.environ.get("MODEL_API_POLL_INTERVAL", "5"))
JOB_RETENTION_DAYS = float(os.environ.get("MODEL_API_RETENTION_DAYS", "7"))

# Job data (logs, uploaded inputs, generated outputs, the request body of each
# job) is kept until JOB_RETENTION_DAYS passes or the job is purged -- unless
# the request that created the job set `incognito: true`, in which case it is
# kept no longer than needed. That choice is made per request, not here: see
# common/retention.py for the exact incognito rules.
#
# An upload that no job references at all is deleted this many minutes after
# it was uploaded (covers a file uploaded and then never used, a rejected
# submission, or the leftover of a job that has been purged or has expired).
UNUSED_UPLOAD_TTL_MINUTES = float(os.environ.get("MODEL_API_UNUSED_UPLOAD_TTL_MINUTES", "60"))
# Optional soft safety valve on concurrent Slurm jobs this service has in
# flight at once, across every backend combined (job_store's inflight count
# is global, not per-backend). Unset (default) = no cap.
_max_inflight = os.environ.get("MODEL_API_MAX_INFLIGHT")
MAX_INFLIGHT_JOBS = int(_max_inflight) if _max_inflight else None
MAX_UPLOAD_BYTES = int(os.environ.get("MODEL_API_MAX_UPLOAD_MB", "2048")) * 1024 * 1024

# ---------------------------------------------------------------------------
# Slurm partition fallback -- see common/slurm.py::resolve_partition. Every
# generation request may optionally name a `partition` field; if it names
# one that doesn't exist on the cluster right now (or names none at all),
# the job falls back to this, silently, for every backend alike.
# ---------------------------------------------------------------------------
FALLBACK_PARTITION = os.environ.get("MODEL_API_FALLBACK_PARTITION", "background")

# ---------------------------------------------------------------------------
# Slurm job naming -- prepended to every backend's own `--job-name` value
# (e.g. "ltx23-api-<id8>" -> "data-validation-ltx23-api-<id8>") at the point
# each backend's dispatch module actually builds its `sbatch` command, so
# every job this API submits is identifiable as this deployment's in
# `squeue`/`sacct` output alongside jobs from unrelated tools on the same
# shared cluster.
# ---------------------------------------------------------------------------
SLURM_JOB_NAME_PREFIX = os.environ.get("MODEL_API_SLURM_JOB_NAME_PREFIX", "data-validation-")


def ensure_directories() -> None:
    for d in (JOBS_DIR, UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)
