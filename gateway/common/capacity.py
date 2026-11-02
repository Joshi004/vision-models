"""GPU capacity limits that can be changed while the gateway runs.

Two kinds of limit, both stored in the job database (the `settings` table) and edited from the
web UI (GET/PUT /v1/capacity), not from environment variables:

  * global_gpus: the most GPUs this service may hold at once, across every backend.
  * max_jobs[model]: the most jobs of one model that run at once. For a warm-worker backend
    (common/worker_pool.py) one worker runs one job at a time, so this is also the most warm
    workers it keeps; for a backend that runs one Slurm job per request it is the most queued or
    running jobs.

Lowering a limit never cancels a job that is running. The pools stop starting workers while they
are at or over a limit, stop their idle (and still-starting) workers that are over it, and the
busy ones simply finish; new work starts again once usage is back under the limit. Raising a
limit lets queued requests start on the next scheduling pass (PUT calls
worker_pool.pools_apply_now() so that pass happens at once).

Validation, "sum rule": the limits of all models, each multiplied by that model's planning GPUs
per job (4 for H3, 1 for every other model), plus the GPUs of the fixed Qwen3.8 server, must not
add up to more than global_gpus. The runtime check uses the GPUs a job really holds, which for H3
can be 8.

Locking: budget_lock guards every "is there room, then take it" step. A pool takes its own lock
first and then budget_lock; nothing here ever takes a pool's lock, it only reads the plain GPU
count each pool publishes, so the two cannot deadlock.

Stdlib only, like the rest of common/.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from common import job_store

logger = logging.getLogger("model-api.capacity")

POOL = "pool"  # a warm-worker backend (common/worker_pool.py)
ONESHOT = "oneshot"  # one Slurm job per request, 1 GPU each
FIXED = "fixed"  # always the same GPUs while up, no limit of its own (the Qwen3.8 server)

_SETTING_KEY = "capacity"


class CapacityError(RuntimeError):
    """The GPU capacity limit (global or the model's own) leaves no room for this request."""


class InvalidSettings(ValueError):
    """The submitted limits break a rule; the message says which and why."""


@dataclass
class _Model:
    key: str
    label: str
    kind: str
    planning_gpus: int
    default_max_jobs: int
    pipeline_prefixes: tuple[str, ...]  # one-shot backends: which jobs-table pipelines are theirs
    gpus_held: Callable[[], int] | None  # pools and fixed models: GPUs held right now
    snapshot: Callable[[], dict] | None  # pools: {gpus_held, busy, idle, starting, queued}


_MODELS: dict[str, _Model] = {}
budget_lock = threading.RLock()
_reserved = 0  # GPUs promised to a worker that is being submitted but is not in its pool yet

_settings_lock = threading.Lock()
_settings: dict | None = None  # {"global_gpus": int, "max_jobs": {key: int}}, loaded on first use


def register(
    key: str,
    label: str,
    kind: str,
    *,
    planning_gpus: int = 1,
    default_max_jobs: int = 4,
    pipeline_prefixes: tuple[str, ...] = (),
    gpus_held: Callable[[], int] | None = None,
    snapshot: Callable[[], dict] | None = None,
) -> None:
    """Make a model known to the capacity limits. Called at import time by each backend."""
    _MODELS[key] = _Model(
        key=key,
        label=label,
        kind=kind,
        planning_gpus=planning_gpus,
        default_max_jobs=default_max_jobs,
        pipeline_prefixes=pipeline_prefixes,
        gpus_held=gpus_held,
        snapshot=snapshot,
    )


def _limited_models() -> list[_Model]:
    return [m for m in _MODELS.values() if m.kind != FIXED]


def _planned_gpus(max_jobs: dict[str, int]) -> int:
    planned = sum(max_jobs[m.key] * m.planning_gpus for m in _limited_models())
    return planned + sum(m.planning_gpus for m in _MODELS.values() if m.kind == FIXED)


# --- the stored settings -------------------------------------------------------


def _load_locked() -> dict:
    """The settings, read from the database on first use (call with _settings_lock held). The
    first time ever, every model gets the limit it had before limits were configurable and the
    global limit is their sum, so the stored settings are valid; a model added later gets 1."""
    global _settings
    if _settings is not None:
        return _settings
    saved: dict | None = None
    raw = job_store.get_setting(_SETTING_KEY)
    if raw:
        try:
            parsed = json.loads(raw)
            saved = parsed if isinstance(parsed, dict) else None
        except ValueError:
            logger.warning("The stored capacity settings are not valid JSON; starting from the defaults.")
    if saved is None:
        max_jobs = {m.key: m.default_max_jobs for m in _limited_models()}
        settings = {"global_gpus": _planned_gpus(max_jobs), "max_jobs": max_jobs}
        job_store.set_setting(_SETTING_KEY, json.dumps(settings))
    else:
        stored_jobs = saved.get("max_jobs") if isinstance(saved.get("max_jobs"), dict) else {}
        max_jobs = {m.key: max(1, int(stored_jobs.get(m.key, 1))) for m in _limited_models()}
        settings = {"global_gpus": max(1, int(saved.get("global_gpus", _planned_gpus(max_jobs)))), "max_jobs": max_jobs}
    _settings = settings
    return settings


def global_gpus() -> int:
    with _settings_lock:
        return _load_locked()["global_gpus"]


def max_jobs(key: str) -> int:
    """The limit of one model; 1 for a model that has none."""
    with _settings_lock:
        return _load_locked()["max_jobs"].get(key, 1)


def _validated(new_global: int, new_max_jobs: dict[str, int], current: dict) -> dict:
    keys = {m.key for m in _limited_models()}
    unknown = sorted(set(new_max_jobs) - keys)
    if unknown:
        raise InvalidSettings(f"Unknown model(s): {', '.join(unknown)}.")
    if new_global < 1:
        raise InvalidSettings("The global GPU limit must be 1 or more.")
    merged = dict(current["max_jobs"])
    merged.update(new_max_jobs)
    for key, value in merged.items():
        if value < 1:
            raise InvalidSettings(f"The limit for {_MODELS[key].label} must be 1 or more.")
    planned = _planned_gpus(merged)
    if planned > new_global:
        heavy = ", ".join(f"{m.label} counts {m.planning_gpus} GPUs per job" for m in _limited_models() if m.planning_gpus > 1)
        raise InvalidSettings(
            f"The per-model limits add up to {planned} GPUs{f' ({heavy})' if heavy else ''}, which is more than the "
            f"global limit of {new_global}. Lower a model's limit or raise the global limit."
        )
    return {"global_gpus": new_global, "max_jobs": merged}


def update(new_global: int, new_max_jobs: dict[str, int]) -> None:
    """Validate and store new limits (max_jobs may name only the models that change). Raises
    InvalidSettings, changing nothing, if a rule is broken."""
    global _settings
    with _settings_lock:
        validated = _validated(new_global, new_max_jobs, _load_locked())
        job_store.set_setting(_SETTING_KEY, json.dumps(validated))
        _settings = validated
    logger.info("Capacity limits changed: %s", validated)


# --- GPUs in use -------------------------------------------------------------------


def _oneshot_jobs(model: _Model, counts: dict[tuple[str, str], int]) -> tuple[int, int]:
    """(running, queued) jobs of a one-shot backend."""
    running = queued = 0
    for (pipeline, status), n in counts.items():
        if pipeline.startswith(model.pipeline_prefixes):
            if status == "running":
                running += n
            else:
                queued += n
    return running, queued


def _gpus_in_use(counts: dict[tuple[str, str], int] | None = None) -> int:
    total = 0
    for model in list(_MODELS.values()):
        if model.kind == ONESHOT:
            if counts is None:
                counts = job_store.inflight_counts_by_pipeline()
            total += sum(_oneshot_jobs(model, counts)) * model.planning_gpus
        elif model.gpus_held is not None:
            total += model.gpus_held()
    return total


def gpus_in_use() -> int:
    """GPUs held by this service right now: warm workers (starting, idle or busy), queued or
    running one-shot jobs, and the Qwen3.8 server while it is up."""
    return _gpus_in_use()


def try_reserve(gpus: int) -> bool:
    """Take `gpus` from the global budget for a worker about to be submitted, if they fit. The
    caller submits the worker and then calls release(gpus): by then the worker is counted in its
    pool, so the reservation is only there to cover the sbatch call itself."""
    global _reserved
    limit = global_gpus()
    with budget_lock:
        if _gpus_in_use() + _reserved + gpus > limit:
            return False
        _reserved += gpus
        return True


def release(gpus: int) -> None:
    global _reserved
    with budget_lock:
        _reserved = max(0, _reserved - gpus)


# --- one-shot backends ----------------------------------------------------------------


def _check_oneshot_locked(key: str) -> None:
    model = _MODELS[key]
    counts = job_store.inflight_counts_by_pipeline()
    own = sum(_oneshot_jobs(model, counts))
    limit = max_jobs(key)
    if own >= limit:
        raise CapacityError(
            f"{model.label} already has {own} job(s) queued or running (limit {limit}). Try again shortly."
        )
    in_use = _gpus_in_use(counts)
    total = global_gpus()
    if in_use + _reserved + model.planning_gpus > total:
        raise CapacityError(f"GPU capacity limit reached: {in_use} of {total} GPU(s) in use. Try again shortly.")


def check_oneshot(key: str) -> None:
    """Raise CapacityError if one more job of this one-shot backend would pass its own limit or
    the global one. A first, cheap refusal before any work is done; oneshot_admission repeats the
    check at the moment the job row is written."""
    with budget_lock:
        _check_oneshot_locked(key)


@contextmanager
def oneshot_admission(key: str, job_dir: Path | None = None) -> Iterator[None]:
    """Wrap the job-store insert of a one-shot job: the limits are checked again and the insert
    happens before any other submission can be checked, so two requests arriving together cannot
    both take the last slot. If there is no room after all, `job_dir` (already created by the
    caller) is removed and CapacityError raised."""
    with budget_lock:
        try:
            _check_oneshot_locked(key)
        except CapacityError:
            if job_dir is not None:
                shutil.rmtree(job_dir, ignore_errors=True)
            raise
        yield


# --- what the API reports -------------------------------------------------------------------


def status() -> dict:
    """The body of GET /v1/capacity: the limits, GPUs in use, and one row per model."""
    counts = job_store.inflight_counts_by_pipeline()
    with _settings_lock:
        settings = _load_locked()
        limits = dict(settings["max_jobs"])
        total = settings["global_gpus"]
    rows = []
    for model in list(_MODELS.values()):
        row = {
            "key": model.key,
            "label": model.label,
            "kind": model.kind,
            "max_jobs": limits.get(model.key),
            "planning_gpus": model.planning_gpus,
            "gpus_held": 0,
            "busy": 0,
            "idle": 0,
            "starting": 0,
            "queued": 0,
        }
        if model.kind == ONESHOT:
            running, queued = _oneshot_jobs(model, counts)
            row.update(gpus_held=(running + queued) * model.planning_gpus, busy=running, queued=queued)
        elif model.snapshot is not None:
            row.update(model.snapshot())
        elif model.gpus_held is not None:
            held = model.gpus_held()
            row.update(gpus_held=held, busy=held)
        rows.append(row)
    return {
        "global_gpus": total,
        "gpus_in_use": sum(row["gpus_held"] for row in rows),
        "planned_gpus": _planned_gpus(limits),
        "models": rows,
    }
