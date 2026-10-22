"""Numbers-only usage log for MiniMax H3 jobs, for calibrating services/h3/sizing.py.

When an H3 job finishes (the poller calls `record`), one JSON line is appended to
DATA_DIR/metrics/h3_jobs.jsonl: what the gateway decided (sizing.json), what the job
measured (the runner's metrics.<role>.json files: stage timings, peak GPU memory per GPU,
token and row counts, the failed stage and the out-of-memory sizes), the Slurm
accounting (node, elapsed, peak host RAM) and the queue/run times from jobs.db.

Only numbers, stage names and enums go in: never the prompt, a file name, an asset id or an
error message. A job submitted with `incognito: true` is recorded the same way, except
that its job id is left out. The log lives outside jobs/, so retention never deletes it.

Every function that the poller calls is best effort: a problem here is logged and must
never disturb job handling. The poller is the only writer while the gateway runs;
scripts/h3_usage_log.py adds records of older jobs (`backfill`) and flattens the log
into a spreadsheet (`csv`).

A record has:
  job header   job_id (null if incognito), slurm_job_id, pipeline, outcome, incognito, partition,
               gpus, estimated_run_seconds, created_at/started_at/finished_at, queue_wait_s, run_s
  sizing       sizing.json as written at submission (absent for jobs from before it existed)
  metrics      {"run": ..., "condition": ..., "worker1": ...}: the runner's files (absent for older jobs)
  setup        the request's shape read from args.json, only when there is no sizing
  parsed_log   timings read from run.log, only when there are no metrics (older jobs)
  slurm        state, node, elapsed_s, max_rss_gib from sacct (when it answers)
  worker_id,   only for a request that ran on a warm worker (services/h3/dispatch.py): the worker, and whether
  warm         it had already run a request (so the checkpoints were in the node's page cache). Its `slurm`
               is then this request's run time (elapsed_s, from jobs.db) and node, and max_rss_gib is null,
               because sacct's figures would be the worker's, not the request's.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import config as common_config
from common import job_store

logger = logging.getLogger("model-api.h3-usage-log")

SCHEMA = 1
METRICS_DIR = common_config.DATA_DIR / "metrics"
LOG_PATH = METRICS_DIR / "h3_jobs.jsonl"

_MAX_WORKERS = 7  # a job has at most 8 GPUs, so workers 1..7

# Written by common/worker_pool.py into the folder of a request that ran on a warm worker.
POOL_ASSIGNMENT_FILE = "pool_assignment.json"


# ---------------------------------------------------------------------------
# Reading what a job left behind.
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_json_list(path: Path) -> list[str] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, list) else None


def _read_metrics(job_dir: Path, gpus: int | None) -> dict:
    """The runner's metrics.<role>.json files. Looked up by name, not by listing the folder: the folder is on a
    shared filesystem where a listing can lag behind files another node just wrote."""
    roles = ["run", "condition", *(f"worker{rank}" for rank in range(1, (gpus or _MAX_WORKERS + 1)))]
    metrics = {}
    for role in roles:
        data = _read_json(job_dir / f"metrics.{role}.json")
        if data is not None:
            metrics[role] = data
    return metrics


_KIB_PER_GIB = 1024 * 1024
_RSS_UNITS = {"K": 1 / _KIB_PER_GIB, "M": 1 / 1024, "G": 1.0, "T": 1024.0}


def _rss_gib(text: str) -> float | None:
    """sacct's MaxRSS ('123456K', '9G', ...) in GiB."""
    found = re.fullmatch(r"([\d.]+)([KMGT])", text.strip())
    return None if found is None else float(found.group(1)) * _RSS_UNITS[found.group(2)]


def _read_slurm(slurm_job_id: str | None) -> dict | None:
    """State, node, elapsed seconds and peak host RAM from one sacct call, or None if it cannot answer."""
    if not slurm_job_id:
        return None
    try:
        result = subprocess.run(
            ["sacct", "-j", slurm_job_id, "-n", "-P", "-o", "JobID,State,ElapsedRaw,NodeList,MaxRSS"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    slurm: dict = {}
    peaks = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        job_id, state, elapsed, node, max_rss = parts
        if job_id == slurm_job_id:
            slurm["state"] = state.split()[0] if state else None  # "CANCELLED by 1234" -> "CANCELLED"
            slurm["elapsed_s"] = int(elapsed) if elapsed.isdigit() else None
            slurm["node"] = node or None
        rss = _rss_gib(max_rss)
        if rss is not None:
            peaks.append(rss)
    if not slurm:
        return None
    slurm["max_rss_gib"] = round(max(peaks), 1) if peaks else None
    return slurm


def _seconds_between(start_iso: str | None, end_iso: str | None) -> float | None:
    if not start_iso or not end_iso:
        return None
    return round((datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)).total_seconds(), 1)


# ---------------------------------------------------------------------------
# Older jobs: no sizing.json and no metrics files, only args.json and run.log.
# ---------------------------------------------------------------------------


def _flag_values(argv: list[str], flag: str) -> list[str]:
    return [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == flag]


def _setup_from_args(job_dir: Path) -> dict | None:
    """The request's shape from args.json (the runner's command line): numbers and enums only."""
    argv = _read_json_list(job_dir / "args.json")
    if argv is None:
        return None

    def number(flag: str):
        values = _flag_values(argv, flag)
        return int(values[0]) if values else None

    prompt = next((value for value in argv if value.startswith("--prompt=")), "--prompt=")
    backend = _flag_values(argv, "--attention-backend")
    parallelism = _flag_values(argv, "--parallelism")
    workflow = _flag_values(argv, "--workflow")
    return {
        "workflow": workflow[0] if workflow else None,
        "parallelism": parallelism[0] if parallelism else None,
        "attention_backend": backend[0] if backend else None,
        "steps": number("--steps"),
        "num_frames": number("--num-frames"),
        "height": number("--height"),
        "width": number("--width"),
        "prompt_chars": len(prompt) - len("--prompt="),
        "match_audio_duration": "--match-audio-duration" in argv,
        "requested_references": {
            "images": len(_flag_values(argv, "--reference-image")),
            "videos": len(_flag_values(argv, "--reference-video")),
            "audios": len(_flag_values(argv, "--reference-audio")),
            "muted_videos": len(_flag_values(argv, "--mute-reference-video")),
            "keyframes": len(_flag_values(argv, "--first-frame")) + len(_flag_values(argv, "--last-frame")),
        },
    }


# What the runner printed before it wrote metrics files: [  718s run] ... lines. Each pattern's groups are numbers.
_LOG_PATTERNS = {
    "set_up_s": r"\[ *\d+s run\] set up in (\d+)s",
    "load_transformer_s": r"\[ *\d+s run\] (?:transformer and VAEs on the GPU|loaded) in (\d+)s",
    "encode_s": r"\[ *\d+s condition\] encoded in (\d+)s",
    "waited_for_conditioning_s": r"conditioning ready \(waited (\d+)s",
    "encoder_peak_gib": r"\[ *\d+s condition\] peak GPU memory ([\d.]+) GiB",
    "run_peak_gib": r"\[ *\d+s run\] peak GPU memory ([\d.]+) GiB",
    "total_s": r"\(total (\d+)s\)",
    "audio_seconds": r"--match-audio-duration: ([\d.]+) s of audio",
}
_GENERATED = r"\[ *\d+s run\] generated (\d+) frames of (\d+)x(\d+) in (\d+)s"
_SPLIT_PEAKS = r"peak GPU memory since the split: generation GPU ([\d.]+) GiB, second GPU ([\d.]+) GiB"
_REFERENCES = r"references: (\d+) image\(s\), (\d+) video\(s\) \((\d+) with a soundtrack\), (\d+) audio file\(s\)"

# The same patterns as run_h3.py::oom_numbers (the gateway cannot import that file).
_SIZE = r"([\d.]+) (MiB|GiB)"
_OOM_PATTERNS = {
    "tried_mib": r"Tried to allocate " + _SIZE,
    "total_mib": r"total capacity of " + _SIZE,
    "free_mib": r"of which " + _SIZE + r" is free",
    "in_use_mib": r"this process has " + _SIZE + r" memory in use",
    "allocated_mib": _SIZE + r" is allocated by PyTorch",
}


def oom_numbers(message: str) -> dict:
    """The figures in a CUDA out-of-memory message, in MiB (plus which GPU), or {} for any other text."""
    if "out of memory" not in message:
        return {}
    numbers: dict = {}
    for key, pattern in _OOM_PATTERNS.items():
        found = re.search(pattern, message)
        if found:
            numbers[key] = round(float(found.group(1)) * (1024 if found.group(2) == "GiB" else 1), 1)
    gpu = re.search(r"GPU (\d+) has a total capacity", message)
    if gpu:
        numbers["gpu"] = int(gpu.group(1))
    return numbers


def _parse_run_log(job_dir: Path) -> dict | None:
    """Timings, peaks and (if the job ran out of GPU memory) the OOM sizes from run.log -- by pattern, so nothing
    but numbers is ever kept from it."""
    try:
        text = (job_dir / "run.log").read_text(errors="replace")
    except OSError:
        return None
    parsed: dict = {}
    for key, pattern in _LOG_PATTERNS.items():
        found = re.search(pattern, text)
        if found:
            parsed[key] = float(found.group(1))
    found = re.search(_GENERATED, text)
    if found:
        frames, width, height, seconds = (int(value) for value in found.groups())
        parsed.update(generated_frames=frames, width=width, height=height, denoise_s=seconds)
    found = re.search(_SPLIT_PEAKS, text)
    if found:
        parsed["split_peak_gib"] = [float(found.group(1)), float(found.group(2))]
    found = re.search(_REFERENCES, text)
    if found:
        parsed["references_seen"] = dict(zip(("images", "videos", "videos_with_sound", "audios"), (int(v) for v in found.groups())))
    for name, pattern in (("parallelism", r"\] parallelism: (split|ulysses)"), ("attention_backend", r"attention backend: (\S+)")):
        found = re.search(pattern, text)
        if found:
            parsed[name] = found.group(1)
    oom_line = next((line for line in text.splitlines() if "out of memory" in line), "")
    if oom_line:
        parsed["oom"] = oom_numbers(oom_line)
    return parsed


# ---------------------------------------------------------------------------
# The record.
# ---------------------------------------------------------------------------


def build_record(row: sqlite3.Row, source: str = "live") -> dict:
    """One usage-log record for a finished H3 job row. `source` is "live" (written by the poller when the job
    ended) or "backfill" (built later from what the job folder still holds)."""
    job_dir = Path(row["job_dir"])
    incognito = bool(row["incognito"])
    # A database from before these columns existed (not yet migrated by a gateway start) has no gpus/estimate.
    gpus = row["gpus"] if "gpus" in row.keys() else None
    estimated_run_seconds = row["estimated_run_seconds"] if "estimated_run_seconds" in row.keys() else None
    record: dict = {
        "schema": SCHEMA,
        "source": source,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "job_id": None if incognito else row["job_id"],
        "slurm_job_id": row["slurm_job_id"],
        "pipeline": row["pipeline"],
        "outcome": row["status"],
        "incognito": incognito,
        "partition": row["partition"],
        "gpus": gpus,
        "estimated_run_seconds": estimated_run_seconds,
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "queue_wait_s": _seconds_between(row["created_at"], row["started_at"]),
        "run_s": _seconds_between(row["started_at"], row["finished_at"]),
    }
    sizing = _read_json(job_dir / "sizing.json")
    metrics = _read_metrics(job_dir, gpus)
    if sizing is not None:
        record["sizing"] = sizing
    if metrics:
        record["metrics"] = metrics
    else:
        record["parsed_log"] = _parse_run_log(job_dir)
    if sizing is None:
        record["setup"] = _setup_from_args(job_dir)
    assignment = _read_json(job_dir / POOL_ASSIGNMENT_FILE)
    if assignment is None:
        record["slurm"] = _read_slurm(row["slurm_job_id"])
    else:
        # A warm worker's Slurm job runs many requests, so sacct's elapsed time and peak RAM belong to
        # the worker, not to this request: the time is this request's run time instead, and there is no RAM figure.
        record["worker_id"] = assignment.get("worker_id")
        record["warm"] = assignment.get("warm")
        run_metrics = metrics.get("run") or {}
        record["slurm"] = {
            "state": None,
            "elapsed_s": None if record["run_s"] is None else int(record["run_s"]),
            "node": run_metrics.get("node"),
            "max_rss_gib": None,
        }
    return record


def append(record: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, separators=(",", ":")) + "\n")


def record(job_id: str) -> None:
    """Append the record of this just-finished H3 job. Called by the poller right after it marks the job
    succeeded or failed, and BEFORE the incognito cleanup deletes the job folder. Never raises."""
    try:
        row = job_store.get_job(job_id)
        if row is not None and row["pipeline"].startswith("h3:"):
            append(build_record(row))
    except Exception:
        logger.exception("usage log: could not record job %s", job_id)


def _record_key(slurm_job_id, worker_id, job_id) -> str | None:
    """What identifies a job in the log: its own Slurm job id, or for a request that ran on a warm worker
    (whose Slurm job is shared by many requests) its job id."""
    if worker_id:
        return f"job:{job_id}" if job_id else None
    return str(slurm_job_id) if slurm_job_id else None


def logged_job_keys() -> set[str]:
    """The keys (see _record_key) of the jobs already in the log, so a backfill never writes a job twice."""
    keys: set[str] = set()
    try:
        with LOG_PATH.open(encoding="utf-8") as log_file:
            for line in log_file:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                key = _record_key(entry.get("slurm_job_id"), entry.get("worker_id"), entry.get("job_id"))
                if key:
                    keys.add(key)
    except OSError:
        pass
    return keys


def job_key(row: sqlite3.Row) -> str | None:
    """The key of a finished job row, as _record_key makes it for the log."""
    pooled = (Path(row["job_dir"]) / POOL_ASSIGNMENT_FILE).exists()
    return _record_key(row["slurm_job_id"], pooled, row["job_id"])


def flatten(value, prefix: str = "") -> dict:
    """A nested record as one flat {"a.b.c": number} row for a spreadsheet. A list of numbers becomes one column per
    position (a per-GPU peak); a list of references becomes one set of columns per entry."""
    flat: dict = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flat.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flat.update(flatten(item, f"{prefix}.{index}"))
    else:
        flat[prefix] = value
    return flat
