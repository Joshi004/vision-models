"""Slurm partition validation, shared by every backend's dispatch module.

Every generation request may optionally name a `partition` to submit its
job to. This is the one place that decides whether a requested partition is
actually real right now, and what to fall back to when it isn't (or when
none was given at all) -- see config.FALLBACK_PARTITION.
"""

from __future__ import annotations

import logging
import subprocess

from common import config

logger = logging.getLogger("model-api.slurm")


def cancel_slurm_job(slurm_job_id: str) -> None:
    """Best-effort `scancel` -- generic across every backend, since cancelling
    a Slurm job doesn't depend on which backend submitted it."""
    subprocess.run(["scancel", slurm_job_id], capture_output=True)


def list_partitions() -> set[str] | None:
    """Every partition name Slurm currently knows about, including hidden
    ones -- plain `sinfo` (without `-a`) hides partitions like this
    cluster's own `background`, which would otherwise make a perfectly
    valid, explicitly-requested `partition=background` look "invalid".

    Returns None if `sinfo` itself could not be run/parsed right now (a
    transient problem, not evidence that no partitions exist).
    """
    try:
        result = subprocess.run(
            ["sinfo", "-a", "-h", "-o", "%P"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("sinfo failed while validating a partition request: %s", e)
        return None
    if result.returncode != 0:
        logger.warning("sinfo exited %d: %s", result.returncode, result.stderr.strip())
        return None
    # Slurm marks the cluster's default partition with a trailing "*" (e.g.
    # "main*") -- strip it so comparisons match the plain partition name.
    return {line.strip().rstrip("*") for line in result.stdout.splitlines() if line.strip()}


def resolve_partition(requested: str | None) -> str:
    """The partition a job should actually submit to.

    - No `partition` given -> config.FALLBACK_PARTITION.
    - A `partition` given that Slurm currently recognizes -> that partition,
      unchanged.
    - A `partition` given that Slurm does NOT currently recognize ->
      config.FALLBACK_PARTITION, silently -- this is deliberately not an
      error (see the API's `partition` field docs).
    - Given but `sinfo` couldn't be checked right now -> pass it through
      unchanged rather than second-guessing a possibly-valid request;
      `sbatch` itself is the final authority and will reject it with a
      clear error if it really is bad.
    """
    if not requested:
        return config.FALLBACK_PARTITION
    valid = list_partitions()
    if valid is None:
        return requested
    if requested not in valid:
        logger.info(
            "Requested partition %r is not currently valid on this cluster; falling back to %r.",
            requested,
            config.FALLBACK_PARTITION,
        )
        return config.FALLBACK_PARTITION
    return requested
