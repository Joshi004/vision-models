"""Live Slurm partition availability -- read-only, cached snapshot backing
GET /v1/partitions (see server.py), so a caller can see roughly how busy
each partition is *before* choosing one for a generation request, instead
of guessing.

Every number here comes straight from the Slurm controller (`scontrol`,
`squeue`), refreshed at most once every _CACHE_SECONDS -- never from the
accounting database (`sacct`), which this cluster has been observed to
drop out independently of the controller (see common/poller.py's own
`sacct` use, which already tolerates exactly that). A snapshot a few
seconds stale beats hammering the controller on every page load, or every
30s UI poll, across however many browser tabs happen to be open at once.

This is a snapshot, not a promise: a partition reporting free GPUs can
still take a moment to actually start a job (a higher-priority job may
already be queued for the same nodes), and every `background` job can be
preempted at any time and restart from scratch -- see
common/run_pipeline_job.py's own _PREEMPTION_EXIT_CODES handling. The UI
surfaces both caveats next to these numbers rather than presenting them as
a guarantee.

Deliberately excludes this cluster's own `hidden` partition (a plain,
undocumented duplicate of `main` -- see slurm.conf) from _KNOWN_PARTITIONS:
it isn't meant to be chosen through this API. common/slurm.py's own
list_partitions()/resolve_partition() are unaffected by that and keep
validating every partition Slurm knows about, `hidden` included, since a
caller could still ask for it by name even though it's never listed here.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from datetime import datetime, timezone

from common import config

logger = logging.getLogger("model-api.partition_status")

# Partitions actually offered to callers, in display order -- whichever one
# is currently config.FALLBACK_PARTITION is moved to the front of this at
# read time (see _ordered_partition_names) so the option a request gets
# without naming one explicitly is always the most prominent row.
_KNOWN_PARTITIONS: tuple[str, ...] = ("background", "main", "health", "toolCall", "VLM")

_CACHE_SECONDS = 20.0

# Substrings of a node's own State= that mean "don't count this node's GPUs
# as available" -- deliberately substring-matched, since Slurm freely
# combines states with "+" (e.g. "MIXED+CLOUD+PLANNED", observed live on
# this cluster; none of those extra flags belong on this list).
_NODE_STATE_UNUSABLE = ("DOWN", "DRAIN", "FAIL", "NOT_RESPONDING", "MAINT", "POWER")

# PENDING reasons that mean "not actually waiting on GPU availability" --
# skipped when counting how many jobs are competing for a partition's GPUs,
# since counting these would overstate how busy a partition looks (e.g. a
# job someone held by hand stays PENDING forever, regardless of how free
# the partition actually is).
_NON_COMPETING_REASONS = frozenset({"JobHeldUser", "JobHeldAdmin", "Dependency", "DependencyNeverSatisfied"})

_KV_RE = re.compile(r"(\w+)=(\S*)")


class PartitionStatusError(RuntimeError):
    """Slurm couldn't be queried right now (controller unreachable, a
    command timed out, or its output didn't parse) -- distinct from an
    ordinary empty result, which is a valid answer."""


def _run(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (subprocess.SubprocessError, OSError) as e:
        raise PartitionStatusError(f"{cmd[0]} failed: {e}") from e
    if result.returncode != 0:
        raise PartitionStatusError(f"{cmd[0]} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def _parse_kv_line(line: str) -> dict[str, str]:
    """"NodeName=foo State=MIXED Gres=..." -> {"NodeName": "foo", ...} --
    shared shape of one line from `scontrol ... -o` (node or partition
    alike)."""
    return dict(_KV_RE.findall(line))


def _gpu_count(tres: str) -> int:
    """Parses a comma-separated TRES string (e.g.
    "cpu=8,mem=64G,gres/gpu=1") -> 1 -- the same shape is used by
    CfgTRES/AllocTRES (scontrol show node) and tres-alloc (squeue). 0 if
    there's no gres/gpu key at all (no GPU involved)."""
    for part in tres.split(","):
        key, _, value = part.partition("=")
        if key == "gres/gpu" and value:
            return int(value)
    return 0


def _expand_hostlist(hostlist: str) -> list[str]:
    """Slurm hostlist syntax -> individual node names.

    Every *running* job on this cluster today requests exactly one node
    (see every backend's own sbatch_cmd: always --gres=gpu:1), so a job's
    own NodeList is always a single bare name with no bracket syntax --
    handled here with no subprocess call at all, since that's by far the
    most frequent case (one call per poll per running preemptible job,
    otherwise). Only a real bracketed range (e.g. a partition's own wide
    Nodes= field, "vlm-[0-49]") is handed to `scontrol show hostnames`,
    Slurm's own expansion, rather than re-implementing range parsing here.
    """
    if not hostlist or hostlist in ("(null)", "None"):
        return []
    if "[" not in hostlist:
        return [name for name in hostlist.split(",") if name]
    return [line for line in _run(["scontrol", "show", "hostnames", hostlist]).splitlines() if line]


def _read_nodes() -> dict[str, dict]:
    """Every node Slurm knows about, keyed by node name, each with
    {usable, total_gpus, free_gpus}. A node in a bad state (down/draining/
    failed/...) reports free_gpus=0 regardless of what AllocTRES says --
    Slurm wouldn't schedule onto it anyway."""
    nodes: dict[str, dict] = {}
    for line in _run(["scontrol", "-a", "show", "node", "-o"]).splitlines():
        if not line.strip():
            continue
        fields = _parse_kv_line(line)
        name = fields.get("NodeName")
        if not name:
            continue
        usable = not any(bad in fields.get("State", "") for bad in _NODE_STATE_UNUSABLE)
        total_gpus = _gpu_count(fields.get("CfgTRES", ""))
        alloc_gpus = _gpu_count(fields.get("AllocTRES", ""))
        nodes[name] = {
            "usable": usable,
            "total_gpus": total_gpus,
            "free_gpus": max(total_gpus - alloc_gpus, 0) if usable else 0,
        }
    return nodes


def _read_partitions() -> dict[str, dict]:
    """Every partition Slurm knows about (including hidden ones -- `-a`),
    keyed by partition name, each with {priority_tier, state, node_names}."""
    partitions: dict[str, dict] = {}
    for line in _run(["scontrol", "-a", "show", "partition", "-o"]).splitlines():
        if not line.strip():
            continue
        fields = _parse_kv_line(line)
        name = fields.get("PartitionName")
        if not name:
            continue
        partitions[name] = {
            "priority_tier": int(fields.get("PriorityTier", "1")),
            "state": fields.get("State", "UNKNOWN"),
            "node_names": set(_expand_hostlist(fields.get("Nodes", ""))),
        }
    return partitions


def _read_reclaimable_gpus_by_node(preemptible_partitions: list[str]) -> dict[str, int]:
    """GPUs currently held by *running* jobs in any of `preemptible_partitions`
    (i.e. `background` today), per node -- these are the GPUs a job
    submitted to a higher-tier partition could reclaim by preempting them.
    Empty dict (not an error) if none are running right now, or if there's
    nothing preemptible to check."""
    usage: dict[str, int] = {}
    if not preemptible_partitions:
        return usage
    out = _run(
        [
            "squeue",
            "-a",
            "-h",
            "-t",
            "R",
            "-p",
            ",".join(preemptible_partitions),
            "-O",
            "Partition:40,NumNodes:10,tres-alloc:300,NodeList:2000",
        ]
    )
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        tres, node_list = parts[-2], parts[-1]
        gpus = _gpu_count(tres)
        if gpus <= 0:
            continue
        nodes = _expand_hostlist(node_list)
        if not nodes:
            continue
        # Split evenly across every node the job spans -- exact for the
        # single-node jobs every backend here actually submits, an
        # approximation only for a hypothetical multi-node job (none seen
        # on this cluster so far).
        share, remainder = divmod(gpus, len(nodes))
        for i, node in enumerate(nodes):
            usage[node] = usage.get(node, 0) + share + (1 if i < remainder else 0)
    return usage


def _read_waiting_counts() -> dict[str, int]:
    """Number of jobs actually competing for GPUs in each partition right
    now -- PENDING, excluding _NON_COMPETING_REASONS (see that name's own
    comment). A job pending against several partitions at once (Slurm
    prints this as e.g. "VLM,background") counts toward each one named."""
    counts: dict[str, int] = {}
    out = _run(["squeue", "-a", "-h", "-t", "PD", "-O", "Partition:200,Reason:60"])
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        partition_field, reason = parts[0], parts[1]
        if reason in _NON_COMPETING_REASONS:
            continue
        for name in partition_field.split(","):
            counts[name] = counts.get(name, 0) + 1
    return counts


def _ordered_partition_names() -> list[str]:
    """_KNOWN_PARTITIONS, with whichever one is currently
    config.FALLBACK_PARTITION moved to the front -- so the option a
    request actually gets when it omits `partition` is always the first,
    most prominent row, even if a deployment's own
    MODEL_API_FALLBACK_PARTITION override ever names something other than
    'background'."""
    default = config.FALLBACK_PARTITION
    names = list(_KNOWN_PARTITIONS)
    if default in names:
        names.remove(default)
    if default:
        names.insert(0, default)
    return names


def _build_snapshot() -> list[dict]:
    """One fresh reading of every field GET /v1/partitions reports, for
    every name _ordered_partition_names() lists -- see PartitionStatus in
    common/schemas.py for what each field means from the API's side."""
    nodes = _read_nodes()
    partitions = _read_partitions()

    max_tier = max((p["priority_tier"] for p in partitions.values()), default=0)
    preemptible_names = [name for name, p in partitions.items() if p["priority_tier"] < max_tier]
    reclaimable_by_node = _read_reclaimable_gpus_by_node(preemptible_names)
    waiting_by_partition = _read_waiting_counts()
    default_partition = config.FALLBACK_PARTITION

    results: list[dict] = []
    for name in _ordered_partition_names():
        info = partitions.get(name)
        if info is None:
            logger.warning("Partition %r is not currently known to Slurm; omitting it from /v1/partitions.", name)
            continue
        member_nodes = [nodes[n] for n in info["node_names"] if n in nodes]
        preemptible = info["priority_tier"] < max_tier
        reclaimable_gpus = None
        if not preemptible:
            reclaimable_gpus = sum(reclaimable_by_node.get(node_name, 0) for node_name in info["node_names"])
        results.append(
            {
                "name": name,
                "is_default": name == default_partition,
                "preemptible": preemptible,
                "state": info["state"],
                "nodes": len(info["node_names"]),
                "total_gpus": sum(n["total_gpus"] for n in member_nodes),
                "free_gpus": sum(n["free_gpus"] for n in member_nodes),
                "reclaimable_gpus": reclaimable_gpus,
                "waiting_jobs": waiting_by_partition.get(name, 0),
            }
        )
    return results


# Module-level cache: a plain dict (mirrors common/poller.py's own
# module-level state, e.g. _pending_since) rather than a class, since
# there's exactly one process-wide snapshot, never one per caller.
_cache: dict[str, object] = {"snapshot": None, "generated_at": None, "monotonic_at": 0.0}
_cache_lock = threading.Lock()


def get_partition_status() -> tuple[list[dict], str]:
    """(partitions, generated_at) -- generated_at is an ISO 8601 UTC
    timestamp string, same convention as JobListResponse.server_time, for
    display only (e.g. "updated 12s ago" in the UI).

    Refreshes at most once every _CACHE_SECONDS, guarded by a lock so
    several requests arriving at once (e.g. multiple browser tabs' own
    PartitionField all polling right after a page load) trigger exactly
    one real Slurm query, not one each -- the rest simply reuse whatever
    that single in-flight refresh produces.

    Raises PartitionStatusError if Slurm can't be queried right now and no
    previous snapshot exists yet to fall back to. Once at least one
    snapshot has ever succeeded, a later failure logs a warning and keeps
    serving that last-known-good snapshot (staler than _CACHE_SECONDS, but
    real data) instead of failing the request outright -- a transient
    controller hiccup shouldn't take down a display-only endpoint.
    """
    with _cache_lock:
        now = time.monotonic()
        if _cache["snapshot"] is not None and now - _cache["monotonic_at"] < _CACHE_SECONDS:
            return _cache["snapshot"], _cache["generated_at"]
        try:
            snapshot = _build_snapshot()
        except PartitionStatusError as e:
            if _cache["snapshot"] is not None:
                logger.warning("Refreshing partition status failed (%s); serving the last-known snapshot instead.", e)
                return _cache["snapshot"], _cache["generated_at"]
            raise
        _cache["snapshot"] = snapshot
        _cache["monotonic_at"] = now
        _cache["generated_at"] = datetime.now(timezone.utc).isoformat()
        return _cache["snapshot"], _cache["generated_at"]
