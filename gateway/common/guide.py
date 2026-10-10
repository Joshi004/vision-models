"""Builds the content behind GET /v1/guide: the single, always-current
reference a third-party system should read to use this whole API, end to
end -- what every backend does, how jobs/uploads work, and the exact
current endpoint reference.

Three live inputs, never copied by hand, so the guide cannot drift from
reality the way a hand-maintained document can:
  1. This process's own config state (common/config.py) -- "facts" below.
  2. The running app's own OpenAPI schema (FastAPI's app.openapi(), the
     same thing /openapi.json serves) -- the endpoint reference.
  3. A handful of short narrative .md files (common/guide_sections/, plus
     one services/<name>/GUIDE.md per backend) -- concepts and advice that
     don't belong in a single endpoint's own Swagger text.

Each endpoint's own Swagger text (services/<name>/openapi_docs.py,
common/openapi_docs.py) stays the single source of truth for that
endpoint's own behavior -- this module never restates it, only renders it
(see _render_operation): change a field's description there and this
guide's own text changes with it on the next request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from common import config

logger = logging.getLogger("model-api.guide")

_SECTIONS_DIR = Path(__file__).resolve().parent / "guide_sections"
_SERVICES_DIR = config.MODEL_API_DIR / "services"

_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "options", "head"})

# Path prefixes shared by every backend -- not registered in server.py's
# backend list because they're not backend-specific. Used only to group the
# generated endpoint reference; see _bucket_operations.
_SHARED_PATH_PREFIXES = ("/v1/health", "/v1/partitions", "/v1/uploads", "/v1/jobs")

# {{name}} placeholders left for the final per-request fill (see render())
# -- never an error if a narrative file uses one of these, unlike every
# other unknown placeholder name (see _fill).
_DEFERRED_PLACEHOLDERS = frozenset({"base_url", "generated_at"})

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class BackendInfo(NamedTuple):
    """One entry in server.py's backend registry -- what this module needs
    to find and label that backend's own section of the guide."""

    id: str  # short id, matches the "<id>:<recipe>" prefix of that backend's own `pipeline` values
    name: str  # human-readable, e.g. "LTX-2.3"
    prefix: str  # path prefix, e.g. "/v1/ltx"
    guide_file: str  # relative to services/, e.g. "ltx/GUIDE.md"


class GuideBundle(NamedTuple):
    """Everything built once per process and reused by every request --
    see _get_bundle. `template` still has {{base_url}}/{{generated_at}}
    unfilled; render() fills those in per request."""

    template: str
    content_hash: str  # "sha256:<hex>" of `template` -- stable across callers/requests
    backend_facts: list[dict[str, str]]


def _fmt_days(days: float) -> str:
    return str(int(days)) if float(days).is_integer() else f"{days:g}"


def _static_facts() -> dict[str, str]:
    """Facts fixed for the life of this process -- computed once, reused by
    both the Swagger header (render_summary(), at import time) and every
    request to GET /v1/guide. Every value here traces back to common/config.py,
    never a hardcoded number, so it can't silently drift from this server's
    actual running configuration."""
    upload_mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
    if config.MAX_INFLIGHT_JOBS is None:
        max_inflight_line = "No fixed concurrency cap is configured on this deployment right now."
    else:
        max_inflight_line = (
            f"At most {config.MAX_INFLIGHT_JOBS} job(s) may be in flight across every backend at "
            "once on this deployment right now; a request past that cap gets a 429."
        )
    return {
        "upload_mb": str(upload_mb),
        "job_retention_days": _fmt_days(config.JOB_RETENTION_DAYS),
        "default_partition": config.FALLBACK_PARTITION,
        "max_inflight_line": max_inflight_line,
    }


STATIC_FACTS: dict[str, str] = _static_facts()


def _fill(text: str, facts: dict[str, str], *, source: str, defer: frozenset[str] = frozenset()) -> str:
    """Replace every {{name}} in text with facts[name]. A name in `defer` is
    left untouched (no log -- it's meant to be filled by a later pass, see
    render()). Any other name with no entry in facts is also left untouched,
    but logged as an error: a docs typo must never take the API down."""

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in defer:
            return match.group(0)
        if name not in facts:
            logger.error("guide: unknown placeholder {{%s}} in %s", name, source)
            return match.group(0)
        return str(facts[name])

    return _PLACEHOLDER_RE.sub(repl, text)


def _load(path: Path, facts: dict[str, str], *, defer: frozenset[str] = frozenset()) -> str | None:
    """Read and fill one narrative file. Returns None (logging an error) if
    it's missing or unreadable -- a missing optional section is skipped, not
    a reason to fail the whole guide."""
    try:
        text = path.read_text()
    except OSError as e:
        logger.error("guide: could not read %s: %s", path, e)
        return None
    return _fill(text, facts, source=str(path), defer=defer).strip()


def render_summary() -> str:
    """The short, caller-facing summary -- used verbatim as the Swagger
    APP_DESCRIPTION (at import time, by common/openapi_docs.py) and as the
    opening section of the full guide. Only STATIC_FACTS are available here
    -- common/guide_sections/00_summary.md must not use {{base_url}} or
    {{generated_at}}, since Swagger renders this at import time, long before
    any request (and its base_url) exists."""
    text = _load(_SECTIONS_DIR / "00_summary.md", STATIC_FACTS)
    if text is None:
        return "A single HTTP API for GPU-cluster model-serving backends on this node."
    return text


# ---------------------------------------------------------------------------
# Endpoint reference: rendered straight from the app's own OpenAPI schema,
# so it is always exactly what /openapi.json and /docs show -- nothing here
# is a second, hand-maintained copy of any endpoint's own docs.
# ---------------------------------------------------------------------------


def _demote_headings(text: str, by: int = 2) -> str:
    """Shift Markdown headings in an endpoint description down so they nest
    under this guide's own heading structure instead of competing with it --
    every endpoint description is written to stand alone at the top level in
    Swagger, where this problem doesn't exist."""

    def repl(match: re.Match[str]) -> str:
        hashes, rest = match.group(1), match.group(2)
        return "#" * min(len(hashes) + by, 6) + rest

    return re.sub(r"^(#{1,6})( .+)$", repl, text, flags=re.MULTILINE)


def _schema_name(schema_obj: dict[str, Any] | None) -> str | None:
    ref = (schema_obj or {}).get("$ref")
    return ref.rsplit("/", 1)[-1] if ref else None


def _render_operation(path: str, method: str, op: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = op.get("summary") or op.get("operationId") or f"{method.upper()} {path}"
    lines.append(f"#### `{method.upper()} {path}` -- {summary}")
    lines.append("")

    description = (op.get("description") or "").strip()
    if description:
        lines.append(_demote_headings(description))
        lines.append("")

    params = [p for p in op.get("parameters") or [] if isinstance(p, dict)]
    if params:
        lines.append("**Parameters:**")
        for p in params:
            p_schema = p.get("schema") or {}
            required = "required" if p.get("required") else "optional"
            lines.append(
                f"- `{p.get('name')}` ({p.get('in')}, {p_schema.get('type', 'any')}, {required}) -- "
                f"{p.get('description', '')}"
            )
        lines.append("")

    for media_type, media in (op.get("requestBody") or {}).get("content", {}).items():
        name = _schema_name(media.get("schema"))
        suffix = f", schema `{name}`" if name else ""
        lines.append(f"**Request body** (`{media_type}`{suffix}):")
        lines.append("")
        for ex_name, example in (media.get("examples") or {}).items():
            if not isinstance(example, dict) or example.get("value") is None:
                continue
            lines.append(f"*Example -- {example.get('summary', ex_name)}:*")
            lines.append("```json")
            lines.append(json.dumps(example["value"], indent=2, default=str))
            lines.append("```")
        lines.append("")

    responses = op.get("responses") or {}
    if responses:
        lines.append("**Responses:**")
        for code in sorted(responses):
            desc = " ".join((responses[code].get("description") or "").split())
            lines.append(f"- `{code}` -- {desc}")
        lines.append("")

    return "\n".join(lines)


def _iter_operations(schema: dict[str, Any]):
    """Yields (path, method, op) for every real operation in the schema,
    skipping /v1/guide itself (covered by this guide's own prose instead of
    its own generated entry) and anything that isn't an HTTP operation
    (OpenAPI allows non-method keys like "parameters" at the path level)."""
    for path, methods in (schema.get("paths") or {}).items():
        if path == "/v1/guide" or not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method in _HTTP_METHODS and isinstance(op, dict):
                yield path, method, op


def _path_matches(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def _bucket_operations(
    schema: dict[str, Any], backends: list[BackendInfo]
) -> tuple[list[tuple[str, str, dict]], dict[str, list[tuple[str, str, dict]]], list[tuple[str, str, dict]]]:
    """Splits every operation into (shared, {backend_id: [...]}, other) --
    "other" is anything mounted on this server that matches neither a known
    shared prefix nor a registered backend, so a new backend added to the
    router without a matching server.py registry entry still shows up here
    instead of silently vanishing from the guide."""
    shared: list[tuple[str, str, dict]] = []
    by_backend: dict[str, list[tuple[str, str, dict]]] = {b.id: [] for b in backends}
    other: list[tuple[str, str, dict]] = []

    for path, method, op in sorted(_iter_operations(schema), key=lambda t: (t[0], t[1])):
        owner = next((b.id for b in backends if _path_matches(path, b.prefix)), None)
        if owner is not None:
            by_backend[owner].append((path, method, op))
        elif any(_path_matches(path, prefix) for prefix in _SHARED_PATH_PREFIXES):
            shared.append((path, method, op))
        else:
            other.append((path, method, op))
    return shared, by_backend, other


def _build_bundle(app: Any, backends: list[BackendInfo]) -> GuideBundle:
    schema = app.openapi()
    shared_ops, backend_ops, other_ops = _bucket_operations(schema, backends)

    sections: list[str] = [render_summary()]
    for name in ("10_overview", "20_access", "30_quickstart", "40_jobs_and_timing", "50_errors"):
        text = _load(_SECTIONS_DIR / f"{name}.md", STATIC_FACTS, defer=_DEFERRED_PLACEHOLDERS)
        if text:
            sections.append(text)

    sections.append(
        "## Endpoint reference\n\n"
        "Every endpoint below is generated directly from this server's own `/openapi.json` -- "
        "nothing here is a separate, hand-maintained copy, so it cannot drift from what `/docs` "
        "shows. Response bodies shown are illustrative examples, not a promise of exact wording; "
        "see `/openapi.json` for the full, formal schema of every request and response."
    )

    if shared_ops:
        sections.append("### Shared endpoints (every backend)")
        sections.extend(_render_operation(p, m, op) for p, m, op in shared_ops)

    backend_facts: list[dict[str, str]] = []
    for b in backends:
        backend_facts.append({"id": b.id, "name": b.name, "prefix": b.prefix})
        sections.append(f"### {b.name} (`{b.prefix}`)")
        guide_text = _load(_SERVICES_DIR / b.guide_file, STATIC_FACTS, defer=_DEFERRED_PLACEHOLDERS)
        if guide_text:
            sections.append(guide_text)
        sections.extend(_render_operation(p, m, op) for p, m, op in backend_ops.get(b.id, []))

    if other_ops:
        sections.append(
            "### Other endpoints\n\n"
            "Endpoints mounted on this server that aren't described above yet -- likely a backend "
            "added to the code without a matching entry in this guide's own registry yet. The "
            "reference below still comes straight from `/openapi.json`, so it's accurate even "
            "though the surrounding prose hasn't caught up."
        )
        sections.extend(_render_operation(p, m, op) for p, m, op in other_ops)

    changelog = _load(_SECTIONS_DIR / "99_changelog.md", STATIC_FACTS, defer=_DEFERRED_PLACEHOLDERS)
    if changelog:
        sections.append(changelog)

    template = "\n\n".join(s for s in sections if s and s.strip())
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return GuideBundle(
        template=template,
        content_hash=f"sha256:{digest}",
        backend_facts=backend_facts,
    )


_bundle_lock = threading.Lock()
_bundle_cache: GuideBundle | None = None


def _get_bundle(app: Any, backends: list[BackendInfo]) -> GuideBundle:
    """Built once per process, on first request, then cached -- re-reading
    every narrative file and re-walking the OpenAPI schema on every call
    would be wasted work for content that only changes on a restart anyway
    (same "restart to pick up changes" convention this API's other
    import-time config already follows)."""
    global _bundle_cache
    if _bundle_cache is None:
        with _bundle_lock:
            if _bundle_cache is None:
                _bundle_cache = _build_bundle(app, backends)
    return _bundle_cache


def render(app: Any, backends: list[BackendInfo], base_url: str) -> tuple[str, dict[str, Any], str, str]:
    """One request's worth of the guide: (markdown, facts, content_hash,
    generated_at). Everything except base_url/generated_at comes from the
    cached GuideBundle (see _get_bundle) -- base_url is filled in fresh per
    request since it depends on how the caller actually reached this API."""
    bundle = _get_bundle(app, backends)
    generated_at = datetime.now(timezone.utc).isoformat()
    markdown = _fill(bundle.template, {"base_url": base_url, "generated_at": generated_at}, source="<render>")
    facts: dict[str, Any] = {
        "base_url": base_url,
        "limits": {
            "max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024),
            "job_retention_days": config.JOB_RETENTION_DAYS,
            "max_inflight_jobs": config.MAX_INFLIGHT_JOBS,
        },
        "polling": {
            "recommended_interval_seconds": [10, 15],
            "webhooks": False,
        },
        "partitions": {
            "default": config.FALLBACK_PARTITION,
            "live_status_endpoint": "/v1/partitions",
        },
        "backends": bundle.backend_facts,
    }
    return markdown, facts, bundle.content_hash, generated_at


def lint(app: Any) -> list[str]:
    """Warnings only, never raised -- run once at startup (see server.py's
    lifespan) so a docs gap (e.g. the missing 'Voice conversion' tag, found
    2026-10-04) surfaces in the log immediately instead of silently drifting
    until someone happens to notice it in /docs."""
    warnings: list[str] = []
    schema = app.openapi()
    declared_tags = {t["name"] for t in schema.get("tags", [])}
    for path, method, op in _iter_operations(schema):
        label = f"{method.upper()} {path}"
        if not op.get("summary"):
            warnings.append(f"{label}: no summary")
        if not (op.get("description") or "").strip():
            warnings.append(f"{label}: no description")
        tags = op.get("tags") or []
        if not tags:
            warnings.append(f"{label}: no tag")
        for t in tags:
            if t not in declared_tags:
                warnings.append(f"{label}: tag {t!r} is not declared in TAGS_METADATA")
        codes = set(op.get("responses") or {})
        if method != "get" and codes <= {"200", "422"}:
            warnings.append(f"{label}: documents no error responses beyond 200/422")
    return warnings
