"""
Phase 0 — Session & Cache Initialization
-----------------------------------------
Implements:
  0.1  Session management (create / load / hash-verify)
  0.2  Design cache check (RTL IR, Spec IR, dep-graph, FSMs, groundings)

Three-Layer Caching Hierarchy:
  L1 — Design cache   : Full RTL IR (dependency graph, modules, FSMs)
                        Key: design_hash  →  skips re-parsing on re-run
  L2 — Context cache  : Assembled context package per rule
                        Key: context_hash →  skips slicing for repeated rules
  L3 — Assertion cache: Validated assertion code per rule
                        Key: (design_hash, rule_id) → eliminates LLM call entirely
"""

import json, hashlib, os, uuid
from pathlib import Path
from config import CACHE_DIR

_cache = Path(CACHE_DIR)
_cache.mkdir(exist_ok=True)


# ── Hashing ───────────────────────────────────────────────────────────────────

def compute_design_hash(spec_file: str, rtl_files: list, version: str = "v1") -> str:
    """Stable hash over spec + all RTL content + version string."""
    h = hashlib.sha256()
    h.update(version.encode())
    for f in sorted([spec_file] + rtl_files):
        if os.path.exists(f):
            h.update(Path(f).read_bytes())
        else:
            h.update(f.encode())        # include path even if file absent
    return h.hexdigest()[:16]


def compute_context_hash(context_package: dict) -> str:
    payload = json.dumps(context_package, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── Session persistence ───────────────────────────────────────────────────────

def load_session(session_id: str) -> dict:
    p = _cache / f"session_{session_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_session(session_id: str, state: dict):
    p = _cache / f"session_{session_id}.json"
    p.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


# ── L1: Design cache (RTL IR + derived artefacts) ────────────────────────────

def load_design_cache(design_hash: str) -> dict | None:
    p = _cache / f"design_{design_hash}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_design_cache(design_hash: str, data: dict):
    p = _cache / f"design_{design_hash}.json"
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"  [cache] L1 design cache saved: {design_hash}")


# ── L2: Context cache (per-rule slice packages) ───────────────────────────────

def load_context_cache(context_hash: str) -> dict | None:
    p = _cache / f"ctx_{context_hash}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_context_cache(context_hash: str, data: dict):
    p = _cache / f"ctx_{context_hash}.json"
    p.write_text(json.dumps(data, indent=2, default=str))


# ── L3: Assertion cache (validated SVA per rule) ──────────────────────────────
# Eliminates the LLM call entirely for repeated or previously-validated rules.
# Only tier1_passed=True assertions are persisted — failed ones are never cached.
# The design_hash is baked into the key so any RTL/spec change auto-invalidates.

def load_assertion_cache(rule_id: str, design_hash: str) -> dict | None:
    """
    Look up a previously validated assertion for (rule_id, design_hash).
    Returns the stored assertion dict, or None if absent / not yet validated.
    """
    p = _cache / f"assert_{design_hash}_{rule_id}.json"
    if p.exists():
        try:
            entry = json.loads(p.read_text(encoding="utf-8"))
            if entry.get("tier1_passed"):
                return entry
        except Exception:
            pass
    return None


def save_assertion_cache(rule_id: str, design_hash: str, assertion: dict):
    """
    Persist a validated assertion for (rule_id, design_hash).
    Must only be called after Tier-1 + Tier-2 validation passes in Phase 6.
    """
    if not assertion.get("tier1_passed"):
        return          # never cache failed assertions
    p = _cache / f"assert_{design_hash}_{rule_id}.json"
    p.write_text(json.dumps(assertion, indent=2, default=str), encoding="utf-8")
    print(f"  [cache] L3 assertion cache saved: rule={rule_id}  design={design_hash}")


def invalidate_assertion_cache(design_hash: str):
    """
    Remove all L3 entries for a given design hash.
    Triggered automatically when init_session detects a hash mismatch,
    i.e. the RTL or spec was modified since the last run.
    """
    removed = 0
    for p in _cache.glob(f"assert_{design_hash}_*.json"):
        p.unlink()
        removed += 1
    if removed:
        print(f"  [cache] L3 assertion cache invalidated: {removed} entry/entries "
              f"removed for design {design_hash}")


# ── Main entry ────────────────────────────────────────────────────────────────

def init_session(spec_file: str, rtl_files: list, version: str = "v1",
                 existing_session_id: str | None = None) -> dict:
    """
    0.1 / 0.2  Create or resume a session.
    Returns session state dict with keys:
        session_id, design_hash, cached_data, from_cache

    Cache behaviour on init:
        L1 hit  → RTL IR reused; L2/L3 entries remain valid.
        L1 miss → full re-parse triggered; stale L3 entries for the old
                  design hash are automatically invalidated.
    """
    session_id   = existing_session_id or str(uuid.uuid4())[:8]
    design_hash  = compute_design_hash(spec_file, rtl_files, version)

    # Try to load existing session state
    session_state = load_session(session_id) if existing_session_id else {}

    # 0.1 — Verify hash consistency; invalidate L3 on mismatch
    stored_hash = session_state.get("design_hash")
    if stored_hash and stored_hash != design_hash:
        print(f"  [session] Design hash mismatch ({stored_hash} → {design_hash}). "
              f"Re-syncing.")
        invalidate_assertion_cache(stored_hash)   # stale L3 entries no longer valid
        session_state = {}

    # 0.2 — L1 design cache check
    cached = load_design_cache(design_hash)

    new_state = {
        "session_id":  session_id,
        "design_hash": design_hash,
        "spec_file":   spec_file,
        "rtl_files":   rtl_files,
        "version":     version,
        **session_state,
    }
    save_session(session_id, new_state)

    return {
        "session_id":  session_id,
        "design_hash": design_hash,
        "cached_data": cached,
        "from_cache":  cached is not None,
    }