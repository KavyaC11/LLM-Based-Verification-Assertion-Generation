"""
Phase 7 — Memory & Feedback Update
"""

import json, math, re, time
from pathlib import Path
from config import CACHE_DIR

_MEMORY_FILE     = Path(CACHE_DIR) / "grounding_memory.json"
_ASSERTION_STORE = Path(CACHE_DIR) / "assertion_store.json"
_DESIGN_META     = Path(CACHE_DIR) / "design_metadata.json"
_COGNI_SESSION   = Path(CACHE_DIR) / "cogni_session.json"
_COGNI_TRIPLES   = Path(CACHE_DIR) / "cogni_triples.json"
_COGNI_CHUNKS    = Path(CACHE_DIR) / "cogni_chunks.json"

_WEIBULL_TK = 0.1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save(path: Path, data: dict):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ── Weibull decay ─────────────────────────────────────────────────────────────

def _weibull_decay(delta_tau: float, lam: float, tk: float = _WEIBULL_TK) -> float:
    if lam <= 0:
        return 1.0
    return math.exp(-((delta_tau / lam) ** tk))


def _median_time_gap(entries: list[dict]) -> float:
    now  = time.time()
    gaps = sorted(
        now - e.get("last_updated", now)
        for e in entries
        if isinstance(e.get("last_updated"), (int, float))
    )
    if not gaps:
        return 1.0
    mid = len(gaps) // 2
    return (gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2)


# ── Unified relevance reranking ───────────────────────────────────────────────

def _unified_rerank_score(ss: float, st: float,
                          delta_tau: float, lam: float) -> float:
    if ss + st == 0:
        return 0.0
    ssem = 2 * ss * st / (ss + st)
    w    = _weibull_decay(delta_tau, lam)
    return ssem * w


# ═══════════════════════════════════════════════════════════════════════════════
# CogniGraph — Three-Tier Hierarchical Memory
# ═══════════════════════════════════════════════════════════════════════════════

def _cogni_update_session(rule_id: str, validation_passed: bool, confidence: int,
                          memory_stats: dict | None = None):
    """Session level: high-level summary + memory stats from Phase 3."""
    sessions = _load(_COGNI_SESSION)
    entry = {
        "summary":      f"Rule {rule_id} processed; "
                        f"{'passed' if validation_passed else 'failed'} "
                        f"tier-1 validation with confidence {confidence}/100.",
        "keywords":     [rule_id, "validation",
                         "pass" if validation_passed else "fail"],
        "temporal_marker": time.time(),
        "last_updated":    time.time(),
    }
    if memory_stats:
        entry["memory_health"] = memory_stats
    sessions[rule_id] = entry
    _save(_COGNI_SESSION, sessions)


def _cogni_update_triples(groundings: dict, validation_passed: bool):
    """Entity-Relation level: dedup triple store."""
    triples = _load(_COGNI_TRIPLES)

    for spec_sig, g in groundings.items():
        rtl_name = g.get("rtl_name")
        if not rtl_name:
            continue

        triple_key = f"{spec_sig}::{rtl_name}"

        if triple_key in triples:
            existing = triples[triple_key]
            existing["source_refs"] = existing.get("source_refs", [])
            existing["source_refs"].append({
                "timestamp":        time.time(),
                "validation_passed": validation_passed,
            })
            existing["source_refs"] = existing["source_refs"][-10:]
            existing["last_updated"] = time.time()
        else:
            triples[triple_key] = {
                "spec_signal":      spec_sig,
                "rtl_signal":       rtl_name,
                "confidence":       g.get("confidence", 0.0),
                "method":           g.get("method", "unknown"),
                "validation_passed": validation_passed,
                "source_refs":      [{
                    "timestamp":        time.time(),
                    "validation_passed": validation_passed,
                }],
                "last_updated":     time.time(),
            }

    _save(_COGNI_TRIPLES, triples)


def _cogni_update_chunks(rule_id: str, assertion: dict, context: dict | None):
    """Chunk level: raw assertion + triple back-links."""
    chunks = _load(_COGNI_CHUNKS)

    triple_refs = []
    if context:
        for g in context.get("grounded_signals", []):
            if g.get("rtl_name"):
                triple_refs.append(f"{g.get('spec_name', '')}::{g['rtl_name']}")

    chunks[rule_id] = {
        "assertion_code": assertion.get("property_code", ""),
        "assert_stmt":    assertion.get("assert_statement", ""),
        "confidence":     assertion.get("confidence", 0),
        "tier1_passed":   assertion.get("tier1_passed", False),
        "tier2_passed":   assertion.get("tier2_passed", False),
        "warnings":       assertion.get("warnings", []),
        "triple_refs":    triple_refs,
        "last_updated":   time.time(),
    }
    _save(_COGNI_CHUNKS, chunks)


def retrieve_relevant_memory(query_rule_id: str, top_k: int = 5) -> list:
    """Unified relevance reranking for retrieved triples."""
    triples  = _load(_COGNI_TRIPLES)
    sessions = _load(_COGNI_SESSION)
    now      = time.time()

    if not triples:
        return []

    triple_list = list(triples.values())
    lam = _median_time_gap(triple_list)

    ranked = []
    for triple in triple_list:
        session_entry = sessions.get(query_rule_id, {})
        ss = 1.0 if query_rule_id in session_entry.get("summary", "") else 0.5
        st = triple.get("confidence", 0.5)
        delta_tau = now - triple.get("last_updated", now)
        score = _unified_rerank_score(ss, st, delta_tau, lam)
        ranked.append({**triple, "_rerank_score": round(score, 4)})

    ranked.sort(key=lambda x: x["_rerank_score"], reverse=True)
    return ranked[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# 7.1  Memory store update
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_sva(code: str) -> str:
    """Normalize SVA code for comparison by removing comments and standardizing whitespace."""
    if not code:
        return ""
    # Remove comments
    code = re.sub(r"//.*", "", code)
    # Standardize whitespace: remove newlines, multiple spaces -> single space
    code = re.sub(r"\s+", " ", code).strip()
    # Remove trailing semicolons that might differ
    code = code.strip(';')
    return code

def _update_assertion_store(rule_id: str, assertion: dict, validation_passed: bool):
    store    = _load(_ASSERTION_STORE)
    existing = store.get(rule_id)

    if existing and existing.get("validation_passed") and validation_passed:
        old_code = existing.get("assertion", {}).get("property_code", "")
        new_code = assertion.get("property_code", "")
        norm_old = _normalize_sva(old_code)
        norm_new = _normalize_sva(new_code)
        if norm_old and norm_new and norm_old != norm_new:
            print(f"  [memory] ⚠  Rule {rule_id}: new assertion differs from stored "
                  f"validated assertion — flagging for design_spec_review")
            assertion["warnings"] = assertion.get("warnings", []) + [
                "CONTRADICTION: This rule was previously asserted differently. "
                "Review spec vs RTL alignment."
            ]

    store[rule_id] = {
        "assertion":         assertion,
        "validation_passed": validation_passed,
        "confidence":        assertion.get("confidence", 0),
        "last_updated":      time.time(),
    }
    _save(_ASSERTION_STORE, store)


def _update_context_templates(rule: dict, context: dict, validation_passed: bool):
    meta    = _load(_DESIGN_META)
    rtl     = context.get("rtl_slice", {})
    # derive a design key from clocks/resets as proxy for touched modules
    clocks  = rtl.get("clocks", [])
    key_sig = clocks[0] if clocks else "unknown"
    rule_type  = rule["timing"].get("type", "unspecified")
    key        = f"{key_sig}::{rule_type}"

    entry = meta.get(key, {"success": 0, "fail": 0})
    if validation_passed:
        entry["success"] = entry.get("success", 0) + 1
    else:
        entry["fail"] = entry.get("fail", 0) + 1
    meta[key] = entry
    _save(_DESIGN_META, meta)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.2  Confidence model update
# ═══════════════════════════════════════════════════════════════════════════════

def _update_grounding_confidence(groundings: dict, validation_passed: bool):
    from phase3_grounding import update_grounding_memory
    update_grounding_memory(groundings, validation_passed)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.3  Staleness detection via Weibull decay
# ═══════════════════════════════════════════════════════════════════════════════

def check_staleness(current_rtl_ir: dict, design_version: str = "v1"):
    """
    Detect and mark stale grounding memory entries.
    Mechanism 1: signal absent from RTL → hard stale.
    Mechanism 2: Weibull-decayed confidence < STALE_CONFIDENCE_FLOOR → flag.
    Marked entries will be excluded by phase3's active filter.
    """
    from config import STALE_CONFIDENCE_FLOOR
    memory   = _load(_MEMORY_FILE)
    rtl_sigs = set(current_rtl_ir.get("all_signals", []))
    entries  = list(memory.values())
    lam      = _median_time_gap(entries)
    now      = time.time()
    stale    = []

    for spec_norm, entry in memory.items():
        rtl_name     = entry.get("rtl_name")
        last_updated = entry.get("last_updated", now)
        delta_tau    = now - last_updated

        if rtl_name and rtl_name not in rtl_sigs:
            stale.append(spec_norm)
            entry["stale"]        = True
            entry["stale_reason"] = f"RTL signal '{rtl_name}' no longer in design"
            entry["success_rate"] = entry.get("success_rate", 0.5) * 0.5
            entry["last_updated"] = now
            continue

        w = _weibull_decay(delta_tau, lam)
        decayed_conf = entry.get("success_rate", 0.5) * w
        entry["decayed_confidence"] = round(decayed_conf, 4)
        entry["weibull_weight"]     = round(w, 4)

        if decayed_conf < STALE_CONFIDENCE_FLOOR:
            stale.append(spec_norm)
            entry["stale"]        = True
            entry["stale_reason"] = (
                f"Weibull-decayed confidence {decayed_conf:.3f} < {STALE_CONFIDENCE_FLOOR} "
                f"(w={w:.3f}, Δτ={delta_tau:.0f}s, λ={lam:.0f}s)"
            )

    if stale:
        print(f"  [memory] Staleness: {len(stale)} grounding(s) marked stale "
              f"(signal-absent + Weibull-decay): {stale[:5]}")
        _save(_MEMORY_FILE, memory)

    return stale


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════════════════════

def update_memory(
    rule_id: str,
    groundings: dict,
    assertion: dict,
    validation_passed: bool,
    rule: dict | None = None,
    context: dict | None = None,
    memory_stats: dict | None = None,
):
    """
    7.1 + 7.2 + CogniGraph update.

    Execution order:
      1. Assertion store         (flat key-value, contradiction check)
      2. Grounding confidence    (reinforcement ±)
      3. Context templates       (slice-depth statistics)
      4. CogniGraph session      (high-level summary + temporal marker)
      5. CogniGraph triples      (entity-relation with dedup)
      6. CogniGraph chunks       (raw evidence + back-links)
    """
    _update_assertion_store(rule_id, assertion, validation_passed)
    _update_grounding_confidence(groundings, validation_passed)

    if rule and context:
        _update_context_templates(rule, context, validation_passed)

    _cogni_update_session(rule_id, validation_passed,
                          assertion.get("confidence", 0),
                          memory_stats=memory_stats)
    _cogni_update_triples(groundings, validation_passed)
    _cogni_update_chunks(rule_id, assertion, context)

    print(f"  [Phase 7] Memory updated for rule {rule_id} "
          f"({'✓ passed' if validation_passed else '✗ failed'})")