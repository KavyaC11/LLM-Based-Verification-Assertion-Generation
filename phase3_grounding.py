"""
Phase 3 — Grounding & Alignment
"""

import re, json, time, math
from pathlib import Path
from rapidfuzz import fuzz, process as fuzz_proc
from config import (
    GROUNDING_HIGH, GROUNDING_MED, FUZZY_THRESHOLD, CACHE_DIR,
    STALE_CONFIDENCE_FLOOR, STALE_PRUNE_AFTER_RUNS,
)

_MEMORY_FILE  = Path(CACHE_DIR) / "grounding_memory.json"
_RUN_CTR_FILE = Path(CACHE_DIR) / "grounding_run_counter.json"


# ── Persistence helpers ───────────────────────────────────────────────────────
# loads historical knowledge about previous spec-to-RTL mappings
def load_memory() -> dict:
    if _MEMORY_FILE.exists():
        try:
            return json.loads(_MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

# updated grounding history is persisted
def save_memory(mem: dict):
    _MEMORY_FILE.parent.mkdir(exist_ok=True)
    _MEMORY_FILE.write_text(json.dumps(mem, indent=2), encoding="utf-8")


# ── Run counter ────────────────────────────────────────────────────
# tracks how many times the grounding phase has been run
def _increment_run_counter() -> int:
    """Increment the pipeline run counter and return the new count."""
    data = {}
    if _RUN_CTR_FILE.exists():
        try:
            data = json.loads(_RUN_CTR_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    count = data.get("runs", 0) + 1
    _RUN_CTR_FILE.write_text(json.dumps({"runs": count}), encoding="utf-8")
    return count


# ── Active memory filter — called at retrieval time ───────────────
# retrieval-time stale filtering
def _filter_active_memory(memory: dict) -> dict:
    """
    Return a view of memory with stale entries removed.

    An entry is excluded if:
      (a) it is explicitly marked stale=True, OR
      (b) its decayed_confidence (set by phase7 staleness check) is below
          STALE_CONFIDENCE_FLOOR, OR
      (c) its success_rate itself is below STALE_CONFIDENCE_FLOOR.

    Stale entries are not merely downweighted — they are excluded
    from influencing new confidence scores entirely.
    """
    active = {}
    pruned = 0
    for k, v in memory.items():
        if v.get("stale", False):
            pruned += 1
            continue
        decayed = v.get("decayed_confidence", v.get("success_rate", 1.0))
        if decayed < STALE_CONFIDENCE_FLOOR:
            pruned += 1
            continue
        active[k] = v
    if pruned:
        print(f" Active stale filter: {pruned} entries excluded "
              f"({len(active)} active of {len(memory)} total)")
    return active


# ── Hard prune — called every STALE_PRUNE_AFTER_RUNS runs ─────────
# performs physical pruning of the grounding memory file on disk
def prune_stale_memory() -> int:
    """
    Physically remove stale entries from grounding_memory.json.

    Called automatically when the pipeline run counter reaches a multiple of
    STALE_PRUNE_AFTER_RUNS.  Returns the number of entries pruned.
    """
    memory = load_memory()
    active = _filter_active_memory(memory)
    pruned = len(memory) - len(active)
    if pruned > 0:
        save_memory(active)
        print(f" Hard prune: removed {pruned} stale entries from disk "
              f"({len(active)} remain)")
    return pruned


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", str(s))
    return s.lower().replace("-", "_").replace(" ", "_")


# ── 3.1  Single-signal grounding ─────────────────────────────────────────────
# tries to ground one spec signal to one RTL signal
# exact normalized match
# fuzzy similarity
# graph-context bonus
# memory bonus
# threshold-based confidence labeling
def _ground_one(
    spec_sig: str,
    rtl_norm: dict,
    rtl_norm_list: list,
    dep_graph: dict,
    already_grounded: set,
    memory: dict,           # already-filtered active memory
) -> dict:
    spec_norm = _normalize(spec_sig)
    result = {
        "spec_name":  spec_sig,
        "rtl_name":   None,
        "confidence": 0.0,
        "method":     "none",
        "candidates": [],
    }

    # Step 1 — exact match
    if spec_norm in rtl_norm:
        result.update({
            "rtl_name":   rtl_norm[spec_norm],
            "confidence": 1.0,
            "method":     "exact",
        })
        return result

    # Step 2 — fuzzy matching (top-3 candidates)
    raw_matches = fuzz_proc.extract(
        spec_norm, rtl_norm_list,
        scorer=fuzz.token_sort_ratio, limit=3
    )
    candidates = []
    for norm_cand, score, _ in raw_matches:
        if score >= FUZZY_THRESHOLD:
            candidates.append({
                "rtl_name":   rtl_norm[norm_cand],
                "base_score": score / 100.0 * 0.6,
            })

    if not candidates:
        result["candidates"] = [
            {"rtl_name": rtl_norm[n], "base_score": s / 100.0 * 0.6}
            for n, s, _ in raw_matches
        ]
        return result

    # Step 3 — dependency graph bonus
    for cand in candidates:
        rtl_n = cand["rtl_name"]
        connected = dep_graph.get(rtl_n, {})
        def _sig_names(lst):
            return [e["signal"] if isinstance(e, dict) else e for e in lst]
        related = set(_sig_names(connected.get("depends_on", [])) +
                      connected.get("drives", []))
        if related & already_grounded:
            cand["base_score"] += 0.2
        if connected.get("clock"):
            cand["base_score"] += 0.10

    # Step 4 — memory bonus uses only active (non-stale) entries
    mem_entry = memory.get(spec_norm)
    for cand in candidates:
        if mem_entry and mem_entry.get("rtl_name") == cand["rtl_name"]:
            # Use decayed_confidence if available, else raw success_rate
            effective_rate = mem_entry.get(
                "decayed_confidence", mem_entry.get("success_rate", 0.0)
            )
            cand["base_score"] += 0.15 * effective_rate

    best = max(candidates, key=lambda c: c["base_score"])
    result["candidates"] = candidates

    if best["base_score"] >= GROUNDING_HIGH:
        method = "high"
    elif best["base_score"] >= GROUNDING_MED:
        method = "medium"
    else:
        method = "low"

    result.update({
        "rtl_name":   best["rtl_name"],
        "confidence": min(best["base_score"], 1.0),
        "method":     method,
    })
    return result


# ── 3.2  Disambiguation report ────────────────────────────────────────────────

def _build_disambiguation_report(groundings: dict, rtl_ir: dict) -> list:
    report = []
    for spec_sig, g in groundings.items():
        if g["confidence"] < GROUNDING_MED:
            top3 = g.get("candidates", [])[:3]
            cand_lines = []
            for c in top3:
                mod = rtl_ir["signal_to_module"].get(c["rtl_name"], "?")
                cand_lines.append(
                    f"  • '{c['rtl_name']}' (module: {mod}, score: {c['base_score']:.2f})"
                )
            report.append({
                "spec_signal": spec_sig,
                "message": (
                    f"Cannot confidently map spec term '{spec_sig}'. "
                    f"Top candidates:\n" + "\n".join(cand_lines)
                ),
                "candidates": [c["rtl_name"] for c in top3],
            })
    return report


# ── 3.3  Alignment validation ─────────────────────────────────────────────────

def _alignment_validation(groundings: dict, spec_ir: dict, rtl_ir: dict) -> dict:
    checks, matches = 0, 0

    if spec_ir.get("state_mentions"):
        checks += 1
        if rtl_ir["fsms"]:
            matches += 1

    timing_words = any(
        r["timing"]["type"] not in ("unspecified",)
        for r in spec_ir.get("rules", [])
    )
    if timing_words:
        checks += 1
        has_seq = any(mod.get("clocks") for mod in rtl_ir["modules"].values())
        if has_seq:
            matches += 1

    matched = sum(1 for v in groundings.values() if v["confidence"] >= GROUNDING_MED)
    total   = max(len(groundings), 1)
    checks  += total
    matches += matched
    score    = matches / max(checks, 1)

    return {
        "alignment_score":     round(score, 3),
        "matched_signals":     matched,
        "total_signals":       total,
        "design_review_needed": score < 0.6,
    }


# ── Main entry ────────────────────────────────────────────────────────────────

def ground_signals(spec_ir: dict, rtl_ir: dict) -> dict:
    """
    Memory is actively filtered before Step 4 bonus computation.
    Stale entries do not contribute any confidence bonus.
    Run counter is incremented; hard prune fires every STALE_PRUNE_AFTER_RUNS runs.
    """
    print("[Phase 3] Grounding spec signals to RTL …")

    # Increment run counter and conditionally hard-prune
    run_count = _increment_run_counter()
    if run_count % STALE_PRUNE_AFTER_RUNS == 0:
        print(f" Run {run_count}: triggering hard stale prune …")
        prune_stale_memory()

    raw_memory   = load_memory()
    # Filter before use
    active_memory = _filter_active_memory(raw_memory)

    rtl_signals   = rtl_ir["all_signals"]
    rtl_norm      = {_normalize(s): s for s in rtl_signals}
    rtl_norm_list = list(rtl_norm.keys())
    dep_graph     = rtl_ir.get("dep_graph", {})

    groundings       = {}
    already_grounded = set()

    for spec_sig in spec_ir["mentioned_signals"]:
        g = _ground_one(
            spec_sig, rtl_norm, rtl_norm_list,
            dep_graph, already_grounded, active_memory,   # pass filtered memory
        )
        groundings[spec_sig] = g
        if g["rtl_name"] and g["confidence"] >= GROUNDING_MED:
            already_grounded.add(g["rtl_name"])

    disambiguation = _build_disambiguation_report(groundings, rtl_ir)
    alignment      = _alignment_validation(groundings, spec_ir, rtl_ir)

    matched = alignment["matched_signals"]
    total   = alignment["total_signals"]
    stale_excluded = len(raw_memory) - len(active_memory)
    print(f"  → {matched}/{total} signals grounded "
          f"(alignment score: {alignment['alignment_score']:.2f})")
    if stale_excluded:
        print(f"  → {stale_excluded} stale memory entries excluded")
    if disambiguation:
        print(f"  → {len(disambiguation)} signal(s) need disambiguation")
    if alignment["design_review_needed"]:
        print("  ⚠  Alignment < 0.6 — spec may not match this RTL")

    return {
        "groundings":        groundings,
        "alignment_score":   alignment["alignment_score"],
        "alignment_detail":  alignment,
        "disambiguation":    disambiguation,
        "unmatched": [
            k for k, v in groundings.items()
            if v["confidence"] < GROUNDING_MED
        ],
        # expose in results for ablation
        "memory_stats": {
            "total_entries":  len(raw_memory),
            "active_entries": len(active_memory),
            "stale_excluded": stale_excluded,
        },
    }


# ── Memory update (called from Phase 7) ──────────────────────────────────────

def update_grounding_memory(groundings: dict, validation_passed: bool):
    memory = load_memory()
    delta  = 0.05 if validation_passed else -0.10
    for spec_sig, g in groundings.items():
        k = _normalize(spec_sig)
        if k not in memory:
            memory[k] = {
                "rtl_name":    g["rtl_name"],
                "success_rate": 0.5,
                "count":        0,
                "last_updated": time.time(),
            }
        memory[k]["success_rate"] = min(1.0, max(0.0,
            memory[k]["success_rate"] + delta))
        memory[k]["count"]       += 1
        memory[k]["last_updated"] = time.time()
        if g["rtl_name"]:
            memory[k]["rtl_name"] = g["rtl_name"]
        # Reset stale flag on successful update
        if validation_passed:
            memory[k].pop("stale", None)
            memory[k].pop("stale_reason", None)
    save_memory(memory)