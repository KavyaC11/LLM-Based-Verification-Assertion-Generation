"""
Phase 4 — Context Construction
"""

from pathlib import Path
from collections import deque
import heapq
import re

from phase0_session import compute_context_hash, load_context_cache, save_context_cache
from config import (
    MAX_SLICE_DEPTH, MAX_FWD_DEPTH, MAX_SNIPPET_CHARS,
    SLICE_TOKEN_BUDGET, SLICE_MIN_SIGNALS, SLICE_SCORE_FLOOR,
    SNIPPET_CONTEXT_LINES, SNIPPET_RELEVANCE_BONUS,
)


# ── Budget-aware priority slicer ─────────────────────────────────

def budget_aware_slice(
    seed_signals: set,
    dep_graph: dict,
    grounding_confidences: dict,   # {rtl_signal_name: confidence_float}
    token_budget: int = SLICE_TOKEN_BUDGET,
    min_signals: int = SLICE_MIN_SIGNALS,
    score_floor: float = SLICE_SCORE_FLOOR,
    direction: str = "backward",   # "backward" | "forward" | "both"
) -> set:
    """
    Priority-queue slice

    Enqueues neighbours as (−score, signal, hop_distance) so that Python's
    min-heap always pops the highest-relevance unexplored signal next.

    Score formula:  score(s) = conf(s) × (1 / (hop + 1))
      where conf(s) = grounding_confidences.get(s, 0.3)  [default 0.3 for
      transitively-reached signals with no direct grounding].

    Stops when:
      • the assembled snippet token estimate exceeds token_budget, AND
        at least min_signals have been included, OR
      • the heap is empty, OR
      • the next signal's score falls below score_floor.
    """
    if not seed_signals:
        return set()

    visited  = set()
    included = set()
    token_est = 0   # rough estimate: 1 token ≈ 4 chars, signal name ≈ 10 chars avg

    # Initial heap: seed signals at hop=0
    heap = []
    for sig in seed_signals:
        if sig:
            conf  = grounding_confidences.get(sig, 1.0)  # seeds are grounded
            score = conf / 1.0   # hop=0 → divisor=1
            heapq.heappush(heap, (-score, sig, 0))

    def _neighbours(sig: str, hop: int):
        node = dep_graph.get(sig, {})
        nexts = []
        if direction in ("backward", "both"):
            for dep in node.get("depends_on", []):
                dep_sig = dep["signal"] if isinstance(dep, dict) else dep
                if dep_sig and dep_sig not in visited:
                    nexts.append((dep_sig, hop + 1))
        if direction in ("forward", "both"):
            for drv in node.get("drives", []):
                drv_sig = drv["signal"] if isinstance(drv, dict) else drv
                if drv_sig and drv_sig not in visited:
                    nexts.append((drv_sig, hop + 1))
        return nexts

    while heap:
        neg_score, sig, hop = heapq.heappop(heap)
        score = -neg_score

        if sig in visited:
            continue
        visited.add(sig)

        # Check floor and budget
        if score < score_floor and len(included) >= min_signals:
            break
        if token_est >= token_budget and len(included) >= min_signals:
            break

        included.add(sig)
        token_est += len(sig) // 4 + 1   # rough token cost of the signal name

        # Enqueue neighbours
        for next_sig, next_hop in _neighbours(sig, hop):
            if next_sig not in visited:
                conf      = grounding_confidences.get(next_sig, 0.3)
                nxt_score = conf / (next_hop + 1)
                heapq.heappush(heap, (-nxt_score, next_sig, next_hop))

    return included


# ── Fallback BFS slicers (kept for clock/reset cones) ────────────────────────

def backward_slice(start: str, dep_graph: dict,
                   max_depth: int = MAX_SLICE_DEPTH) -> set:
    visited, queue = set(), deque([(start, 0)])
    while queue:
        sig, depth = queue.popleft()
        if sig in visited or depth > max_depth:
            continue
        visited.add(sig)
        for dep in dep_graph.get(sig, {}).get("depends_on", []):
            dep_sig = dep["signal"] if isinstance(dep, dict) else dep
            queue.append((dep_sig, depth + 1))
    return visited


def forward_slice(start: str, dep_graph: dict,
                  max_depth: int = MAX_FWD_DEPTH) -> set:
    visited, queue = set(), deque([(start, 0)])
    while queue:
        sig, depth = queue.popleft()
        if sig in visited or depth > max_depth:
            continue
        visited.add(sig)
        for drv in dep_graph.get(sig, {}).get("drives", []):
            drv_sig = drv["signal"] if isinstance(drv, dict) else drv
            queue.append((drv_sig, depth + 1))
    return visited


# ── 4.1  FSM-to-rule alignment ──────────────────────────────────

def _align_fsm_to_rule(rule: dict, spec_ir: dict, rtl_ir: dict,
                       dep_graph: dict) -> dict:
    trigger_text    = rule["trigger"]["expression"].lower()
    obligation_text = rule["obligation"]["expression"].lower()
    spec_states     = set(spec_ir.get("state_mentions", []))

    for fsm_name, fsm in rtl_ir["fsms"].items():
        fsm_states = {s.upper() for s in fsm.get("states", [])}
        spec_up    = {s.upper() for s in spec_states}
        if fsm_states & spec_up:
            return {fsm_name: {**fsm, "matched_states": list(fsm_states & spec_up)}}

    best_fsm, best_count = None, 0
    for fsm_name, fsm in rtl_ir["fsms"].items():
        state_reg = fsm.get("state_register", "")
        cone      = backward_slice(state_reg, dep_graph, max_depth=3)
        count = sum(
            1 for s in list(rtl_ir.get("signal_to_module", {}).keys())[:50]
            if s in cone and (s in trigger_text or s in obligation_text)
        )
        if count > best_count:
            best_fsm, best_count = fsm_name, count

    if best_fsm:
        return {best_fsm: rtl_ir["fsms"][best_fsm]}
    return {}


# ── 4.3  Context builder ──────────────────────────────────────────────────────

def build_context(
    rule: dict,
    grounding_result: dict,
    rtl_ir: dict,
    spec_ir: dict,
) -> dict:
    """
    Uses budget_aware_slice() instead of fixed-depth BFS.
    """
    dep_graph  = rtl_ir.get("dep_graph", {})
    groundings = grounding_result["groundings"]

    trigger_text    = rule["trigger"]["expression"].lower()
    obligation_text = rule["obligation"]["expression"].lower()

    # Map spec signals → RTL signals for trigger and obligation
    trigger_signals    = set()
    obligation_signals = set()
    grounding_conf_map: dict[str, float] = {}   # rtl_name → confidence

    for spec_sig, g in groundings.items():
        if not g["rtl_name"]:
            continue
        grounding_conf_map[g["rtl_name"]] = g.get("confidence", 0.5)
        if spec_sig.lower() in trigger_text or any(
            t in trigger_text for t in spec_sig.lower().split("_")
        ):
            trigger_signals.add(g["rtl_name"])
        if spec_sig.lower() in obligation_text or any(
            t in obligation_text for t in spec_sig.lower().split("_")
        ):
            obligation_signals.add(g["rtl_name"])

    if not trigger_signals and not obligation_signals:
        ranked = sorted(
            [(k, v) for k, v in groundings.items() if v["rtl_name"]],
            key=lambda x: -x[1]["confidence"],
        )
        trigger_signals    = {v["rtl_name"] for _, v in ranked[:3]}
        obligation_signals = {v["rtl_name"] for _, v in ranked[3:6]}
        for _, v in ranked[:6]:
            grounding_conf_map[v["rtl_name"]] = v.get("confidence", 0.3)

    seed_signals = (trigger_signals | obligation_signals) - {None}

    # ── Budget-aware priority slice ───────────────────────────────
    slice_backward = budget_aware_slice(
        seed_signals=seed_signals,
        dep_graph=dep_graph,
        grounding_confidences=grounding_conf_map,
        token_budget=SLICE_TOKEN_BUDGET,
        min_signals=SLICE_MIN_SIGNALS,
        direction="backward",
    )
    slice_forward = budget_aware_slice(
        seed_signals=trigger_signals - {None},
        dep_graph=dep_graph,
        grounding_confidences=grounding_conf_map,
        token_budget=max(SLICE_TOKEN_BUDGET // 3, 20),
        min_signals=2,
        direction="forward",
    )
    slice_signals = slice_backward | slice_forward

    # Clock and reset cones — always included via BFS (structural requirement)
    clocks, resets = set(), set()
    for mod in rtl_ir["modules"].values():
        clocks |= {c for c in mod.get("clocks", []) if c}
        resets |= {r for r in mod.get("resets", []) if r}

    # Sanitise
    slice_signals      = {s for s in slice_signals      if s}
    trigger_signals    = {s for s in trigger_signals    if s}
    obligation_signals = {s for s in obligation_signals if s}

    context = {
        "rule": rule,
        "grounded_signals": [
            g for spec_sig, g in groundings.items()
            if (
                g["rtl_name"] in (slice_signals | trigger_signals | obligation_signals)
                or spec_sig.lower() in trigger_text
                or spec_sig.lower() in obligation_text
            )
            and g["rtl_name"]
        ],
        "rtl_slice": {
            "trigger_signals":    list(trigger_signals),
            "obligation_signals": list(obligation_signals),
            "signals":            list(slice_signals),
            "clocks":             list(clocks),
            "resets":             list(resets),
        },
        "dep_graph": dep_graph,
    }

    ctx_hash = compute_context_hash(context)
    cached   = load_context_cache(ctx_hash)
    if cached:
        return cached
    save_context_cache(ctx_hash, context)
    context["context_hash"] = ctx_hash
    return context