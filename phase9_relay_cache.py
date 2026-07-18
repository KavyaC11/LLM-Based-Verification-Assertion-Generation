"""
Relay Cache — Prompt Mapping & Caching to Reduce Network Traffic
"""

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import CACHE_DIR

_RELAY_CACHE_DIR = Path(CACHE_DIR) / "relay_cache"
_RELAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (from RelayCaching §5.5 sensitivity analysis) ─────────────
TAU_DEV         = 1.5    # deviation score multiplier (τ_diff in paper)
TAU_INF         = 1.45   # influence score multiplier (τ_down in paper)
SUFFIX_LEN      = 10     # suffix-aware tail tokens (L_suf in paper)
SIMILARITY_HIGH = 0.92   # cosine sim threshold for safe reuse (macro-level, §3.1)
SIMILARITY_MED  = 0.75   # medium sim → partial reuse with rectification
REUSE_RATE_LOG  = True   # log reuse rates for RQ2.5 analysis

# ── SVA keywords = "high-influence tokens" (influence-based selection, §4.3.2) ─
_SVA_INFLUENCE_TOKENS = {
    "posedge", "negedge", "disable", "iff", "property", "endproperty",
    "assert", "sequence", "endsequence", "|->", "|=>", "##", "always",
    "eventually", "rose", "fell", "stable", "past", "assign", "reg", "wire",
    "trigger", "obligation", "clock", "reset",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Granularity Levels — RQ2.5
# ═══════════════════════════════════════════════════════════════════════════════

class CacheGranularity:
    PROMPT    = "prompt"          # full context hash — exact match
    CHUNK     = "chunk"           # RTL snippet chunk — relay handoff analogy
    REASONING = "reasoning"       # prior LLM reasoning trace reuse


# ═══════════════════════════════════════════════════════════════════════════════
# Token-level fingerprinting 
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Simple word/symbol tokenizer."""
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|##\d+|[|][->=>]+|[^\s\w]|\d+", text.lower())


def _token_vector(tokens: list[str], vocab: dict) -> list[float]:
    """Bag-of-words frequency vector over shared vocab."""
    vec = [0.0] * len(vocab)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return round(dot, 4)


def _build_vocab(*texts: str) -> dict:
    tokens = set()
    for t in texts:
        tokens.update(_tokenize(t))
    return {tok: i for i, tok in enumerate(sorted(tokens))}


def prompt_similarity(text_a: str, text_b: str) -> float:
    """
    RQ2.4: Compute cosine similarity between two prompts.
    Analogy to RelayCaching §3.1 macro-level KV alignment check.
    Value cosine similarity is the primary deviation indicator.
    """
    vocab = _build_vocab(text_a, text_b)
    if not vocab:
        return 0.0
    toks_a = _tokenize(text_a)
    toks_b = _tokenize(text_b)
    va = _token_vector(toks_a, vocab)
    vb = _token_vector(toks_b, vocab)
    return _cosine_similarity(va, vb)


# ═══════════════════════════════════════════════════════════════════════════════
# Deviation-based token selection (RQ2.6 — §4.3.1)
# ═══════════════════════════════════════════════════════════════════════════════

def _deviation_scores(tokens_new: list[str], tokens_cached: list[str]) -> dict[str, float]:
    """
    RQ2.6: Compute per-token deviation scores.
    Tokens present in new but absent in cached (or vice-versa) = high deviation.
    Analogy to d_{j,ℓ} = 1 - cos(v_reuse, v_full) in Eq. (1).
    """
    set_new    = set(tokens_new)
    set_cached = set(tokens_cached)
    all_tokens = set_new | set_cached

    scores = {}
    for tok in all_tokens:
        in_new    = 1.0 if tok in set_new    else 0.0
        in_cached = 1.0 if tok in set_cached else 0.0
        scores[tok] = abs(in_new - in_cached)   # 0 = identical, 1 = fully new/removed

    return scores


def _high_deviation_tokens(dev_scores: dict[str, float], tau: float = TAU_DEV) -> set[str]:
    """
    RQ2.6: Select tokens whose deviation exceeds τ × mean deviation.
    Mean-relative threshold adapts to magnitude (§4.3.1, Eq. 5).
    """
    if not dev_scores:
        return set()
    mu = sum(dev_scores.values()) / len(dev_scores)
    threshold = tau * mu
    return {tok for tok, score in dev_scores.items() if score >= threshold}


# ═══════════════════════════════════════════════════════════════════════════════
# Influence-based token selection (RQ2.6 — §4.3.2)
# ═══════════════════════════════════════════════════════════════════════════════

def _influence_scores(tokens: list[str]) -> dict[str, float]:
    """
    RQ2.6: Score each token by its SVA/domain influence.
    SVA keywords = high-attention tokens that downstream generation attends to.
    Analogy to s_inf(j) = Σ α_{t,l,h,j} (Eq. 6), here approximated by
    domain-keyword membership as a proxy for attention weight.
    """
    scores = {}
    for i, tok in enumerate(tokens):
        base = 1.0 if tok in _SVA_INFLUENCE_TOKENS else 0.1
        # suffix-aware boost: last SUFFIX_LEN tokens get extra weight (§4.3.2)
        if i >= len(tokens) - SUFFIX_LEN:
            base *= 1.5
        scores[tok] = base
    return scores


def _high_influence_tokens(tokens: list[str], tau: float = TAU_INF) -> set[str]:
    """Select high-influence tokens using mean-relative threshold."""
    scores = _influence_scores(tokens)
    if not scores:
        return set()
    mu = sum(scores.values()) / len(scores)
    threshold = tau * mu
    return {tok for tok, score in scores.items() if score >= threshold}


# ═══════════════════════════════════════════════════════════════════════════════
# Rectification Set (RQ2.6 — §4.3, Eq. 9: I_final = I_dev ∪ I_inf)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rectification_set(
    tokens_new: list[str],
    tokens_cached: list[str],
) -> tuple[set[str], dict]:
    """
    RQ2.6: Identify tokens requiring rectification before serving cached result.
    Combines deviation-based (changed tokens) and influence-based (SVA keywords)
    selection — exactly RelayCaching §4.3, I_final = I_dev ∪ I_inf.
    Returns (rectification_set, debug_stats).
    """
    dev_scores  = _deviation_scores(tokens_new, tokens_cached)
    i_dev       = _high_deviation_tokens(dev_scores)
    i_inf       = _high_influence_tokens(tokens_new)
    i_final     = i_dev | i_inf

    stats = {
        "total_tokens":     len(set(tokens_new) | set(tokens_cached)),
        "i_dev_size":       len(i_dev),
        "i_inf_size":       len(i_inf),
        "i_final_size":     len(i_final),
        "rectification_pct": round(len(i_final) / max(len(set(tokens_new)), 1) * 100, 1),
    }
    return i_final, stats


# ═══════════════════════════════════════════════════════════════════════════════
# Layer-Range Profiler analogue (RQ2.5 — §4.2)
# U-shaped profile → here mapped to prompt sections:
#   shallow = system persona + rule header  (stable, low deviation)
#   middle  = RTL code snippets             (highest deviation, needs rectification)
#   deep    = timing/reset boilerplate      (partially stable)
# ═══════════════════════════════════════════════════════════════════════════════

def _split_prompt_layers(prompt: str) -> dict[str, str]:
    """
    RQ2.5: Split prompt into layer analogs.
    Shallow (persona/rule) → Middle (RTL snippets) → Deep (timing/output format).
    Middle layer = highest deviation source, needs most rectification.
    """
    layers = {"shallow": "", "middle": "", "deep": ""}

    # Extract verilog code block = "middle layer" (highest deviation)
    code_match = re.search(r"```verilog(.*?)```", prompt, re.DOTALL)
    if code_match:
        layers["middle"] = code_match.group(1).strip()
        without_code = prompt[:code_match.start()] + prompt[code_match.end():]
    else:
        without_code = prompt

    # Split remaining: before signal mappings = shallow, after = deep
    split_marker = "═══ SIGNAL MAPPINGS"
    if split_marker in without_code:
        parts = without_code.split(split_marker, 1)
        layers["shallow"] = parts[0].strip()
        layers["deep"]    = parts[1].strip()
    else:
        # Fallback: first half shallow, second half deep
        mid = len(without_code) // 2
        layers["shallow"] = without_code[:mid].strip()
        layers["deep"]    = without_code[mid:].strip()

    return layers


def layer_similarity_profile(prompt_new: str, prompt_cached: str) -> dict[str, float]:
    """
    RQ2.4 + RQ2.5: Compute per-layer similarity profile.
    Analogous to RelayCaching §3.2 U-shaped layer-wise similarity.
    Middle layer (RTL code) expected to have lowest similarity.
    """
    layers_new    = _split_prompt_layers(prompt_new)
    layers_cached = _split_prompt_layers(prompt_cached)

    profile = {}
    for layer_name in ("shallow", "middle", "deep"):
        profile[layer_name] = prompt_similarity(
            layers_new[layer_name],
            layers_cached[layer_name],
        )
    return profile


def reuse_decision(
    sim_overall: float,
    layer_profile: dict[str, float],
) -> tuple[str, str]:
    """
    RQ2.4: Decide reuse strategy based on similarity profile.
    Returns (decision, reason).

    decision:
      "full_reuse"     → serve cached result directly (sim ≥ SIMILARITY_HIGH)
      "partial_reuse"  → reuse with middle-layer rectification (SIMILARITY_MED ≤ sim < HIGH)
      "no_reuse"       → full recomputation needed (sim < SIMILARITY_MED)

    Mirrors RelayCaching's three stages:
      Skip [L0, Lstart) → Full recompute [Lstart, Ldet) → Sparse rectify [Ldet, Lend]
    """
    middle_sim = layer_profile.get("middle", 0.0)

    if sim_overall >= SIMILARITY_HIGH and middle_sim >= SIMILARITY_HIGH:
        return "full_reuse", f"overall={sim_overall:.3f} middle={middle_sim:.3f} → safe direct reuse"

    if sim_overall >= SIMILARITY_MED:
        if middle_sim < SIMILARITY_MED:
            return "partial_reuse", (
                f"overall={sim_overall:.3f} but middle={middle_sim:.3f} low → "
                f"rectify RTL snippet layer"
            )
        return "partial_reuse", (
            f"overall={sim_overall:.3f} medium → partial reuse with rectification"
        )

    return "no_reuse", f"overall={sim_overall:.3f} < {SIMILARITY_MED} → full recompute"


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Entry (multi-granularity — RQ2.5)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RelayCacheEntry:
    granularity:    str                  # CacheGranularity.*
    prompt_hash:    str                  # SHA256[:16] of the cached prompt
    prompt_text:    str                  # full prompt text (for similarity comparison)
    prompt_tokens:  list[str]            # tokenized prompt
    result:         dict                 # cached assertion or reasoning trace
    tier1_passed:   bool                 # validation status at cache time
    confidence:     int                  # LLM confidence at cache time
    rule_id:        str
    design_hash:    str
    timestamp:      float = field(default_factory=time.time)
    hit_count:      int   = 0
    reuse_rate:     float = 0.0          # fraction of tokens reused vs recomputed

    def to_dict(self) -> dict:
        return {
            "granularity":   self.granularity,
            "prompt_hash":   self.prompt_hash,
            "prompt_text":   self.prompt_text[:500],   # store truncated for disk
            "prompt_tokens": self.prompt_tokens[:200], # store truncated
            "result":        self.result,
            "tier1_passed":  self.tier1_passed,
            "confidence":    self.confidence,
            "rule_id":       self.rule_id,
            "design_hash":   self.design_hash,
            "timestamp":     self.timestamp,
            "hit_count":     self.hit_count,
            "reuse_rate":    self.reuse_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelayCacheEntry":
        return cls(
            granularity   = d["granularity"],
            prompt_hash   = d["prompt_hash"],
            prompt_text   = d.get("prompt_text", ""),
            prompt_tokens = d.get("prompt_tokens", []),
            result        = d["result"],
            tier1_passed  = d["tier1_passed"],
            confidence    = d.get("confidence", 0),
            rule_id       = d["rule_id"],
            design_hash   = d["design_hash"],
            timestamp     = d.get("timestamp", 0.0),
            hit_count     = d.get("hit_count", 0),
            reuse_rate    = d.get("reuse_rate", 0.0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Relay Cache Store
# ═══════════════════════════════════════════════════════════════════════════════

class RelayCache:
    """
    Multi-granularity prompt cache with RelayCaching-inspired similarity
    gating and validation.

    RQ2.4: Similarity-gated lookup (macro + layer profile)
    RQ2.5: Three granularity levels (prompt / chunk / reasoning)
    RQ2.6: Deviation+influence rectification set + tier-1 re-validation
    """

    def __init__(self, design_hash: str):
        self.design_hash = design_hash
        self._store_path = _RELAY_CACHE_DIR / f"relay_{design_hash}.json"
        self._entries: list[RelayCacheEntry] = []
        self._stats = {
            "total_lookups": 0,
            "full_reuse_hits": 0,
            "partial_reuse_hits": 0,
            "misses": 0,
            "validation_failures": 0,
            "tokens_saved": 0,
            "tokens_recomputed": 0,
        }
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if self._store_path.exists():
            try:
                raw = json.loads(self._store_path.read_text())
                self._entries = [RelayCacheEntry.from_dict(e) for e in raw.get("entries", [])]
                self._stats   = raw.get("stats", self._stats)
            except Exception:
                self._entries = []

    def _save(self):
        self._store_path.write_text(json.dumps({
            "design_hash": self.design_hash,
            "entries": [e.to_dict() for e in self._entries],
            "stats": self._stats,
        }, indent=2, default=str))

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(
        self,
        prompt: str,
        result: dict,
        rule_id: str,
        granularity: str = CacheGranularity.PROMPT,
    ) -> str:
        """Store a validated result. Only store tier1-passed results (§L3 analogy)."""
        if not result.get("tier1_passed"):
            return ""   # never cache failed assertions (mirrors L3 cache policy)

        tokens    = _tokenize(prompt)
        phash     = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        # Remove old entry for same rule+granularity
        self._entries = [
            e for e in self._entries
            if not (e.rule_id == rule_id and e.granularity == granularity)
        ]

        entry = RelayCacheEntry(
            granularity  = granularity,
            prompt_hash  = phash,
            prompt_text  = prompt,
            prompt_tokens= tokens,
            result       = result,
            tier1_passed = True,
            confidence   = result.get("confidence", 0),
            rule_id      = rule_id,
            design_hash  = self.design_hash,
        )
        self._entries.append(entry)
        self._save()
        return phash

    # ── Lookup — RQ2.4 ────────────────────────────────────────────────────────

    def lookup(
        self,
        prompt: str,
        rule_id: str,
        granularity: str = CacheGranularity.PROMPT,
    ) -> Optional[dict]:
        """
        RQ2.4: Find best matching cached entry.
        1. Exact hash match → full_reuse
        2. Cosine similarity ≥ threshold + layer profile → partial_reuse
        3. Below threshold → miss
        """
        self._stats["total_lookups"] += 1
        tokens_new = _tokenize(prompt)
        phash_new  = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        # Filter to same granularity
        candidates = [e for e in self._entries if e.granularity == granularity]
        if not candidates:
            self._stats["misses"] += 1
            return None

        best_entry  = None
        best_sim    = -1.0
        best_decision = "no_reuse"
        best_layer_profile = {}

        for entry in candidates:
            # Exact hash match (same rule_id shortcut)
            if entry.prompt_hash == phash_new:
                best_entry     = entry
                best_sim       = 1.0
                best_decision  = "full_reuse"
                best_layer_profile = {"shallow": 1.0, "middle": 1.0, "deep": 1.0}
                break

            # Cosine similarity check
            sim = prompt_similarity(prompt, entry.prompt_text)
            if sim > best_sim:
                best_sim    = sim
                best_entry  = entry
                layer_profile = layer_similarity_profile(prompt, entry.prompt_text)
                decision, _   = reuse_decision(sim, layer_profile)
                best_decision = decision
                best_layer_profile = layer_profile

        if best_entry is None or best_decision == "no_reuse":
            self._stats["misses"] += 1
            return None

        # RQ2.6: Validate cached result before serving
        validated = self._validate(
            entry         = best_entry,
            tokens_new    = tokens_new,
            decision      = best_decision,
            layer_profile = best_layer_profile,
        )

        if validated is None:
            self._stats["validation_failures"] += 1
            self._stats["misses"] += 1
            return None

        # Update stats
        best_entry.hit_count += 1
        if best_decision == "full_reuse":
            self._stats["full_reuse_hits"] += 1
            self._stats["tokens_saved"] += len(tokens_new)
        else:
            self._stats["partial_reuse_hits"] += 1
            rect_set, rect_stats = compute_rectification_set(
                tokens_new, best_entry.prompt_tokens
            )
            recomputed = rect_stats["i_final_size"]
            saved      = len(tokens_new) - recomputed
            self._stats["tokens_saved"]      += max(saved, 0)
            self._stats["tokens_recomputed"] += recomputed
            validated["relay_rectification"] = rect_stats

        validated["relay_decision"]      = best_decision
        validated["relay_similarity"]    = best_sim
        validated["relay_layer_profile"] = best_layer_profile
        validated["relay_from_cache"]    = True

        self._save()
        return validated

    # ── Validation — RQ2.6 ────────────────────────────────────────────────────

    def _validate(
        self,
        entry: RelayCacheEntry,
        tokens_new: list[str],
        decision: str,
        layer_profile: dict[str, float],
    ) -> Optional[dict]:
        """
        RQ2.6: Validate cached result before serving.
        Three-stage validation mirroring RelayCaching's rectification pipeline:
          Stage 1 — Tier-1 syntax re-check (always)
          Stage 2 — Deviation+influence rectification set assessment
          Stage 3 — Middle-layer (RTL snippet) deviation check
        """
        result = dict(entry.result)

        # Stage 1: Tier-1 syntax re-check (non-negotiable)
        from phase6_generate import _tier1_syntax_check
        code = result.get("property_code", "") + " " + result.get("assert_statement", "")
        t1_errs = _tier1_syntax_check(code)
        if t1_errs:
            # Cached result has syntax errors — do not serve
            return None

        # Stage 2: Deviation + influence rectification assessment
        rect_set, rect_stats = compute_rectification_set(
            tokens_new, entry.prompt_tokens
        )

        # If rectification set is too large relative to new tokens, refuse reuse
        # (analogy: if too many tokens deviate, full recompute is safer)
        rect_pct = rect_stats["rectification_pct"]
        if decision == "full_reuse" and rect_pct > 15.0:
            # Silently downgrade to partial reuse check
            decision = "partial_reuse"

        if decision == "partial_reuse" and rect_pct > 50.0:
            # More than 50% of tokens flagged → not safe to reuse
            return None

        # Stage 3: Middle-layer (RTL code) similarity gate
        middle_sim = layer_profile.get("middle", 0.0)
        if middle_sim < SIMILARITY_MED and decision == "full_reuse":
            return None

        # Check that high-influence SVA tokens from cached result still appear
        # in the new prompt (ensures obligation/trigger signals still present)
        cached_sva_tokens = {
            t for t in entry.prompt_tokens if t in _SVA_INFLUENCE_TOKENS
        }
        new_tokens_set = set(tokens_new)
        missing_critical = cached_sva_tokens - new_tokens_set
        if len(missing_critical) > 3:
            # Too many critical SVA tokens missing → context has changed too much
            return None

        # Annotate with rectification metadata for downstream logging
        result["relay_validated"]     = True
        result["relay_rect_pct"]      = rect_pct
        result["relay_missing_tokens"] = list(missing_critical)
        return result

    # ── Chunk-level cache — RQ2.5 ─────────────────────────────────────────────

    def store_chunk(self, chunk_text: str, chunk_hash: str, chunk_result: dict) -> bool:
        """
        RQ2.5: Cache at RTL snippet chunk granularity.
        Analogous to RelayCaching relay handoff: reuse decoding KV of upstream
        agent (RTL chunk processed by prior rule) in downstream prefill.
        """
        chunk_entry = RelayCacheEntry(
            granularity   = CacheGranularity.CHUNK,
            prompt_hash   = chunk_hash,
            prompt_text   = chunk_text,
            prompt_tokens = _tokenize(chunk_text),
            result        = chunk_result,
            tier1_passed  = True,
            confidence    = chunk_result.get("relevance_score", 50),
            rule_id       = f"chunk_{chunk_hash}",
            design_hash   = self.design_hash,
        )
        self._entries.append(chunk_entry)
        self._save()
        return True

    def lookup_chunk(self, chunk_text: str) -> Optional[dict]:
        """
        RQ2.4 + RQ2.5: Find a reusable chunk with similarity gating.
        High similarity → relay handoff (reuse cached processing).
        """
        candidates = [e for e in self._entries if e.granularity == CacheGranularity.CHUNK]
        best_sim, best_entry = -1.0, None
        for e in candidates:
            sim = prompt_similarity(chunk_text, e.prompt_text)
            if sim > best_sim:
                best_sim, best_entry = sim, e

        if best_entry is None or best_sim < SIMILARITY_HIGH:
            return None

        result = dict(best_entry.result)
        result["chunk_relay_similarity"] = best_sim
        result["chunk_relay_reused"]     = True
        return result

    # ── Reasoning-trace cache — RQ2.5 ─────────────────────────────────────────

    def store_reasoning(self, rule_id: str, prompt: str, reasoning_trace: str,
                        final_result: dict) -> bool:
        """
        RQ2.5: Cache at reasoning-trace granularity.
        The full chain-of-thought from a prior LLM call is stored.
        Downstream calls with similar context can reuse the reasoning prefix.
        """
        if not final_result.get("tier1_passed"):
            return False
        entry = RelayCacheEntry(
            granularity   = CacheGranularity.REASONING,
            prompt_hash   = hashlib.sha256(prompt.encode()).hexdigest()[:16],
            prompt_text   = prompt,
            prompt_tokens = _tokenize(prompt + " " + reasoning_trace),
            result        = {"reasoning_trace": reasoning_trace, **final_result},
            tier1_passed  = True,
            confidence    = final_result.get("confidence", 0),
            rule_id       = rule_id,
            design_hash   = self.design_hash,
        )
        self._entries = [e for e in self._entries
                         if not (e.rule_id == rule_id and
                                 e.granularity == CacheGranularity.REASONING)]
        self._entries.append(entry)
        self._save()
        return True

    def lookup_reasoning(self, prompt: str) -> Optional[str]:
        """
        RQ2.4 + RQ2.5: Retrieve a cached reasoning trace for prompt continuation.
        Returns the reasoning string if similarity is sufficient, else None.
        Middle-layer (RTL code) deviation is the key gating criterion.
        """
        candidates = [e for e in self._entries
                      if e.granularity == CacheGranularity.REASONING]
        best_sim, best_entry = -1.0, None
        best_layer = {}
        for e in candidates:
            sim = prompt_similarity(prompt, e.prompt_text)
            if sim > best_sim:
                best_sim, best_entry = sim, e
                best_layer = layer_similarity_profile(prompt, e.prompt_text)

        if best_entry is None:
            return None

        decision, reason = reuse_decision(best_sim, best_layer)
        if decision == "no_reuse":
            return None

        # For reasoning traces, require higher bar — middle layer must be similar
        if best_layer.get("middle", 0.0) < SIMILARITY_HIGH:
            return None

        return best_entry.result.get("reasoning_trace")

    # ── Stats / reporting ─────────────────────────────────────────────────────

    def reuse_rate(self) -> float:
        """
        RQ2.5: Compute overall KV/token reuse rate (mirrors RelayCaching §5.2).
        Reuse rate = saved / (saved + recomputed).
        """
        saved      = self._stats["tokens_saved"]
        recomputed = self._stats["tokens_recomputed"]
        total      = saved + recomputed
        return round(saved / total, 3) if total > 0 else 0.0

    def stats_summary(self) -> dict:
        s = dict(self._stats)
        total = max(s["total_lookups"], 1)
        s["full_reuse_rate"]    = round(s["full_reuse_hits"]    / total, 3)
        s["partial_reuse_rate"] = round(s["partial_reuse_hits"] / total, 3)
        s["miss_rate"]          = round(s["misses"]             / total, 3)
        s["token_reuse_rate"]   = self.reuse_rate()
        s["cached_entries"]     = len(self._entries)
        return s


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level convenience API (used by phase6_generate and main)
# ═══════════════════════════════════════════════════════════════════════════════

_cache_registry: dict[str, RelayCache] = {}


def get_relay_cache(design_hash: str) -> RelayCache:
    if design_hash not in _cache_registry:
        _cache_registry[design_hash] = RelayCache(design_hash)
    return _cache_registry[design_hash]


def relay_lookup(
    prompt: str,
    rule_id: str,
    design_hash: str,
    granularity: str = CacheGranularity.PROMPT,
) -> Optional[dict]:
    """Top-level lookup: returns cached result dict or None."""
    return get_relay_cache(design_hash).lookup(prompt, rule_id, granularity)


def relay_store(
    prompt: str,
    result: dict,
    rule_id: str,
    design_hash: str,
    granularity: str = CacheGranularity.PROMPT,
) -> str:
    """Top-level store: persists validated result."""
    return get_relay_cache(design_hash).store(prompt, result, rule_id, granularity)


def relay_store_reasoning(
    rule_id: str,
    prompt: str,
    reasoning: str,
    result: dict,
    design_hash: str,
) -> bool:
    return get_relay_cache(design_hash).store_reasoning(rule_id, prompt, reasoning, result)


def relay_lookup_reasoning(prompt: str, design_hash: str) -> Optional[str]:
    return get_relay_cache(design_hash).lookup_reasoning(prompt)


def relay_stats(design_hash: str) -> dict:
    return get_relay_cache(design_hash).stats_summary()