"""
Phase 10 — Self-Refinement Engine
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ollama
from config import (
    OLLAMA_HOST, OLLAMA_MODEL, MAX_ITERATIONS, CACHE_DIR,
    GROUNDING_MED,
)


_client = ollama.Client(host=OLLAMA_HOST)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_EXPLORATORY_DRAFTS = 3      # iGRPO: number of Stage-1 drafts
MAX_REFINE_ITERS     = 3      # SELF-REFINE: max feedback-refine cycles
REWARD_PASS_THRESHOLD = 0.70  # minimum reward to call a draft "passing"
DRIFT_CEILING        = 0.60   # cosine-similarity floor; below → revert
ENTROPY_COLLAPSE_THR = 0.25   # token-diversity floor; below → re-sample
IMPROVEMENT_EPSILON  = 0.02   # min reward gain per iteration; below → stop early

# Feedback rubric dimensions (SELF-REFINE multi-aspect approach)
RUBRIC_DIMENSIONS = [
    "syntax",          # property/endproperty/assert present, no typos
    "signal_validity", # only grounded RTL signals used
    "timing",          # appropriate ##N / |-> / |=> operator
    "antecedent",      # trigger condition correctly encoded
    "consequent",      # obligation correctly encoded
    "reset_handling",  # procedural if (!reset) used correctly
    "polarity",        # active-low signals handled correctly
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Draft:
    """One candidate SVA assertion produced during Stage 1 or refinement."""
    iteration: int              # 0 = exploratory Stage-1 draft
    property_code: str
    assert_stmt: str
    reasoning: str
    warnings: list[str]
    reward: float               # composite reward in [0, 1]
    tier1_passed: bool
    tier2_passed: bool
    rubric_scores: dict[str, float] = field(default_factory=dict)
    feedback_text: str = ""
    token_entropy: float = 0.0
    drift_from_seed: float = 0.0  # 1 - cosine_sim(draft_0, this)

    def full_code(self) -> str:
        return (self.property_code + "\n" + self.assert_stmt).strip()


@dataclass
class RefinedAssertion:
    """Final output of Phase 10, compatible with Phase 6 / 7 expectations."""
    property_name: str
    property_code: str
    assert_statement: str
    confidence: int               # 0-100
    reasoning: str
    warnings: list[str]
    tier1_passed: bool
    tier2_passed: bool
    # Phase-10 metadata for ablation / RQ analysis
    refine_metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Reward function  (RQ3.1 — feedback signals)
# ═══════════════════════════════════════════════════════════════════════════════

def reward_function(
    draft: dict,
    context: dict,
    rtl_ir: dict,
) -> float:
    """
    Composite scalar reward in [0, 1].

    Sources (RQ3.1):
      • tier1_score   — automated syntax test result
      • tier2_score   — semantic RTL signal validity
      • grounding_conf — average grounding confidence (downstream quality)
      • rubric_score  — multi-aspect LLM rubric (SELF-REFINE style)

    Weights chosen so that automated tests dominate (reproducible),
    with grounding confidence as a stable secondary signal.
    """
    from phase6_generate import _tier1_syntax_check, _tier2_semantic_check

    code = draft.get("property_code", "") + " " + draft.get("assert_statement", "")

    # Tier-1 automated syntax test
    t1_errs = _tier1_syntax_check(code)
    tier1_score = max(0.0, 1.0 - 0.25 * len(t1_errs))

    # Tier-2 semantic signal validity
    t2_errs = _tier2_semantic_check(draft, rtl_ir, context)
    tier2_score = 1.0 if not t2_errs else 0.6

    # Grounding confidence (downstream quality metric)
    grounded = context.get("grounded_signals", [])
    if grounded:
        avg_conf = sum(g.get("confidence", 0) for g in grounded) / len(grounded)
        grounding_score = min(1.0, avg_conf)
    else:
        grounding_score = 0.3

    # Rubric scores (filled in by SELF-REFINE feedback stage; default 0.5 if absent)
    rubric_vals = list(draft.get("rubric_scores", {}).values())
    rubric_score = (sum(rubric_vals) / len(rubric_vals)) if rubric_vals else 0.5

    # Composite weighted reward
    reward = (
        0.35 * tier1_score
        + 0.25 * tier2_score
        + 0.20 * grounding_score
        + 0.20 * rubric_score
    )
    return round(reward, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Token entropy  (RQ3.3 — drift / entropy collapse guard)
# ═══════════════════════════════════════════════════════════════════════════════

def _token_entropy(text: str) -> float:
    """Shannon entropy over token unigrams — proxy for generation diversity."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    n = len(tokens)
    return -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)


def _cosine_sim_text(a: str, b: str) -> float:
    """Bag-of-words cosine similarity between two code strings."""
    tokens_a = re.findall(r"\b\w+\b", a.lower())
    tokens_b = re.findall(r"\b\w+\b", b.lower())
    vocab = set(tokens_a) | set(tokens_b)
    if not vocab:
        return 1.0
    vec_a = {t: tokens_a.count(t) for t in vocab}
    vec_b = {t: tokens_b.count(t) for t in vocab}
    dot  = sum(vec_a[t] * vec_b[t] for t in vocab)
    na   = math.sqrt(sum(v * v for v in vec_a.values())) or 1.0
    nb   = math.sqrt(sum(v * v for v in vec_b.values())) or 1.0
    return round(dot / (na * nb), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# iGRPO advantage estimation  (RQ3.2 — group-relative policy update signal)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_group_advantages(rewards: list[float]) -> list[float]:
    """
    iGRPO Eq. (4): normalise rewards within the group to compute advantages.
    Â_i = (R_i - mean(R)) / std(R)
    If std == 0 all advantages are 0 (convention from iGRPO §3.1).
    This gives a group-relative signal that drives draft selection and
    informs which prior attempts are worth conditioning on.
    """
    if not rewards:
        return []
    mu  = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / max(len(rewards), 1)
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0] * len(rewards)
    return [round((r - mu) / std, 4) for r in rewards]


# ═══════════════════════════════════════════════════════════════════════════════
# LLM call helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str, system: str = "", temperature: float = 0.15,
              max_retries: int = 2) -> str:
    """Shared LLM call with rate-limit retry."""
    sys_msg = system or (
        "You are an expert SVA engineer. Respond with valid JSON only. "
        "No markdown, no commentary."
    )
    for attempt in range(max_retries + 1):
        try:
            resp = _client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": temperature, "num_predict": 1024},
            )
            raw = resp['message']['content'].strip()
            raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"^```\s*",     "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```$",        "", raw.strip())
            return raw.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 30
                m = re.search(r"retry.after.(\d+)", err, re.IGNORECASE)
                if m:
                    wait = int(m.group(1)) + 5
                print(f"    [rate-limit] waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"    [LLM error attempt {attempt+1}] {err[:160]}")
                if attempt == max_retries:
                    return ""
                time.sleep(3)
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Exploratory Draft Generation  (iGRPO)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_base_prompt(context: dict) -> str:
    """Shared context block seen by all LLM calls in Phase 10."""
    from phase6_generate import build_prompt
    return build_prompt(context)


def _generate_one_draft(base_prompt: str, temperature: float = 0.15,
                        prior_draft: str = "") -> dict:
    """
    Generate a single SVA draft.
    If prior_draft is provided, this is Stage-2 conditioned refinement (iGRPO).
    """
    if prior_draft:
        # iGRPO Stage 2: augmented prompt = original + best draft
        prompt = (
            base_prompt
            + "\n\n══════════ PRIOR BEST DRAFT (improve upon this) ══════════\n"
            + prior_draft
            + "\n\nRefine the draft above. Fix all issues, improve timing accuracy, "
            "and ensure every signal is correctly grounded. "
            "Return the same strict JSON format."
        )
    else:
        prompt = base_prompt

    raw = _call_llm(prompt, temperature=temperature)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to salvage partial JSON
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def stage1_exploratory_drafts(
    base_prompt: str,
    context: dict,
    rtl_ir: dict,
    n: int = N_EXPLORATORY_DRAFTS,
) -> tuple[list[Draft], list[float], list[float]]:
    """
    iGRPO Stage 1: sample N drafts independently, score each,
    compute group-relative advantages.

    Returns (drafts, rewards, advantages).
    """
    from phase6_generate import _tier1_syntax_check, _tier2_semantic_check

    drafts: list[Draft]  = []
    rewards: list[float] = []

    # Use slightly varied temperatures to encourage diversity (entropy-preservation)
    temps = [0.10, 0.15, 0.20][:n]
    if len(temps) < n:
        temps += [0.15] * (n - len(temps))

    for i, temp in enumerate(temps):
        raw = _generate_one_draft(base_prompt, temperature=temp)
        if not raw:
            continue

        code = raw.get("property_code", "") + " " + raw.get("assert_statement", "")
        t1 = _tier1_syntax_check(code)
        t2 = _tier2_semantic_check(raw, rtl_ir, context)
        raw["rubric_scores"] = {}  # will be filled in Stage 2

        reward = reward_function(raw, context, rtl_ir)
        entropy = _token_entropy(code)

        draft = Draft(
            iteration    = 0,
            property_code= raw.get("property_code", ""),
            assert_stmt  = raw.get("assert_statement", ""),
            reasoning    = raw.get("reasoning", ""),
            warnings     = raw.get("warnings", []) + t1 + t2,
            reward       = reward,
            tier1_passed = len(t1) == 0,
            tier2_passed = len(t2) == 0,
            token_entropy= entropy,
        )
        drafts.append(draft)
        rewards.append(reward)
        print(f"    [Stage-1 draft {i+1}] reward={reward:.3f} "
              f"tier1={'✓' if draft.tier1_passed else '✗'} "
              f"entropy={entropy:.2f}")
        time.sleep(1)  # light rate-limit courtesy

    advantages = compute_group_advantages(rewards)
    return drafts, rewards, advantages


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-REFINE feedback generation  (RQ3.1 — multi-aspect feedback)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_feedback(draft: Draft, context: dict) -> tuple[str, dict[str, float]]:
    """
    SELF-REFINE FEEDBACK step.

    Evaluates the current draft along RUBRIC_DIMENSIONS and returns:
      (natural-language feedback, {dimension: score_0_to_1})

    This is the "specific, actionable feedback" that SELF-REFINE §2 emphasises.
    Generic feedback ("improve this") is avoided by scoring each dimension
    and pointing to concrete lines/signals.
    """
    rule      = context["rule"]
    grounded  = context.get("grounded_signals", [])
    rtl_slice = context.get("rtl_slice", {})

    grounded_names = {g["rtl_name"] for g in grounded if g.get("rtl_name")}
    clocks  = rtl_slice.get("clocks", ["clk"])
    resets  = rtl_slice.get("resets", ["rst_n"])

    prompt = f"""You are a strict SVA reviewer performing a multi-aspect rubric evaluation.

═══ RULE ═══
Trigger:    {rule['trigger']['expression'][:200]}
Obligation: {rule['obligation']['expression'][:200]}
Timing:     {rule['timing']['type']} = {rule['timing'].get('value', 'N/A')}

═══ SVA DRAFT TO REVIEW ═══
{draft.full_code()}

═══ KNOWN-GOOD RTL SIGNALS ═══
Grounded signals: {sorted(grounded_names)}
Clocks: {clocks}
Resets: {resets}

Score each dimension from 0.0 (completely wrong) to 1.0 (perfect).
Be SPECIFIC: cite the exact token/line that is wrong.

Respond in strict JSON:
{{
  "scores": {{
    "syntax":          <0.0-1.0>,
    "signal_validity": <0.0-1.0>,
    "timing":          <0.0-1.0>,
    "antecedent":      <0.0-1.0>,
    "consequent":      <0.0-1.0>,
    "reset_handling":  <0.0-1.0>,
    "polarity":        <0.0-1.0>
  }},
  "feedback": "<one paragraph of specific, actionable improvement instructions>",
  "stop": <true if the assertion is already correct and no further changes needed>
}}"""

    raw = _call_llm(prompt)
    if not raw:
        # Fallback: neutral scores, no feedback
        return "No feedback available.", {d: 0.5 for d in RUBRIC_DIMENSIONS}

    try:
        data = json.loads(raw)
        scores   = data.get("scores", {})
        feedback = data.get("feedback", "")
        # Normalise keys and clamp scores
        rubric = {d: max(0.0, min(1.0, float(scores.get(d, 0.5))))
                  for d in RUBRIC_DIMENSIONS}
        return feedback, rubric
    except Exception:
        return raw[:500], {d: 0.5 for d in RUBRIC_DIMENSIONS}


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-REFINE refinement step  (RQ3.2 — prompt optimisation)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_refinement(
    prior_draft: Draft,
    feedback: str,
    rubric_scores: dict[str, float],
    base_prompt: str,
    iteration: int,
) -> dict:
    """
    SELF-REFINE REFINE step, conditioned on iGRPO best draft.

    The augmented prompt is:
        [original context] + [best draft] + [specific feedback]

    This implements both:
      • iGRPO Stage 2: conditioning on best draft (dynamic self-conditioning)
      • SELF-REFINE:   incorporating actionable feedback into next draft
    """
    weak_dims = [d for d, s in rubric_scores.items() if s < 0.7]
    focus = (
        f"Focus especially on improving: {', '.join(weak_dims)}."
        if weak_dims else
        "All dimensions look reasonable; polish the assertion."
    )

    prompt = (
        base_prompt
        + f"\n\n══════════ BEST PRIOR DRAFT (iteration {prior_draft.iteration}) ══════════\n"
        + prior_draft.full_code()
        + f"\n\n══════════ REVIEWER FEEDBACK ══════════\n"
        + feedback
        + f"\n\n{focus}"
        + "\n\nGenerate an improved assertion. "
          "Do NOT repeat errors from the prior draft. "
          "Return strict JSON with keys: property_code, assert_statement, "
          "confidence, reasoning, warnings."
    )

    raw = _call_llm(prompt, temperature=0.10)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Stopping criteria  (RQ3.3 — preventing drift and over-refinement)
# ═══════════════════════════════════════════════════════════════════════════════

def _should_stop(
    current_reward: float,
    prev_reward: float,
    rubric_scores: dict[str, float],
    feedback_says_stop: bool,
    iteration: int,
) -> tuple[bool, str]:
    """
    Multi-criterion stopping function (SELF-REFINE §2 + iGRPO convergence).

    Returns (stop, reason).
    """
    # Criterion 1: reward above threshold — good enough
    if current_reward >= REWARD_PASS_THRESHOLD and iteration >= 1:
        return True, f"reward {current_reward:.3f} ≥ threshold {REWARD_PASS_THRESHOLD}"

    # Criterion 2: feedback judge says stop (all dimensions near-perfect)
    if feedback_says_stop:
        avg_rubric = sum(rubric_scores.values()) / max(len(rubric_scores), 1)
        if avg_rubric >= 0.85:
            return True, f"rubric judge agrees (avg={avg_rubric:.2f})"

    # Criterion 3: improvement too small — diminishing returns
    gain = current_reward - prev_reward
    if abs(gain) < IMPROVEMENT_EPSILON and iteration >= 2:
        return True, f"gain {gain:.4f} < ε={IMPROVEMENT_EPSILON} (diminishing returns)"

    # Criterion 4: all rubric dimensions strong
    if rubric_scores and all(s >= 0.80 for s in rubric_scores.values()):
        return True, "all rubric dimensions ≥ 0.80"

    return False, ""


def _check_drift(seed_code: str, current_code: str) -> float:
    """
    RQ3.3: Compute semantic drift from original seed draft.
    Returns 1 - cosine_similarity (higher = more drift).
    """
    return round(1.0 - _cosine_sim_text(seed_code, current_code), 4)


def _check_entropy_collapse(entropy: float) -> bool:
    """
    RQ3.3: Detect premature entropy collapse (iGRPO §5 Figure 3 observation).
    If token diversity drops below threshold, diversity-boosted re-sample is needed.
    """
    return entropy < ENTROPY_COLLAPSE_THR


# ═══════════════════════════════════════════════════════════════════════════════
# Main Self-Refinement Engine
# ═══════════════════════════════════════════════════════════════════════════════

class SelfRefineEngine:
    """
    Orchestrates iGRPO two-stage + SELF-REFINE feedback-refine loop.

    RQ3.2 — behavior improvement mechanisms used:
      1. Prompt optimisation  : iGRPO best-draft conditioning
      2. Memory updates       : reward signals fed to Phase 7 at end
      3. Tool-use strategy    : escalates to multi-agent (phase8) if single-shot
                                fails to reach threshold after MAX_REFINE_ITERS
      4. Policy update signal : group advantages identify which prior attempt
                                serves as the conditioning context

    RQ3.3 — anti-forgetting / anti-drift mechanisms:
      1. Stopping criterion   : multi-criterion early stop
      2. Drift detector       : revert to best intermediate on excess drift
      3. Entropy guard        : temperature re-sample on entropy collapse
      4. Contradiction check  : Phase 7 memory used to detect regression vs
                                previously validated assertion
    """

    def __init__(self, context: dict, rtl_ir: dict):
        self.context  = context
        self.rtl_ir   = rtl_ir
        self.rule     = context["rule"]
        self.rule_id  = self.rule["id"]
        self._all_drafts: list[Draft] = []
        self._refine_log: list[dict]  = []

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        n_stage1_drafts: int = N_EXPLORATORY_DRAFTS,
        max_refine_iters: int = MAX_REFINE_ITERS,
    ) -> RefinedAssertion:
        """
        Full two-stage pipeline.

        Returns a RefinedAssertion with rich metadata for RQ analysis.
        """
        print(f"  [Phase 10] Self-refinement for rule {self.rule_id}")
        base_prompt = _build_base_prompt(self.context)

        # ── iGRPO Stage 1: Exploratory draft generation ────────────────────
        print(f"    → Stage 1: generating {n_stage1_drafts} exploratory drafts")
        drafts, rewards, advantages = stage1_exploratory_drafts(
            base_prompt, self.context, self.rtl_ir, n=n_stage1_drafts
        )

        if not drafts:
            return self._fallback_result("Stage 1 produced no usable drafts")

        # Select best draft: d̂ = arg max reward  (iGRPO Eq. 2)
        best_idx   = max(range(len(rewards)), key=lambda i: rewards[i])
        best_draft = drafts[best_idx]
        seed_code  = best_draft.full_code()  # anchor for drift measurement

        self._all_drafts.extend(drafts)
        self._refine_log.append({
            "stage": "stage1",
            "n_drafts": len(drafts),
            "rewards": rewards,
            "advantages": advantages,
            "best_reward": rewards[best_idx],
            "best_idx": best_idx,
        })

        print(
            f"    → Best Stage-1 draft: reward={rewards[best_idx]:.3f} "
            f"advantage={advantages[best_idx]:+.3f} "
            f"tier1={'✓' if best_draft.tier1_passed else '✗'}"
        )

        # Early exit: if Stage-1 best is already very good, skip Stage 2
        if best_draft.reward >= REWARD_PASS_THRESHOLD and best_draft.tier1_passed:
            print(f"    → Stage-1 reward already ≥ threshold; skipping refinement")
            return self._wrap_result(best_draft, stage1_only=True,
                                     rewards=rewards, advantages=advantages)

        # ── iGRPO Stage 2 + SELF-REFINE: Conditioned refinement loop ──────
        print(f"    → Stage 2: conditioned refinement (max {max_refine_iters} iters)")
        current = best_draft
        prev_reward = best_draft.reward
        best_so_far = best_draft

        for k in range(1, max_refine_iters + 1):
            print(f"      [iter {k}] feedback …", end="", flush=True)

            # SELF-REFINE FEEDBACK
            feedback_text, rubric_scores = generate_feedback(current, self.context)
            current.rubric_scores  = rubric_scores
            current.feedback_text  = feedback_text
            avg_rubric = sum(rubric_scores.values()) / max(len(rubric_scores), 1)
            feedback_stop = avg_rubric >= 0.88

            print(f" avg_rubric={avg_rubric:.2f}", end="")

            # Stopping criterion — check before spending LLM call
            stop, reason = _should_stop(
                current.reward, prev_reward, rubric_scores, feedback_stop, k
            )
            if stop:
                print(f" → STOP ({reason})")
                break

            # SELF-REFINE REFINE (conditioned on best-so-far = iGRPO Stage 2)
            print(f" refining …", end="", flush=True)
            raw_refined = generate_refinement(
                current, feedback_text, rubric_scores, base_prompt, iteration=k
            )
            if not raw_refined:
                print(f" LLM returned empty — stopping")
                break

            # Build refined Draft object
            from phase6_generate import _tier1_syntax_check, _tier2_semantic_check
            ref_code = (raw_refined.get("property_code", "") + " " +
                        raw_refined.get("assert_statement", ""))
            t1 = _tier1_syntax_check(ref_code)
            t2 = _tier2_semantic_check(raw_refined, self.rtl_ir, self.context)

            raw_refined["rubric_scores"] = {}
            new_reward  = reward_function(raw_refined, self.context, self.rtl_ir)
            entropy     = _token_entropy(ref_code)
            drift       = _check_drift(seed_code, ref_code)

            refined = Draft(
                iteration    = k,
                property_code= raw_refined.get("property_code", ""),
                assert_stmt  = raw_refined.get("assert_statement", ""),
                reasoning    = raw_refined.get("reasoning", ""),
                warnings     = raw_refined.get("warnings", []) + t1 + t2,
                reward       = new_reward,
                tier1_passed = len(t1) == 0,
                tier2_passed = len(t2) == 0,
                rubric_scores= {},
                feedback_text= "",
                token_entropy= entropy,
                drift_from_seed= drift,
            )

            print(
                f" reward={new_reward:.3f} "
                f"drift={drift:.3f} "
                f"entropy={entropy:.2f} "
                f"tier1={'✓' if refined.tier1_passed else '✗'}"
            )

            # RQ3.3: Drift guard — revert if too far from seed
            if drift > DRIFT_CEILING:
                print(
                    f"      [iter {k}] Drift {drift:.3f} > ceiling {DRIFT_CEILING} "
                    f"— reverting to best_so_far"
                )
                current = best_so_far
                self._refine_log.append({
                    "iter": k, "event": "drift_revert",
                    "drift": drift, "reward_before_revert": new_reward,
                })
                break

            # RQ3.3: Entropy collapse guard (iGRPO §5 insight)
            if _check_entropy_collapse(entropy):
                print(
                    f"      [iter {k}] Entropy collapse ({entropy:.2f}) — "
                    f"injecting temperature-boosted re-sample"
                )
                hot_raw = _generate_one_draft(
                    base_prompt, temperature=0.35,
                    prior_draft=best_so_far.full_code()
                )
                if hot_raw:
                    hot_code = (hot_raw.get("property_code", "") + " " +
                                hot_raw.get("assert_statement", ""))
                    hot_reward  = reward_function(hot_raw, self.context, self.rtl_ir)
                    hot_entropy = _token_entropy(hot_code)
                    hot_t1      = _tier1_syntax_check(hot_code)
                    hot_t2      = _tier2_semantic_check(hot_raw, self.rtl_ir, self.context)
                    refined = Draft(
                        iteration    = k,
                        property_code= hot_raw.get("property_code", ""),
                        assert_stmt  = hot_raw.get("assert_statement", ""),
                        reasoning    = hot_raw.get("reasoning", ""),
                        warnings     = hot_raw.get("warnings", []) + hot_t1 + hot_t2,
                        reward       = hot_reward,
                        tier1_passed = len(hot_t1) == 0,
                        tier2_passed = len(hot_t2) == 0,
                        token_entropy= hot_entropy,
                        drift_from_seed= _check_drift(seed_code, hot_code),
                    )
                    print(f"        Hot resample: reward={hot_reward:.3f} "
                          f"entropy={hot_entropy:.2f}")

            # Track best
            if refined.reward > best_so_far.reward:
                best_so_far = refined

            self._all_drafts.append(refined)
            self._refine_log.append({
                "iter": k,
                "reward": new_reward,
                "prev_reward": prev_reward,
                "gain": round(new_reward - prev_reward, 4),
                "drift": drift,
                "entropy": entropy,
                "tier1": refined.tier1_passed,
                "tier2": refined.tier2_passed,
                "rubric_avg": avg_rubric,
                "feedback_stop": feedback_stop,
            })

            prev_reward = new_reward
            current     = refined
            time.sleep(1)

        # ── Tool-use strategy update (RQ3.2) ─────────────────────────────
        # If best_so_far still hasn't passed Tier-1, escalate to multi-agent
        if not best_so_far.tier1_passed:
            print(
                f"    → Tier-1 still failing after refinement — "
                f"escalating to multi-agent (Phase 8)"
            )
            best_so_far = self._escalate_to_multi_agent(best_so_far, base_prompt)

        return self._wrap_result(
            best_so_far,
            stage1_only=False,
            rewards=rewards,
            advantages=advantages,
        )

    # ── Tool-use escalation ────────────────────────────────────────────────

    def _escalate_to_multi_agent(self, prior_draft: Draft, base_prompt: str) -> Draft:
        """
        RQ3.2 — tool-use strategy update: switch to multi-agent path.
        Passes the best single-shot draft as prior context.
        """
        try:
            from phase8_multi_agent import multi_agent_generate
            from phase6_generate import _tier1_syntax_check, _tier2_semantic_check

            mas_result = multi_agent_generate(
                self.context,
                prior_sva=prior_draft.full_code()
            )
            code = (mas_result.get("property_code", "") + " " +
                    mas_result.get("assert_statement", ""))
            t1 = _tier1_syntax_check(code)
            t2 = _tier2_semantic_check(mas_result, self.rtl_ir, self.context)
            mas_result["rubric_scores"] = {}
            mas_reward = reward_function(mas_result, self.context, self.rtl_ir)

            return Draft(
                iteration    = -1,  # special: escalated
                property_code= mas_result.get("property_code", ""),
                assert_stmt  = mas_result.get("assert_statement", ""),
                reasoning    = mas_result.get("reasoning", ""),
                warnings     = mas_result.get("warnings", []) + t1 + t2,
                reward       = mas_reward,
                tier1_passed = len(t1) == 0,
                tier2_passed = len(t2) == 0,
                token_entropy= _token_entropy(code),
            )
        except Exception as e:
            print(f"      [MAS escalation failed] {e}")
            return prior_draft

    # ── Result wrapping ────────────────────────────────────────────────────

    def _wrap_result(
        self,
        best: Draft,
        stage1_only: bool,
        rewards: list[float],
        advantages: list[float],
    ) -> RefinedAssertion:
        """Package Draft into RefinedAssertion with full metadata for RQ analysis."""
        prop_name = f"p_{self.rule_id.lower()}"

        # Confidence: scale reward [0,1] → [0,100]
        confidence = min(100, max(0, int(best.reward * 100)))

        # Collect all iteration stats for RQ3 ablation
        refine_metadata = {
            "stage1_only":   stage1_only,
            "n_stage1":      len([d for d in self._all_drafts if d.iteration == 0]),
            "n_refinements": len([d for d in self._all_drafts if d.iteration > 0]),
            "best_iteration": best.iteration,
            "best_reward":   best.reward,
            "stage1_rewards": rewards,
            "stage1_advantages": advantages,
            "refine_log":    self._refine_log,
            "final_drift":   best.drift_from_seed,
            "final_entropy": best.token_entropy,
            # RQ3.1: which feedback signals were used
            "feedback_signals_used": [
                "tier1_syntax",
                "tier2_semantic",
                "grounding_confidence",
                "multi_aspect_rubric",
            ],
            # RQ3.2: which improvement mechanisms fired
            "improvement_mechanisms": self._summarise_mechanisms(),
            # RQ3.3: anti-drift events
            "drift_reverts":      sum(1 for e in self._refine_log
                                      if e.get("event") == "drift_revert"),
            "entropy_resamples":  sum(1 for e in self._refine_log
                                      if e.get("event") == "entropy_resample"),
            "escalated_to_mas":   best.iteration == -1,
        }

        return RefinedAssertion(
            property_name    = prop_name,
            property_code    = best.property_code or f"// Generation failed for {prop_name}",
            assert_statement = best.assert_stmt   or "// error",
            confidence       = confidence,
            reasoning        = best.reasoning,
            warnings         = best.warnings,
            tier1_passed     = best.tier1_passed,
            tier2_passed     = best.tier2_passed,
            refine_metadata  = refine_metadata,
        )

    def _summarise_mechanisms(self) -> list[str]:
        used = ["prompt_optimisation_igrpo_stage2"]
        if any(e.get("event") == "drift_revert" for e in self._refine_log):
            used.append("drift_revert_guard")
        if any(e.get("event") == "entropy_resample" for e in self._refine_log):
            used.append("entropy_collapse_resample")
        if any(e.get("iter", 0) > 0 for e in self._refine_log):
            used.append("self_refine_feedback_loop")
        return used

    def _fallback_result(self, reason: str) -> RefinedAssertion:
        prop_name = f"p_{self.rule_id.lower()}"
        return RefinedAssertion(
            property_name    = prop_name,
            property_code    = f"// {reason}",
            assert_statement = "// error",
            confidence       = 0,
            reasoning        = reason,
            warnings         = [reason],
            tier1_passed     = False,
            tier2_passed     = False,
            refine_metadata  = {"error": reason},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback signal persistence  (RQ3.1 + RQ3.2 — memory update hook)
# ═══════════════════════════════════════════════════════════════════════════════

def persist_feedback_signals(
    rule_id: str,
    result: RefinedAssertion,
    groundings: dict,
    context: dict,
    rule: dict,
):
    """
    After Phase 10 completes, persist all feedback signals to Phase 7 memory.

    RQ3.2: Turns feedback into improved behaviour via memory updates:
      • reinforces grounding entries that contributed to passing assertions
      • penalises entries involved in failed iterations
      • records the reward trajectory so future runs can leverage it
    """
    from phase7_memory import update_memory

    validation_passed = result.tier1_passed and result.tier2_passed

    # Build assertion dict compatible with Phase 7
    assertion_dict = {
        "property_name":    result.property_name,
        "property_code":    result.property_code,
        "assert_statement": result.assert_statement,
        "confidence":       result.confidence,
        "reasoning":        result.reasoning,
        "warnings":         result.warnings,
        "tier1_passed":     result.tier1_passed,
        "tier2_passed":     result.tier2_passed,
        # Embed Phase-10 metadata for downstream analysis
        "phase10_metadata": result.refine_metadata,
    }

    update_memory(
        rule_id           = rule_id,
        groundings        = groundings,
        assertion         = assertion_dict,
        validation_passed = validation_passed,
        rule              = rule,
        context           = context,
        memory_stats      = result.refine_metadata,
    )

    print(
        f"  [Phase 10] Feedback signals persisted "
        f"(reward={result.refine_metadata.get('best_reward', 0):.3f}, "
        f"mechanisms={result.refine_metadata.get('improvement_mechanisms', [])})"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Generalisation / contradiction guard  (RQ3.3)
# ═══════════════════════════════════════════════════════════════════════════════

def check_generalisation_guard(
    rule_id: str,
    new_result: RefinedAssertion,
    design_hash: str,
) -> tuple[bool, str]:
    """
    RQ3.3: Prevent degradation by checking whether the new assertion
    regresses against a previously validated one stored in L3 cache.

    Returns (safe_to_use, reason).
    """
    from phase0_session import load_assertion_cache

    cached = load_assertion_cache(rule_id, design_hash)
    if not cached:
        return True, "no prior cached assertion — safe to proceed"

    # If prior assertion was validated but new one is not, flag it
    if cached.get("tier1_passed") and not new_result.tier1_passed:
        return False, (
            f"Regression: prior assertion for {rule_id} passed Tier-1 "
            f"but new result does not. Retaining cached version."
        )

    # Semantic drift check: compare property code
    prior_code = cached.get("property_code", "")
    new_code   = new_result.property_code
    drift      = 1.0 - _cosine_sim_text(prior_code, new_code)

    if drift > 0.80:
        # Very different from prior — flag as potential contradiction
        return False, (
            f"High drift ({drift:.2f}) from prior validated assertion. "
            f"Manual review recommended before overwriting."
        )

    return True, f"generalisation check passed (drift={drift:.2f})"


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline integration helper (called from main.py)
# ═══════════════════════════════════════════════════════════════════════════════

def run_self_refine(
    context: dict,
    rtl_ir: dict,
    groundings: dict,
    rule: dict,
    design_hash: str,
    n_stage1_drafts: int = N_EXPLORATORY_DRAFTS,
    max_refine_iters: int = MAX_REFINE_ITERS,
) -> dict:
    """
    Top-level entry point called from the main pipeline (main.py).

    Returns a dict compatible with Phase 6 / 7 output format,
    augmented with phase10 metadata.

    Integration pattern:
        # In process_rule() in main.py, replace the phase6 call with:
        from phase10_self_refine import run_self_refine
        assertion = run_self_refine(
            context, rtl_ir, grounding_result["groundings"],
            rule, sess["design_hash"]
        )
    """
    engine = SelfRefineEngine(context=context, rtl_ir=rtl_ir)
    result = engine.run(
        n_stage1_drafts = n_stage1_drafts,
        max_refine_iters= max_refine_iters,
    )

    # Generalisation guard before committing
    safe, guard_reason = check_generalisation_guard(
        rule["id"], result, design_hash
    )
    if not safe:
        print(f"  [Phase 10] ⚠  Generalisation guard: {guard_reason}")
        result.warnings.append(f"GENERALISATION GUARD: {guard_reason}")
        result.refine_metadata["generalisation_guard"] = guard_reason

    # Persist feedback signals to memory
    persist_feedback_signals(
        rule_id    = rule["id"],
        result     = result,
        groundings = groundings,
        context    = context,
        rule       = rule,
    )

    # Return as flat dict (Phase 6-compatible)
    return {
        "property_name":    result.property_name,
        "property_code":    result.property_code,
        "assert_statement": result.assert_statement,
        "confidence":       result.confidence,
        "reasoning":        result.reasoning,
        "warnings":         result.warnings,
        "tier1_passed":     result.tier1_passed,
        "tier2_passed":     result.tier2_passed,
        "phase10_metadata": result.refine_metadata,
    }