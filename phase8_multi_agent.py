"""
Multi-Agent Coordinator
"""

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import ollama
from config import OLLAMA_HOST, OLLAMA_MODEL


_client = ollama.Client(host=OLLAMA_HOST)

# ═══════════════════════════════════════════════════════════════════════════════
# Agent Role Definitions
# ═══════════════════════════════════════════════════════════════════════════════

class AgentRole(Enum):
    TRIGGER_ANALYST   = "trigger_analyst"    # Focuses on antecedent/trigger extraction
    OBLIGATION_MAPPER = "obligation_mapper"  # Focuses on consequent/obligation mapping
    TIMING_SPECIALIST = "timing_specialist"  # Focuses on timing/cycle constraints
    SVA_SYNTHESIZER   = "sva_synthesizer"   # Assembles final SVA from analysis
    REVIEWER          = "reviewer"           # Reviews and critiques the generated SVA

AGENT_PERSONAS = {
    AgentRole.TRIGGER_ANALYST: (
        "You are Alex, a hardware verification engineer with 10 years specializing in "
        "protocol trigger conditions and antecedent logic. You are meticulous about edge "
        "cases and signal transitions. Your role: analyze ONLY the trigger/antecedent "
        "side of the rule — what conditions must be true to fire the assertion."
    ),
    AgentRole.OBLIGATION_MAPPER: (
        "You are Morgan, an RTL design expert specializing in output behavior and "
        "consequent conditions. You excel at mapping spec obligations to RTL signal "
        "assignments. Your role: analyze ONLY the obligation/consequent side — what "
        "must hold after the trigger fires."
    ),
    AgentRole.TIMING_SPECIALIST: (
        "You are Jordan, a formal verification expert focused exclusively on temporal "
        "logic and cycle-accurate timing. You are precise about ##N operators and "
        "timing windows. Your role: determine the exact SVA timing operator needed."
    ),
    AgentRole.SVA_SYNTHESIZER: (
        "You are Casey, a senior SVA architect specializing in Yosys-compatible SystemVerilog Assertions. "
        "You synthesize inputs from multiple analysts into a single coherent, syntactically correct SVA. "
        "CRITICAL: You MUST use procedural assertions only, inside an 'always @(posedge clk) begin ... end' block. "
        "Handle resets with 'if (!reset)' and absolutely AVOID 'property ... endproperty' and 'disable iff'."
        "Your role: combine the trigger, obligation, and timing analyses into valid, Yosys-compatible SVA."
    ),
    AgentRole.REVIEWER: (
        "You are Riley, a formal verification reviewer who checks SVA for correctness, "
        "coverage, and potential false positives. Think carefully about what other agents "
        "might have missed and how your review complements their work. "
        "Your role: critique and suggest improvements to the generated assertion."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Communication Structures
# ═══════════════════════════════════════════════════════════════════════════════

class CommStructure(Enum):
    HORIZONTAL = "horizontal"  # All agents discuss, result aggregated (diversity)
    VERTICAL   = "vertical"    # Solver + reviewers, iterative refinement (quality)


# ═══════════════════════════════════════════════════════════════════════════════
# Redundancy Detection
# ═══════════════════════════════════════════════════════════════════════════════

def _fingerprint(text: str) -> str:
    """Semantic fingerprint: normalize + hash for fast similarity check."""
    # Normalize: lowercase, strip whitespace, remove comments
    normalized = re.sub(r"//.*", "", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on token sets — fast pre-filter before LLM cost."""
    tokens_a = set(re.findall(r"\b\w+\b", a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


@dataclass
class RedundancyDetector:
    """
    RQ2.2: Early redundancy detection.
    Maintains a registry of seen outputs. Detects duplicates before incurring
    LLM cost using fingerprinting + Jaccard similarity.
    """
    _seen_fingerprints: dict = field(default_factory=dict)
    _seen_outputs: list = field(default_factory=list)
    similarity_threshold: float = 0.75  # above this → redundant

    def is_redundant(self, text: str, agent_role: AgentRole) -> tuple[bool, Optional[str]]:
        """
        Returns (is_redundant, existing_equivalent).
        Checks fingerprint first (O(1)), then Jaccard (cheap), then flags.
        """
        fp = _fingerprint(text)

        # Exact match
        if fp in self._seen_fingerprints:
            existing_role = self._seen_fingerprints[fp]
            if existing_role != agent_role.value:
                return True, self._seen_fingerprints.get(fp + "_text", "")

        # Fuzzy match — check against recent outputs
        for seen_text, seen_role in self._seen_outputs[-10:]:
            if seen_role == agent_role.value:
                continue  # same role is expected to produce similar output
            similarity = _token_overlap(text, seen_text)
            if similarity >= self.similarity_threshold:
                return True, seen_text

        return False, None

    def register(self, text: str, agent_role: AgentRole):
        fp = _fingerprint(text)
        self._seen_fingerprints[fp] = agent_role.value
        self._seen_fingerprints[fp + "_text"] = text[:200]
        self._seen_outputs.append((text, agent_role.value))


# ═══════════════════════════════════════════════════════════════════════════════
# Synergy Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_synergy_score(agent_outputs: dict[str, str]) -> float:
    """
    Approximate PID-inspired synergy score.
    Synergy > 0 means agents provide complementary (non-overlapping) info.
    High redundancy (overlap) → low synergy → we should stop adding agents.

    Simplified: synergy = 1 - mean_pairwise_overlap
    performance benefits when synergy AND redundancy coexist.
    """
    roles = list(agent_outputs.keys())
    if len(roles) < 2:
        return 1.0  # single agent: maximum unique contribution

    overlaps = []
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            overlaps.append(_token_overlap(agent_outputs[roles[i]], agent_outputs[roles[j]]))

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    synergy = 1.0 - mean_overlap
    return round(synergy, 3)


def _compute_redundancy_score(agent_outputs: dict[str, str]) -> float:
    """
    Redundancy = alignment on shared content.
    High redundancy = good alignment but no new info.
    We want: high synergy + moderate redundancy (both needed for performance).
    """
    roles = list(agent_outputs.keys())
    if len(roles) < 2:
        return 0.0

    overlaps = []
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            overlaps.append(_token_overlap(agent_outputs[roles[i]], agent_outputs[roles[j]]))

    return round(sum(overlaps) / len(overlaps), 3) if overlaps else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Task Allocation Strategy
# ═══════════════════════════════════════════════════════════════════════════════

def _assess_rule_complexity(rule: dict, context: dict) -> float:
    """
    Score rule complexity 0.0–1.0 to choose allocation strategy.
    Complex rules → HORIZONTAL (diverse agents cover breadth).
    Simple rules  → VERTICAL (single solver + reviewer refinement).
    """
    score = 0.0

    # FSM involvement
    if context.get("rtl_slice", {}).get("fsm_subgraph"):
        score += 0.25

    # Timing ambiguity
    timing_type = rule.get("timing", {}).get("type", "unspecified")
    if timing_type in ("unspecified", "eventually", "soon"):
        score += 0.20

    # Multiple signals
    n_trigger = len(context.get("rtl_slice", {}).get("trigger_signals", []))
    n_oblig = len(context.get("rtl_slice", {}).get("obligation_signals", []))
    if n_trigger + n_oblig > 3:
        score += 0.20

    # Low grounding confidence
    grounded = context.get("grounded_signals", [])
    if grounded:
        avg_conf = sum(g.get("confidence", 0) for g in grounded) / len(grounded)
        if avg_conf < 0.70:
            score += 0.20

    # Ambiguity flags
    flags = rule.get("ambiguity_flags", [])
    score += min(len(flags) * 0.05, 0.15)

    return min(score, 1.0)


def select_allocation_strategy(rule: dict, context: dict) -> tuple[CommStructure, list[AgentRole]]:

    complexity = _assess_rule_complexity(rule, context)

    if complexity >= 0.5:
        # Complex rule → HORIZONTAL: recruit specialized experts
        structure = CommStructure.HORIZONTAL
        roles = [
            AgentRole.TRIGGER_ANALYST,
            AgentRole.OBLIGATION_MAPPER,
            AgentRole.TIMING_SPECIALIST,
            AgentRole.SVA_SYNTHESIZER,
        ]
        print(f"    [MAS] Horizontal structure (complexity={complexity:.2f}) — 4 agents")
    else:
        # Simple rule → VERTICAL: single solver + reviewer
        structure = CommStructure.VERTICAL
        roles = [AgentRole.SVA_SYNTHESIZER, AgentRole.REVIEWER]
        print(f"    [MAS] Vertical structure (complexity={complexity:.2f}) — 2 agents")

    return structure, roles


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Execution
# ═══════════════════════════════════════════════════════════════════════════════

def _call_agent(role: AgentRole, task_prompt: str, history: list[dict],
                tom_enabled: bool = True) -> str:
    """
    Call a single specialized agent.
    tom_enabled: Theory-of-Mind prompt (Riedl §3 ToM condition) — instructs agent
    to consider what other agents might contribute and complement them.
    """
    persona = AGENT_PERSONAS[role]

    # think about what other agents might do
    tom_instruction = ""
    if tom_enabled and history:
        other_roles = [h["role_name"] for h in history if h.get("role_name") != role.value]
        if other_roles:
            tom_instruction = (
                f"\n\nIMPORTANT — Theory of Mind: Other agents ({', '.join(other_roles)}) "
                f"have already contributed. Think step-by-step about what they likely covered "
                f"and focus ONLY on what they may have missed or gotten wrong. "
                f"Your contribution must be complementary, not redundant."
            )

    system_msg = persona + tom_instruction

    messages = [{"role": "system", "content": system_msg}]

    # Include relevant history
    for h in history[-3:]:  # last 3 turns to stay within context
        messages.append({"role": "assistant", "content": f"[{h['role_name']}]: {h['content']}"})

    messages.append({"role": "user", "content": task_prompt})

    for attempt in range(3):
        try:
            resp = _client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={"temperature": 0.3, "num_predict": 600},
            )
            return resp['message']['content'].strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 30
                m = re.search(r"retry.after.(\d+)", err, re.IGNORECASE)
                if m:
                    wait = int(m.group(1)) + 5
                print(f"      [rate limit] waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"      [agent {role.value} error attempt {attempt+1}] {err[:100]}")
                if attempt == 2:
                    return f"// Agent {role.value} failed: {err[:80]}"
                time.sleep(3)
    return f"// Agent {role.value} exhausted retries"


# ═══════════════════════════════════════════════════════════════════════════════
# Horizontal Workflow
# ═══════════════════════════════════════════════════════════════════════════════

def _run_horizontal(
    roles: list[AgentRole],
    base_prompt: str,
    rule: dict,
    context: dict,
    redundancy_detector: RedundancyDetector,
) -> dict:
    """
    RQ2.1 + RQ2.2 + RQ2.3:
    Each agent works on their specialty. Redundancy is checked before each call.
    Synergy is measured after all agents contribute.
    """
    history = []
    agent_outputs = {}
    skipped = []

    # Build role-specific sub-prompts
    role_prompts = {
        AgentRole.TRIGGER_ANALYST: (
            base_prompt +
            "\n\nYour specific task: Analyze ONLY the trigger/antecedent condition. "
            "Output: (1) the exact RTL signal(s) for the trigger, (2) the condition expression, "
            "(3) any edge-type needed (posedge/negedge/level). Be concise, max 150 words."
        ),
        AgentRole.OBLIGATION_MAPPER: (
            base_prompt +
            "\n\nYour specific task: Analyze ONLY the obligation/consequent. "
            "Output: (1) the exact RTL signal(s) for the obligation, (2) the expected value/condition, "
            "(3) whether it's combinational or registered. Be concise, max 150 words."
        ),
        AgentRole.TIMING_SPECIALIST: (
            base_prompt +
            "\n\nYour specific task: Determine ONLY the SVA timing operator. "
            "Output: (1) the ##N or |-> or |=> operator needed, (2) cycle count justification, "
            "(3) whether disable iff is needed. Be concise, max 100 words."
        ),
        AgentRole.SVA_SYNTHESIZER: (
            base_prompt +
            f"\n\nPrevious agent analyses:\n" +
            "\n".join(f"[{h['role_name']}]: {h['content']}" for h in history) +
            "\n\nYour task: Synthesize the above into one complete, syntactically valid SVA property. "
            "CRITICAL: You MUST use procedural assertions only, inside an 'always @(posedge clk) begin ... end' block. "
            "Handle resets with 'if (!reset)' and absolutely AVOID 'property ... endproperty' and 'disable iff'."
            "Output ONLY the SVA code block (the 'always' block containing the assert statement)."
        ),
    }

    for role in roles:
        # Check for redundancy BEFORE calling the LLM
        role_task = role_prompts.get(role, base_prompt)

        # Pre-check: if this role's domain is already fully covered, skip
        if role != AgentRole.SVA_SYNTHESIZER and len(agent_outputs) >= 2:
            synergy = _compute_synergy_score(agent_outputs)
            if synergy < 0.15:  # all agents saying same thing → stop early
                print(f"      [MAS] Low synergy ({synergy:.2f}) — stopping early, skipping {role.value}")
                skipped.append(role.value)
                continue

        # Update synthesizer prompt with current history
        if role == AgentRole.SVA_SYNTHESIZER:
            role_task = (
                base_prompt +
                f"\n\nPrevious agent analyses:\n" +
                "\n".join(f"[{h['role_name']}]: {h['content']}" for h in history) +
                "\n\nSynthesize into one complete SVA property block. "
                "CRITICAL: You MUST use procedural assertions only, inside an 'always @(posedge clk) begin ... end' block. "
                "Handle resets with 'if (!reset)' and absolutely AVOID 'property ... endproperty' and 'disable iff'."
                "Output ONLY SVA code (the 'always' block containing the assert statement)."
            )

        output = _call_agent(role, role_task, history, tom_enabled=True)

        # Post-call redundancy check — flag if output duplicates another agent
        is_dup, existing = redundancy_detector.is_redundant(output, role)
        if is_dup:
            print(f"      [MAS] Redundancy detected for {role.value} — flagged (not discarded)")
            # We keep it but log it; reviewer will consolidate
            output = output + "\n// [REDUNDANCY FLAG: overlaps with prior agent output]"

        redundancy_detector.register(output, role)
        agent_outputs[role.value] = output
        history.append({"role_name": role.value, "content": output})
        time.sleep(1)  # rate limit buffer

    synergy = _compute_synergy_score(agent_outputs)
    redundancy = _compute_redundancy_score(agent_outputs)

    return {
        "structure": "horizontal",
        "agent_outputs": agent_outputs,
        "history": history,
        "skipped_agents": skipped,
        "synergy_score": synergy,
        "redundancy_score": redundancy,
        "synthesis": agent_outputs.get(AgentRole.SVA_SYNTHESIZER.value, ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Vertical Workflow
# ═══════════════════════════════════════════════════════════════════════════════

def _run_vertical(
    roles: list[AgentRole],
    base_prompt: str,
    rule: dict,
    context: dict,
    redundancy_detector: RedundancyDetector,
    max_refinement_rounds: int = 2,
) -> dict:
    """
    RQ2.3: Vertical structure — solver proposes, reviewer refines.
    A=a*k (k refinements until consensus or max iterations).
    """
    solver_role = AgentRole.SVA_SYNTHESIZER
    reviewer_role = AgentRole.REVIEWER

    history = []
    agent_outputs = {}
    rounds = []

    # Solver initial proposal
    solver_prompt = (
        base_prompt +
        "\n\nGenerate a complete SVA property. "
        "CRITICAL: You MUST use procedural assertions only, inside an 'always @(posedge clk) begin ... end' block. "
        "Handle resets with 'if (!reset)' and absolutely AVOID 'property ... endproperty' and 'disable iff'."
        "Output ONLY valid SVA code (the 'always' block containing the assert statement)."
    )
    solution = _call_agent(solver_role, solver_prompt, history, tom_enabled=False)
    agent_outputs[solver_role.value] = solution
    history.append({"role_name": solver_role.value, "content": solution})
    rounds.append({"round": 0, "solver": solution, "reviewer": None, "agreed": False})

    for round_num in range(1, max_refinement_rounds + 1):
        # Check if reviewer output would be redundant (all [Agree] = done)
        review_prompt = (
            base_prompt +
            f"\n\nCurrent SVA proposal:\n{solution}\n\n"
            "Review this assertion. If it is correct and complete, respond with exactly '[Agree]'. "
            "Otherwise, provide specific corrections. "
            "CRITICAL: Ensure the SVA is Yosys-compatible (procedural, no named properties, if(!reset) for reset). "
            "Think about what the solver may have missed."
        )

        review = _call_agent(reviewer_role, review_prompt, history, tom_enabled=True)
        agent_outputs[reviewer_role.value + f"_r{round_num}"] = review
        history.append({"role_name": reviewer_role.value, "content": review})

        rounds[-1]["reviewer"] = review

        # AgentVerse stopping criterion: reviewer agrees
        if "[Agree]" in review or "[agree]" in review.lower():
            rounds[-1]["agreed"] = True
            print(f"      [MAS] Vertical consensus at round {round_num}")
            break

        # Solver refines based on reviewer feedback
        refine_prompt = (
            base_prompt +
            f"\n\nReviewer feedback:\n{review}\n\n"
            "Refine your SVA accordingly. "
            "CRITICAL: Ensure the SVA is Yosys-compatible (procedural, no named properties, if(!reset) for reset). "
            "Output ONLY the corrected SVA code."
        )
        solution = _call_agent(solver_role, refine_prompt, history, tom_enabled=False)
        agent_outputs[solver_role.value + f"_r{round_num}"] = solution
        history.append({"role_name": solver_role.value, "content": solution})
        rounds.append({"round": round_num, "solver": solution, "reviewer": None, "agreed": False})
        time.sleep(1)

    synergy = _compute_synergy_score(agent_outputs)
    redundancy = _compute_redundancy_score(agent_outputs)

    return {
        "structure": "vertical",
        "agent_outputs": agent_outputs,
        "history": history,
        "rounds": rounds,
        "skipped_agents": [],
        "synergy_score": synergy,
        "redundancy_score": redundancy,
        "synthesis": solution,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Stage
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate_result(synthesis: str, rule: dict, history: list[dict]) -> dict:
    """
    AgentVerse §2.4: Evaluate current state vs desired goal.
    Returns feedback for potential re-run with adjusted group composition.
    """
    from phase6_generate import _tier1_syntax_check
    t1_errs = _tier1_syntax_check(synthesis)

    score = 1.0 - (len(t1_errs) * 0.25)
    score = max(0.0, score)

    return {
        "tier1_errors": t1_errs,
        "score": round(score, 2),
        "needs_refinement": len(t1_errs) > 0,
        "feedback": (
            "SVA syntax issues found: " + "; ".join(t1_errs)
            if t1_errs else "SVA looks syntactically correct."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def multi_agent_generate(context: dict, prior_sva: str = "") -> dict:
    """
    Full multi-agent assertion generation pipeline.

    RQ2.1: Role differentiation via AgentVerse expert recruitment + Riedl personas
    RQ2.2: Redundancy detection before/after each agent call
    RQ2.3: Strategy selection (horizontal vs vertical) based on rule complexity

    Returns enriched assertion dict compatible with phase6_generate output format.
    """
    rule = context["rule"]
    rtl_slice = context["rtl_slice"]
    grounded = context.get("grounded_signals", [])

    # Build shared base prompt (context seen by all agents)
    sig_lines = "\n".join(
        f"  '{g['spec_name']}' → '{g['rtl_name']}' (conf {g['confidence']:.2f})"
        for g in grounded if g.get("rtl_name") and g.get("confidence", 0) >= 0.5
    )
    snippets = "\n\n".join(rtl_slice.get("code_snippets", []))[:1200]
    clocks = rtl_slice.get("clocks", ["clk"])
    clk = clocks[0] if clocks else "clk"
    resets = rtl_slice.get("resets", ["rst_n"])
    rst = resets[0] if resets else "rst_n"
    reset_expr = f"!{rst}" if rst.endswith("_n") else rst

    base_prompt = f"""RTL Assertion Generation Task
Rule ID: {rule['id']}
Trigger: {rule['trigger']['expression'][:200]}
Obligation: {rule['obligation']['expression'][:200]}
Timing: type={rule['timing']['type']}, value={rule['timing'].get('value', 'N/A')}

Signal Mappings (spec → RTL):
{sig_lines or '  (none — infer from code)'}

Clock: {clk}  Reset: {rst} (use: disable iff ({reset_expr}))

RTL Code:
```verilog
{snippets or '// (no snippets)'}
```
{"Prior SVA attempt (improve upon): " + prior_sva if prior_sva else ""}"""

    # Select allocation strategy
    structure, roles = select_allocation_strategy(rule, context)

    # Initialize redundancy detector for this rule
    detector = RedundancyDetector()

    # Execute selected structure
    if structure == CommStructure.HORIZONTAL:
        run_result = _run_horizontal(roles, base_prompt, rule, context, detector)
    else:
        run_result = _run_vertical(roles, base_prompt, rule, context, detector)

    synthesis = run_result["synthesis"]

    # Evaluation stage
    # For multi-agent, we need to extract the property code and assert statement
    # from the synthesis to run the _evaluate_result.
    # The synthesis should already be the full always block.
    if not synthesis or "always @(posedge" not in synthesis:
        # Fallback if synthesis is not a valid always block
        print(f"      [MAS] Synthesis not a valid always block, attempting rescue.")
        rescue_result = _run_horizontal(
            [AgentRole.TRIGGER_ANALYST, AgentRole.SVA_SYNTHESIZER],
            base_prompt,
            rule, context, RedundancyDetector()
        )
        synthesis = rescue_result["synthesis"]
        run_result["rescue_used"] = True

    # Now evaluate the final synthesis
    eval_result = _evaluate_result(synthesis, rule, run_result["history"])

    prop_code = synthesis.strip()
    assert_stmt = f"// Assertion for {rule['id']} is embedded in the always block above."

    prop_name = f"p_{rule['id'].lower()}"

    # log synergy/redundancy for analysis
    synergy = run_result["synergy_score"]
    redundancy = run_result["redundancy_score"]
    print(
        f"      [MAS] Synergy={synergy:.2f} Redundancy={redundancy:.2f} "
        f"Skipped={run_result['skipped_agents']}"
    )

    return {
        "property_name": prop_name,
        "property_code": prop_code,
        "assert_statement": assert_stmt,
        "confidence": max(0, min(100, int((eval_result["score"] * 0.7 + synergy * 0.3) * 100))),
        "reasoning": (
            f"Multi-agent ({run_result['structure']}, {len(run_result['agent_outputs'])} agents): "
            f"synergy={synergy:.2f}, redundancy={redundancy:.2f}. {eval_result['feedback']}"
        ),
        "warnings": eval_result["tier1_errors"],
        "tier1_errors": eval_result["tier1_errors"],
        # MAS metadata
        "mas_metadata": {
            "structure": run_result["structure"],
            "synergy_score": synergy,
            "redundancy_score": redundancy,
            "agents_used": list(run_result["agent_outputs"].keys()),
            "agents_skipped": run_result["skipped_agents"],
            "rescue_used": run_result.get("rescue_used", False),
        },
    }