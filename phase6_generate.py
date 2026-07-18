"""
Phase 6 — Assertion Generation
"""

import json, re, time
import ollama
from config import (
    OLLAMA_HOST, OLLAMA_MODEL, DEFAULT_TIMING_WINDOW, MAX_SNIPPET_CHARS,
)

_client = ollama.Client(host=OLLAMA_HOST)

_YOSYS_SYSTEM_PROMPT = """You are an Expert Formal Verification Engineer writing assertions for open-source SymbiYosys (SBY).

Your task is to translate natural language specifications into mathematically rigorous, Yosys-compatible formal properties. Because open-source Yosys lacks a full SVA parser, you MUST adhere to these strict rules:

=== YOSYS NATIVE SYNTAX RULES ===
1. NO SVA ARROWS OR DELAYS: You are strictly forbidden from using SystemVerilog sequence operators (`|->`, `|=>`, `##1`, `##[1:10]`). Yosys will crash if it sees them.
2. PURE VERILOG ONLY: You must express all implications using standard procedural `if` statements inside an `always` block.
3. USE $past(): To check properties across clock cycles, use the `$past(signal)` system function instead of delay hashes.
4. PROCEDURAL ASSERT: Use `assert(condition);` inside the always block. Do not use `assert property`.
5. ASSUMPTIONS: If the rule requires the system to be out of reset, or requires an input to remain stable, write an `assume()` statement at the top of your block to constrain the solver (e.g., `assume(!rst);`).

=== HARDWARE REALITY RULES ===
6. THE 1-CYCLE RULE: If an English specification implies cause-and-effect (e.g., "When X goes high, Y is asserted"), assume there is a 1-clock-cycle register delay unless stated otherwise. Use `if ($past(X)) assert(Y);`. 

=== LOGICAL MAPPING RULES ===
7. STRICT SIGNAL MAPPING: Only use signals from the provided RTL list.
8. THE STRICT REFUSAL RULE: If you cannot find a logical, existing RTL signal from the provided list, you MUST completely abort. Do NOT invent names. Do NOT use placeholder comments like `/* inferred */`. Do NOT leave empty `if ()` conditions. If a mapping is impossible, output exactly: "ERROR: UNMAPPABLE_SIGNAL" and nothing else.
9. AVOID VACUOUS PASSES: Pay strict attention to your reset logic. If you write assume(!rst); at the top of your block, you MUST place your assertions inside if (!rst) begin. Do NOT place assertions inside if (rst), because the solver will never reach them and it will generate a false positive.

=== REASONING PROTOCOL (MUST DO THIS FIRST) ===
Before writing the Verilog code, you MUST write a commented block explaining your hardware logic. You must explicitly answer these three questions:
// 1. CLOCK: Which clock domain does this rule belong to?
// 2. DELAY: Does the obligation happen on the EXACT SAME cycle as the trigger, or the NEXT cycle ($past)? 
// 3. MAPPING: Exactly which RTL signals map to the nouns in the specification?

=== OUTPUT FORMAT ===
9. STRICTLY NO CHAT: Output ONLY the commented reasoning block followed by the raw Verilog code. Do not include markdown formatting (like ```verilog), explanations, or conversational text. Any English text outside of `//` comments will crash the downstream parser.

=== EXAMPLE ===
// 1. CLOCK: sys_clk
// 2. DELAY: Next cycle. The spec implies a 1-cycle flop delay, so I will use $past(req).
// 3. MAPPING: Request maps to `req_i`, Acknowledge maps to `ack_o`.
always @(posedge sys_clk) begin
  assume(!rst_n == 0);
  if (rst_n) begin
    if ($past(req_i)) begin
      assert(ack_o);
    end
  end
end
"""

_SVA_KEYWORDS = {
    "assert", "assume", "cover", "always", "begin", "end", "if", "else",
    "sequence", "endsequence",
    "posedge", "negedge",
    # "if", "else", "begin", "end", # Already added above
    "logic", "reg", "wire", "input", "output", "inout",
    "always", "assign", "module", "endmodule",
    "not", "and", "or", "throughout", "until", "within",
    "true", "false", "rose", "fell", "stable", "past",
    "first_match", "intersect", "s_eventually", "s_until",
    "nexttime", "always", "s_always", "s_nexttime",
}
_SVA_SYSTEM_FUNCS = re.compile(r"^\$")
_VERILOG_LITERAL_RE = re.compile(r"^\d*'[sdhbo]\w+$", re.IGNORECASE)


# ── 6.1  Prompt construction ────────────────────────────────────

def _resolve_clock(context: dict) -> str:
    """Smarter clock resolution: prioritize clocks related to rule signals."""
    rtl_slice = context.get("rtl_slice", {})
    rule_text = (context["rule"]["trigger"]["expression"] + " " +
                 context["rule"]["obligation"]["expression"]).lower()
    all_clocks = [c for c in rtl_slice.get("clocks", []) if c]
    if not all_clocks:
        return "clk" # fallback

    # 1. Prioritize clock mentioned in the rule text
    for clk in all_clocks:
        if clk.lower() in rule_text:
            return clk

    # 2. Prioritize clock connected to rule signals in dependency graph
    dep_graph = context.get("dep_graph", {})
    rule_signals = set(rtl_slice.get("trigger_signals", []) + rtl_slice.get("obligation_signals", []))
    for sig in rule_signals:
        node = dep_graph.get(sig, {})
        if node and node.get("clock") in all_clocks:
            return node["clock"]

    # 3. Fallback to first clock in the list
    return all_clocks[0]


def _resolve_reset(rtl_slice: dict) -> tuple[str, bool]:
    resets = [r for r in rtl_slice.get("resets", []) if r]
    if not resets:
        return "rst_n", True
    rst = resets[0]
    active_low = rst.endswith("_n") or rst.endswith("_b") or rst.endswith("N")
    return rst, active_low


def _timing_to_sva(timing: dict) -> str:
    t = timing.get("type", "unspecified")
    v = timing.get("value")
    if t == "immediate":
        return ""
    if t == "next_cycle":
        return "##1"
    if t in ("within_cycles", "after_cycles") and v:
        return f"##{v}"
    if t == "within_cycles" and not v:
        return f"##{DEFAULT_TIMING_WINDOW}"
    return f"##{DEFAULT_TIMING_WINDOW}"


def build_prompt(context: dict, prior_error: str = "") -> str:
    rule      = context["rule"]
    grounded  = context["grounded_signals"]
    rtl_slice = context["rtl_slice"]

    clk              = _resolve_clock(context)
    rst, active_low  = _resolve_reset(rtl_slice)

    trigger_rtl_sigs = sorted(list(set(rtl_slice.get("trigger_signals", []))))
    oblig_rtl_sigs   = sorted(list(set(rtl_slice.get("obligation_signals", []))))
    other_slice_sigs = sorted(list(
        set(rtl_slice.get("signals", [])) - set(trigger_rtl_sigs) - set(oblig_rtl_sigs)
    ))

    available_sigs_text = (
        f"Trigger-related    : {trigger_rtl_sigs or ['(none available)']}\\n"
        f"Obligation-related : {oblig_rtl_sigs or ['(none available)']}\\n"
        f"Other relevant     : {other_slice_sigs or ['(none available)']}"
    )

    sig_lines = [
        f"  '{g['spec_name']}' → '{g['rtl_name']}' (conf {g['confidence']:.2f})"
        for g in grounded
        if g.get("rtl_name") and g.get("confidence", 0) >= 0.5
    ]
    sig_map = "\n".join(sig_lines) or "  (no direct mappings — infer from code)"

    error_section = ""
    if prior_error:
        error_section = f"\n\nPREVIOUS ATTEMPT FAILED — fix these issues:\n{prior_error}\n"

    return f"""
══════════ RULE {rule['id']} ══════════
Trigger    : {rule['trigger']['expression'][:250]}
Obligation : {rule['obligation']['expression'][:250]}
Timing     : type={rule['timing']['type']}, value={rule['timing'].get('value', 'N/A')} cycles
Guards     : {rule.get('guards', [])}

══════════ GROUNDED SIGNAL MAPPINGS ══════════
(spec term → RTL signal, confidence)
{sig_map}

══════════ AVAILABLE RTL SIGNALS ══════════
{available_sigs_text}

══════════ CLOCK & RESET CONTEXT ══════════
Clock candidate : {clk}
Reset candidate : {rst} (active-{'low' if active_low else 'high'})

{error_section}
Now, generate the Yosys-compatible Verilog code for the assertion based on the rules provided in your system instructions.
"""


# ── 6.2  LLM inference ──────────────────────────────────────────

def _call_llm(prompt: str, max_retries: int = 3) -> str | None:
    for attempt in range(max_retries + 1):
        try:
            resp = _client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": _YOSYS_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.1, "num_predict": 1024},
            )
            raw = resp['message']['content'].strip()
            return raw.strip()

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 30
                m = re.search(r"retry.after.(\d+)", err, re.IGNORECASE)
                if m:
                    wait = int(m.group(1)) + 5
                print(f"    [rate limit] waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"    [LLM error attempt {attempt+1}] {err[:200]}")
                if attempt == max_retries:
                    return None
                time.sleep(3)
    return None


# ── 6.3  Tier-1 syntax check ────────────────────────────────────

def _tier1_syntax_check(code: str) -> list[str]:
    errs = []
    if "ERROR: UNMAPPABLE_SIGNAL" in code:
        errs.append("LLM Refused: Unmappable signals.")
        return errs # Return immediately

    if "property" in code or "endproperty" in code:
        errs.append("Named properties are forbidden for Yosys.")

    if "assert property" in code:
        errs.append("'assert property' is forbidden; use 'assert(...)' instead.")

    if "|->" in code or "|=>" in code:
        errs.append("SVA sequence operators ('|->', '|=>') are forbidden.")

    if "##" in code:
        errs.append("SVA delay operator ('##') is forbidden; use $past() instead.")

    if "disable iff" in code:
        errs.append("'disable iff' is forbidden; use procedural 'if (!reset)' instead.")

    # To pass, must contain "always @" AND "assert".
    if "always @" not in code:
        errs.append("Missing 'always @' block for procedural assertion.")

    if "assert" not in code:
        errs.append("Missing 'assert' statement.")

    return errs


# ── 6.3  Tier-2 semantic check ──────────────────────────────────

def _tier2_semantic_check(result: dict, rtl_ir: dict, context: dict) -> list[str]:
    code = result.get("property_code", "")

    # 1. Sanitize code: strip comments and string literals
    sanitized_code = re.sub(r"//.*", "", code) # Strip single-line comments
    sanitized_code = re.sub(r"/\*.*?\*/", "", sanitized_code, flags=re.DOTALL) # Strip block comments
    sanitized_code = re.sub(r'"(.*?)"', "", sanitized_code) # Strip string literals

    prop_name = result.get("property_name", "")

    skip = _SVA_KEYWORDS | {prop_name, prop_name.replace("_", "")}
    all_rtl_lower    = {s.lower() for s in rtl_ir.get("all_signals", [])}
    slice_sigs_lower = {
        s.lower() for s in (
            context["rtl_slice"].get("trigger_signals", []) +
            context["rtl_slice"].get("obligation_signals", []) +
            context["rtl_slice"].get("signals", []) +
            context["rtl_slice"].get("clocks", []) +
            context["rtl_slice"].get("resets", [])
        ) if s
    }
    allowed = all_rtl_lower | slice_sigs_lower

    used = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]+)\b", sanitized_code)
    unknown = [
        s for s in used
        if s.lower() not in allowed
        and s.lower() not in skip
        and len(s) > 2
        and not _SVA_SYSTEM_FUNCS.match(s)
        and not s.isdigit()
        and not _VERILOG_LITERAL_RE.match(s)
    ]
    if unknown:
        return [f"Signal(s) not found in RTL: {list(set(unknown))[:5]}"]
    return []

# ── Main generation function ──────────────────────────────────────────────────

def generate_assertion(context: dict, max_retries: int = 2) -> dict:
    print(f"  [Phase 6] Generating assertion for rule {context['rule']['id']}")
    prior_error = ""

    for attempt in range(max_retries + 1):
        prompt = build_prompt(context, prior_error=prior_error)
        raw_code = _call_llm(prompt)

        if raw_code is None:
            return _error_result(context, "LLM call failed (all retries exhausted)")

        # The user's new prompt asks for raw code, not JSON.
        # We check for the refusal rule first.
        if "ERROR: UNMAPPABLE_SIGNAL" in raw_code:
            print(f"    [attempt {attempt+1}] LLM refused to generate: Unmappable Signal.")
            return _error_result(context, "LLM flagged rule as unmappable: A key signal for the trigger or obligation could not be grounded to the available RTL signals.")

        # Since we get raw code, we can't get confidence/reasoning from the LLM.
        # We construct the result dictionary manually.
        result = {
            "property_name": f"p_{context['rule']['id'].lower()}",
            "property_code": raw_code,
            "assert_statement": "// Embedded in property_code for Yosys",
            "confidence": 95, # Default high confidence as we can't get it from LLM
            "reasoning": "Generated via Yosys-specific prompt.",
            "warnings": [],
        }

        t1_errs = _tier1_syntax_check(raw_code)
        if t1_errs and attempt < max_retries:
            prior_error = "SVA syntax issues:\n" + "\n".join(f"  - {e}" for e in t1_errs)
            print(f"    [attempt {attempt+1}] Tier-1 fail: {t1_errs} → retrying")
            time.sleep(2)
            continue

        # If we are here, it means either Tier-1 passed or we are out of retries.
        print(f"    Confidence: {result.get('confidence', '?')}/100")
        result["tier1_errors"] = t1_errs
        return result

    return _error_result(context, f"Failed after {max_retries+1} attempts")

def validate_assertion_tier1(result: dict, rtl_ir: dict, context: dict) -> dict:
    code     = result.get("property_code", "")
    warnings = list(result.get("warnings", []))

    t1_errs  = _tier1_syntax_check(code)
    t2_errs  = _tier2_semantic_check(result, rtl_ir, context)

    warnings += t1_errs + t2_errs
    result["warnings"]      = [w for w in warnings if w]
    result["tier1_passed"]  = len(t1_errs)  == 0
    result["tier2_passed"]  = len(t2_errs)  == 0 

    return result


def _error_result(context: dict, reason: str) -> dict:
    rule_id = context["rule"]["id"]
    return {
        "property_name":    f"p_{rule_id.lower()}",
        "property_code":    f"// Generation failed: {reason}",
        "assert_statement": "// error",
        "confidence":       0,
        "reasoning":        reason,
        "warnings":         [reason],
        "tier1_passed":     False,
        "tier2_passed":     False,
    }