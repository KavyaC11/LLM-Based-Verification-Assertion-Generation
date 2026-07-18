"""
Phase 5 — Sufficiency Check & Iterative Refinement
"""

from config import L1_WEIGHT, L2_WEIGHT, OVERALL_THRESHOLD, MAX_ITERATIONS


# ── 5.1  Level 1: Structural ──────────────────────────────────────────────────

def check_l1_structural(context: dict) -> tuple[bool, list]:
    issues = []
    rtl    = context["rtl_slice"]
    rule   = context["rule"]

    if not rtl["clocks"]:
        issues.append(
            "No clock identified. Provide clock signal name for this design."
        )
    if not rtl["trigger_signals"] and not rtl["obligation_signals"]:
        issues.append(
            f"Cannot map trigger or obligation to any RTL signal. "
            f"Trigger='{rule['trigger']['expression'][:60]}' — "
            f"please clarify which RTL signal this refers to."
        )

    return len(issues) == 0, issues


# ── 5.2  Level 2: Semantic ────────────────────────────────────────────────────

def check_l2_semantic(context: dict) -> tuple[float, list]:
    rule   = context["rule"]
    rtl    = context["rtl_slice"]
    issues = []
    score  = 0.0
    checks = 4

    if rtl["trigger_signals"]:
        score += 1
    else:
        issues.append(
            f"Trigger '{rule['trigger']['expression'][:60]}' cannot be expressed "
            f"in RTL terms — no matching signal found."
        )

    if rtl["obligation_signals"]:
        score += 1
    else:
        issues.append(
            f"Obligation '{rule['obligation']['expression'][:60]}' cannot be "
            f"expressed in RTL terms — no matching signal found."
        )

    timing = rule["timing"]
    if timing["type"] not in ("eventually", "soon", "unspecified"):
        score += 1
    else:
        issues.append(
            f"Timing is ambiguous ('{timing['type']}'). "
            f"Specify an exact cycle count or qualifying condition."
        )

    if context["grounded_signals"]:
        avg_conf = sum(
            g["confidence"] for g in context["grounded_signals"]
        ) / len(context["grounded_signals"])
        if avg_conf >= 0.60:
            score += 1
        else:
            issues.append(
                f"Average grounding confidence is low ({avg_conf:.2f}). "
                f"Signal mappings may be unreliable."
            )
    else:
        issues.append("No grounded signals available for this rule.")

    return score / checks, issues


# ── 5.3  Iterative refinement loop ────────────────────────────────────────────

def check_sufficiency(context: dict, iterations: int = 0) -> dict:
    l1_pass,  l1_issues = check_l1_structural(context)
    l2_score, l2_issues = check_l2_semantic(context)

    overall = (
        (1.0 if l1_pass else 0.0) * L1_WEIGHT
        + l2_score * L2_WEIGHT
    )
    all_issues = l1_issues + l2_issues
    passed     = overall >= OVERALL_THRESHOLD

    clarifications = []
    for issue in all_issues:
        clarifications.append({
            "issue":   issue,
            "request": _build_targeted_request(issue, context),
        })

    return {
        "passed":         passed,
        "overall_score":  round(overall, 3),
        "l1_pass":        l1_pass,
        "l2_score":       round(l2_score, 3),
        "issues":         all_issues,
        "clarifications": clarifications,
        "iterations":     iterations,
        "escalate":       (not passed and iterations >= MAX_ITERATIONS),
    }


def _build_targeted_request(issue: str, context: dict) -> str:
    rule = context["rule"]
    if "clock" in issue.lower():
        return (
            f"Rule {rule['id']}: No clock found for the signals involved. "
            f"Please specify the clock signal name (e.g., 'clk', 'sys_clk')."
        )
    if "trigger" in issue.lower() and "rtl" in issue.lower():
        return (
            f"Rule {rule['id']}: Cannot find RTL signal for trigger "
            f"'{rule['trigger']['expression'][:60]}'. "
            f"Provide the exact RTL signal name or module."
        )
    if "obligation" in issue.lower() and "rtl" in issue.lower():
        return (
            f"Rule {rule['id']}: Cannot find RTL signal for obligation "
            f"'{rule['obligation']['expression'][:60]}'. "
            f"Provide the exact RTL signal name or module."
        )
    if "timing" in issue.lower():
        return (
            f"Rule {rule['id']}: Timing is unspecified. "
            f"How many clock cycles until the obligation must hold? "
            f"(e.g., 'within 3 cycles', 'on the next rising edge')"
        )
    return f"Rule {rule['id']}: {issue}"