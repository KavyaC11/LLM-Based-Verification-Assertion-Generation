"""
RTL Assertion Generator — main pipeline orchestrator
===============================================================

Usage:
  python main.py <design_name> [max_rules]

Examples:
  python main.py spi_master 5
  python main.py ethmac 10
  python main.py openMSP430
"""

import json
import os
import sys
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from pprint import pformat

from colorama import Fore, Style, init as colorama_init

colorama_init()

from config             import DESIGNS, MAX_ITERATIONS, CACHE_DIR
from phase0_session     import init_session, save_design_cache
from phase1_spec        import process_spec, dump_spec_ir
from phase2_rtl         import process_rtl, dump_rtl_ir
from phase3_grounding   import ground_signals, load_memory as load_grounding_memory
from phase4_context     import build_context
from phase5_sufficiency import check_sufficiency
from phase6_generate    import generate_assertion, validate_assertion_tier1
from phase7_memory      import update_memory, check_staleness


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header(text: str):
    print(f"\n{Fore.CYAN}{'═'*62}")
    print(f"  {text}")
    print(f"{'═'*62}{Style.RESET_ALL}")


def _print_rule_header(rule: dict, idx: int, total: int):
    print(f"\n{Fore.YELLOW}{'─'*62}")
    print(f"  Rule {rule['id']}  ({idx}/{total})")
    print(f"{'─'*62}{Style.RESET_ALL}")
    print(f"  Trigger:    {rule['trigger']['expression'][:90]}")
    print(f"  Obligation: {rule['obligation']['expression'][:90]}")
    timing = rule["timing"]
    print(
        f"  Timing:     {timing['type']} = {timing.get('value', 'N/A')} "
        f"[confidence: {timing['confidence']}]"
    )
    flags = rule.get("ambiguity_flags", [])
    if flags:
        print(f"  {Fore.YELLOW}Flags: {flags}{Style.RESET_ALL}")
    print(f"  Spec confidence: {rule['confidence']:.2f}")


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_run_dirs(design_name: str, session_id: str) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("results") / design_name / f"{ts}_{session_id}"

    dirs = {
        "root":           root,
        "summary":        root / "00_summary",
        "spec":           root / "01_spec_ir",
        "rtl":            root / "02_rtl_ir",
        "grounding":      root / "03_grounding",
        "memory":         root / "07_memory",
        "cache_snapshot": root / "08_cache_snapshot",
        "per_rule":       root / "rules",
    }

    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    return dirs


def _render_dot_to_png(dot_path: Path, png_path: Path):
    try:
        subprocess.run(
            ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  Dependency graph image → {png_path}")
        return True
    except Exception as e:
        print(f"  {Fore.YELLOW}[warn] Could not render dependency graph PNG: {e}{Style.RESET_ALL}")
        return False


def _snapshot_cache_and_memory(out_dir: Path):
    cache_dir = Path(CACHE_DIR)
    if not cache_dir.exists():
        return

    for f in cache_dir.glob("*"):
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)


def _export_input_manifest(cfg: dict, rtl_ir: dict, out_dir: Path):
    manifest = {
        "spec_file": cfg["spec_file"],
        "rtl_dir": cfg["rtl_dir"],
        "top_module": cfg.get("top_module"),
        "rtl_files": sorted(
            [
                m.get("filepath")
                for m in rtl_ir.get("modules", {}).values()
                if m.get("filepath")
            ]
        ),
    }
    _write_json(out_dir / "input_manifest.json", manifest)


def _export_grounding(grounding_result: dict, out_dir: Path):
    _write_json(out_dir / "grounding_full.json", grounding_result)

    lines = []
    lines.append("Grounding Summary")
    lines.append("=" * 80)
    lines.append(f"Alignment score: {grounding_result.get('alignment_score')}")
    lines.append(f"Unmatched signals: {len(grounding_result.get('unmatched', []))}")
    lines.append("")

    for spec_sig, g in grounding_result.get("groundings", {}).items():
        lines.append(f"[{spec_sig}]")
        lines.append(f"  rtl_name   : {g.get('rtl_name')}")
        lines.append(f"  confidence : {g.get('confidence')}")
        lines.append(f"  method     : {g.get('method')}")
        cands = g.get("candidates", [])
        if cands:
            lines.append("  candidates :")
            for cand in cands[:5]:
                lines.append(f"    - {cand}")
        lines.append("")

    if grounding_result.get("disambiguation"):
        lines.append("")
        lines.append("Disambiguation Needed")
        lines.append("=" * 80)
        for item in grounding_result["disambiguation"]:
            lines.append(f"- {item.get('spec_signal')}: {item.get('message')}")
            for cand in item.get("candidates", []):
                lines.append(f"    * {cand}")
            lines.append("")

    _write_text(out_dir / "grounding_summary.txt", "\n".join(lines))


def _format_context_text(context: dict) -> str:
    rule = context.get("rule", {})
    rtl  = context.get("rtl_slice", {})

    lines = []
    lines.append(f"Rule: {rule.get('id')}")
    lines.append("=" * 100)
    lines.append(f"Trigger    : {rule.get('trigger', {}).get('expression')}")
    lines.append(f"Obligation : {rule.get('obligation', {}).get('expression')}")
    lines.append(f"Timing     : {rule.get('timing', {})}")
    lines.append("")
    lines.append("Grounded Signals")
    lines.append("-" * 100)
    lines.append(pformat(context.get("grounded_signals", [])))
    lines.append("")
    lines.append("RTL Slice")
    lines.append("-" * 100)
    lines.append(f"Trigger signals    : {rtl.get('trigger_signals', [])}")
    lines.append(f"Obligation signals : {rtl.get('obligation_signals', [])}")
    lines.append(f"Clocks             : {rtl.get('clocks', [])}")
    lines.append(f"Resets             : {rtl.get('resets', [])}")

    return "\n".join(lines)


def _format_assertion_text(assertion: dict) -> str:
    lines = []
    lines.append("Assertion Result")
    lines.append("=" * 100)
    lines.append(f"Property name  : {assertion.get('property_name')}")
    lines.append(f"Confidence     : {assertion.get('confidence')}")
    lines.append(f"Tier1 passed   : {assertion.get('tier1_passed')}")
    lines.append(f"Tier2 passed   : {assertion.get('tier2_passed')}")
    lines.append("")
    lines.append("Warnings")
    lines.append("-" * 100)
    for w in assertion.get("warnings", []):
        lines.append(f"- {w}")
    lines.append("")
    lines.append("Reasoning")
    lines.append("-" * 100)
    lines.append(assertion.get("reasoning", ""))
    lines.append("")
    lines.append("Property Code")
    lines.append("-" * 100)
    lines.append(assertion.get("property_code", ""))
    lines.append("")
    lines.append("Assert Statement")
    lines.append("-" * 100)
    lines.append(assertion.get("assert_statement", ""))
    return "\n".join(lines)


def _format_sufficiency_text(sufficiency: dict) -> str:
    lines = []
    lines.append("Sufficiency Result")
    lines.append("=" * 100)
    lines.append(pformat(sufficiency))
    return "\n".join(lines)


def _export_global_artifacts(
    dirs: dict,
    cfg: dict,
    sess: dict,
    spec_ir: dict,
    rtl_ir: dict,
    grounding_result: dict,
):
    _write_json(dirs["summary"] / "session.json", sess)
    _export_input_manifest(cfg, rtl_ir, dirs["summary"])

    spec_dump = dump_spec_ir(spec_ir, out_dir=str(dirs["spec"]))
    rtl_dump  = dump_rtl_ir(rtl_ir, out_dir=str(dirs["rtl"]))

    dot_path = rtl_dump.get("dep_graph_dot")
    if dot_path:
        _render_dot_to_png(Path(dot_path), dirs["rtl"] / "rtl_ir" / "dep_graph.png")

    _export_grounding(grounding_result, dirs["grounding"])

    mem = load_grounding_memory()
    _write_json(dirs["memory"] / "grounding_memory.json", mem)
    _snapshot_cache_and_memory(dirs["cache_snapshot"])

    return {"spec_dump": spec_dump, "rtl_dump": rtl_dump}


def _export_rule_artifacts(rule_result: dict, dirs: dict):
    rule_id  = rule_result["rule_id"]
    rule_dir = dirs["per_rule"] / rule_id
    rule_dir.mkdir(parents=True, exist_ok=True)

    context    = rule_result.get("context", {})
    sufficiency = rule_result.get("sufficiency", {})
    assertion  = rule_result.get("assertion", {})
    rule       = rule_result.get("rule", {})

    _write_json(rule_dir / "rule.json", rule)
    _write_json(rule_dir / "context.json", context)
    _write_text(rule_dir / "context.txt", _format_context_text(context))

    _write_json(rule_dir / "sufficiency.json", sufficiency)
    _write_text(rule_dir / "sufficiency.txt", _format_sufficiency_text(sufficiency))

    _write_json(rule_dir / "assertion.json", assertion)
    _write_text(rule_dir / "assertion.txt", _format_assertion_text(assertion))


def _build_compact_results_summary(results: list) -> list:
    out = []
    for r in results:
        a = r.get("assertion", {})
        s = r.get("sufficiency", {})
        out.append(
            {
                "rule_id":           r.get("rule_id"),
                "property_name":     a.get("property_name"),
                "reasoning":         a.get("reasoning", ""),
                "property_code":     a.get("property_code", ""),
                "assert_statement":  a.get("assert_statement", ""),
                "confidence":        a.get("confidence"),
                "tier1_passed":      a.get("tier1_passed"),
                "tier2_passed":      a.get("tier2_passed"),
                "sufficiency_passed": s.get("passed"),
                "sufficiency_score": s.get("overall_score"),
                "warnings":          a.get("warnings", []),
            }
        )
    return out


# ── Per-rule pipeline ─────────────────────────────────────────────────────────

def process_rule(rule: dict, grounding_result: dict, rtl_ir: dict, spec_ir: dict) -> dict:
    context = build_context(rule, grounding_result, rtl_ir, spec_ir)

    for iteration in range(1, MAX_ITERATIONS + 1):
        sufficiency = check_sufficiency(context, iterations=iteration - 1)
        suf_str = (
            f"{Fore.GREEN}✓{Style.RESET_ALL}"
            if sufficiency["passed"]
            else f"{Fore.YELLOW}✗{Style.RESET_ALL}"
        )
        print(
            f"  Sufficiency [{iteration}/{MAX_ITERATIONS}]: "
            f"{sufficiency['overall_score']:.2f} {suf_str}  "
            f"(L1={'✓' if sufficiency['l1_pass'] else '✗'} "
            f"L2={sufficiency['l2_score']:.2f})"
        )

        if sufficiency["passed"]:
            break

        if sufficiency.get("escalate"):
            print(
                f"  {Fore.RED}⚠  Escalating to human review after "
                f"{MAX_ITERATIONS} iterations{Style.RESET_ALL}"
            )
            for c in sufficiency.get("clarifications", [])[:3]:
                print(f"    → {c['request']}")
            break

        for c in sufficiency.get("clarifications", [])[:3]:
            print(f"    {Fore.YELLOW}⚠ {c['issue']}{Style.RESET_ALL}")

    assertion = generate_assertion(context)
    assertion = validate_assertion_tier1(assertion, rtl_ir, context)

    memory_stats = grounding_result.get("memory_stats")

    update_memory(
        rule_id=rule["id"],
        groundings=grounding_result["groundings"],
        assertion=assertion,
        validation_passed=assertion.get("tier1_passed", False),
        rule=rule,
        context=context,
        memory_stats=memory_stats,
    )

    return {
        "rule_id":    rule["id"],
        "rule":       rule,
        "assertion":  assertion,
        "sufficiency": sufficiency,
        "context":    context,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(design_name: str, max_rules: int = 5) -> list:
    if design_name not in DESIGNS:
        print(
            f"{Fore.RED}Unknown design '{design_name}'. "
            f"Available: {list(DESIGNS.keys())}{Style.RESET_ALL}"
        )
        sys.exit(1)

    cfg = DESIGNS[design_name]
    _print_header(f"RTL Assertion Generator — {design_name}")

    rtl_files = []
    for root, _, files in os.walk(cfg["rtl_dir"]):
        for f in files:
            if f.endswith((".v", ".sv", ".vhd", ".vhdl")):
                rtl_files.append(os.path.join(root, f))

    # ── Phase 0 ───────────────────────────────────────────────────────────────
    sess = init_session(cfg["spec_file"], rtl_files)
    print(f"[Phase 0] Session: {sess['session_id']}  Hash: {sess['design_hash']}")

    dirs = _make_run_dirs(design_name, sess["session_id"])
    print(f"  Output folder: {dirs['root']}")

    cached = sess.get("cached_data")
    if cached and cached.get("rtl_ir"):
        print(f"  {Fore.GREEN}Cache hit — reusing RTL IR{Style.RESET_ALL}")
        rtl_ir = cached["rtl_ir"]
    else:
        rtl_ir = process_rtl(cfg["rtl_dir"], top_module=cfg.get("top_module"))
        save_design_cache(sess["design_hash"], {"rtl_ir": rtl_ir})

    check_staleness(rtl_ir)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    spec_ir = process_spec(cfg["spec_file"], rtl_ir=rtl_ir)

    if not spec_ir["rules"]:
        print(f"{Fore.YELLOW}No rules extracted from spec.{Style.RESET_ALL}")
        _write_text(dirs["summary"] / "run_summary.txt", "No rules extracted from spec.")
        return []

    chunk_stats = spec_ir.get("chunk_stats", {})
    if chunk_stats:
        saved = chunk_stats.get("skipped_tfidf", 0) + chunk_stats.get("skipped_keyword_gate", 0)
        print(
            f"  [U2] Chunks: {chunk_stats['total_chunks']} total → "
            f"{chunk_stats['tfidf_selected']} TF-IDF selected → "
            f"{chunk_stats['llm_calls_made']} LLM calls "
            f"({saved} skipped)"
        )

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    grounding_result = ground_signals(spec_ir, rtl_ir)

    if grounding_result["alignment_detail"].get("design_review_needed"):
        print(
            f"{Fore.YELLOW}⚠  Low alignment score — spec may not match this RTL."
            f"{Style.RESET_ALL}"
        )
    if grounding_result["disambiguation"]:
        print(
            f"{Fore.YELLOW}⚠  {len(grounding_result['disambiguation'])} signal(s) "
            f"require disambiguation{Style.RESET_ALL}"
        )

    mem_stats = grounding_result.get("memory_stats", {})
    if mem_stats:
        print(
            f"  [U3] Memory: {mem_stats['active_entries']}/{mem_stats['total_entries']} "
            f"active ({mem_stats['stale_excluded']} stale excluded)"
        )

    # export global artifacts early
    _export_global_artifacts(
        dirs=dirs,
        cfg=cfg,
        sess=sess,
        spec_ir=spec_ir,
        rtl_ir=rtl_ir,
        grounding_result=grounding_result,
    )

    # ── Phases 4-7 ────────────────────────────────────────────────────────────
    rules_to_process = spec_ir["rules"][:max_rules]
    print(f"\n{Fore.CYAN}Processing {len(rules_to_process)} rule(s) …{Style.RESET_ALL}")

    results = []
    for i, rule in enumerate(rules_to_process, start=1):
        if i > 1:
            time.sleep(2)
        _print_rule_header(rule, i, len(rules_to_process))

        try:
            result = process_rule(rule, grounding_result, rtl_ir, spec_ir)
            results.append(result)
            _export_rule_artifacts(result, dirs)

            a = result["assertion"]
            tier_color = Fore.GREEN if a.get("tier1_passed") else Fore.YELLOW
            print(
                f"  {tier_color}→ {a.get('property_name')} "
                f"| confidence {a.get('confidence', 0)}/100 "
                f"| tier1={'✓' if a.get('tier1_passed') else '✗'} "
                f"| tier2={'✓' if a.get('tier2_passed') else '✗'} "
                f"{Style.RESET_ALL}"
            )
            for w in a.get("warnings", [])[:3]:
                print(f"    {Fore.YELLOW}⚠ {w}{Style.RESET_ALL}")

        except Exception as e:
            print(f"  {Fore.RED}Rule {rule['id']} failed: {e}{Style.RESET_ALL}")
            fail_result = {
                "rule_id": rule["id"],
                "rule": rule,
                "assertion": {
                    "tier1_passed": False,
                    "tier2_passed": False,
                    "confidence": 0,
                    "property_code": f"// error: {e}",
                    "assert_statement": "// error",
                    "warnings": [str(e)],
                    "reasoning": f"Generation failed: {e}",
                },
                "sufficiency": {"passed": False, "overall_score": 0},
                "context": {
                    "rule": rule,
                    "grounded_signals": [],
                    "rtl_slice": {
                        "trigger_signals":    [],
                        "obligation_signals": [],
                        "clocks":             [],
                        "resets":             [],
                    },
                },
            }
            results.append(fail_result)
            _export_rule_artifacts(fail_result, dirs)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["assertion"].get("tier1_passed"))
    avg_conf = (
        sum(r["assertion"].get("confidence", 0) for r in results) / max(len(results), 1)
    )

    _print_header(
        f"Summary: {passed}/{len(results)} tier-1  |  "
        f"avg conf {avg_conf:.1f}/100"
    )

    if chunk_stats:
        total = chunk_stats.get("total_chunks", 0)
        skipped = (
            chunk_stats.get("skipped_tfidf", 0)
            + chunk_stats.get("skipped_keyword_gate", 0)
        )
        print(
            f"  [U2] LLM calls saved by TF-IDF: {skipped}/{total} chunks "
            f"({100 * skipped // max(total, 1)}%)"
        )

    if mem_stats and mem_stats.get("stale_excluded", 0) > 0:
        print(
            f"  [U3] Stale entries excluded from grounding: "
            f"{mem_stats['stale_excluded']}"
        )

    compact_summary = _build_compact_results_summary(results)
    _write_json(dirs["summary"] / "results_summary.json", compact_summary)

    summary_txt = []
    summary_txt.append(f"Design           : {design_name}")
    summary_txt.append(f"Session ID       : {sess['session_id']}")
    summary_txt.append(f"Design hash      : {sess['design_hash']}")
    summary_txt.append("")
    summary_txt.append(f"Rules processed  : {len(results)}")
    summary_txt.append(f"Tier1 passed     : {passed}")
    summary_txt.append(f"Average conf     : {avg_conf:.1f}/100")
    summary_txt.append("")
    summary_txt.append(f"Spec rules       : {len(spec_ir.get('rules', []))}")
    summary_txt.append(f"RTL modules      : {len(rtl_ir.get('modules', {}))}")
    summary_txt.append(f"RTL FSMs         : {len(rtl_ir.get('fsms', {}))}")
    summary_txt.append(f"RTL signals      : {len(rtl_ir.get('all_signals', []))}")
    summary_txt.append(f"Alignment score  : {grounding_result.get('alignment_score')}")
    summary_txt.append("")
    if grounding_result.get("memory_stats"):
        summary_txt.append("Memory stats")
        summary_txt.append("-" * 40)
        summary_txt.append(pformat(grounding_result["memory_stats"]))
        summary_txt.append("")

    summary_txt.append("Rules")
    summary_txt.append("-" * 40)
    for item in compact_summary:
        summary_txt.append(
            f"{item['rule_id']}: "
            f"{item.get('property_name')} | "
            f"conf={item.get('confidence')} | "
            f"tier1={'✓' if item.get('tier1_passed') else '✗'} | "
            f"suff={item.get('sufficiency_score')}"
        )

    _write_text(dirs["summary"] / "run_summary.txt", "\n".join(summary_txt))

    # snapshot memory/cache again after updates from per-rule processing
    _write_json(dirs["memory"] / "grounding_memory.json", load_grounding_memory())
    _snapshot_cache_and_memory(dirs["cache_snapshot"])

    # also keep a top-level pointer for quick access
    Path(CACHE_DIR).mkdir(exist_ok=True)
    latest_ptr = Path(CACHE_DIR) / f"results_{design_name}.json"
    latest_ptr.write_text(json.dumps(compact_summary, indent=2, default=str), encoding="utf-8")
    print(f"{Fore.GREEN}Results summary → {latest_ptr}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Full run folder  → {dirs['root']}{Style.RESET_ALL}")

    # Print generated assertions
    print(f"\n{Fore.CYAN}═══ Generated Assertions ═══{Style.RESET_ALL}")
    for r in results:
        a = r["assertion"]
        icon = (
            f"{Fore.GREEN}✓{Style.RESET_ALL}"
            if a.get("tier1_passed")
            else f"{Fore.YELLOW}✗{Style.RESET_ALL}"
        )
        print(f"   // {a.get('reasoning', '')}")
        print(a.get("property_code", "// N/A"))
        print(a.get("assert_statement", ""))

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <design_name> [max_rules]")
        print(f"Available designs: {list(DESIGNS.keys())}")
        sys.exit(1)

    design_name = sys.argv[1]
    max_rules = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_pipeline(design_name, max_rules)