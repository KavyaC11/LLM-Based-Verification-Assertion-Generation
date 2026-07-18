# LLM-Based Verification Assertion Generation

Automatic generation of SystemVerilog Assertions (SVA) from RTL sources and natural-language hardware specifications. The pipeline addresses long-context limits, agentic memory, multi-agent coordination, prompt caching, and feedback-driven refinement.

**Phases:** 0–10 | **Core run:** Phases 0–7 | **Advanced:** Phases 8–10

## Problem

Given RTL and a specification, produce correct SVA properties without sending entire designs to an LLM. Specs and RTL are large; signal names rarely match; generation often needs iteration and memory across runs.

## Pipeline

| Phase | Name | Role |
|-------|------|------|
| 0 | Session & cache | Design hashing; L1/L2/L3 reuse of RTL IR, context, assertions |
| 1 | Spec processing | PDF/text → Spec IR (rules) via TF-IDF ranking + LLM extraction |
| 2 | RTL processing | Verilog/SV/VHDL → RTL IR (ports, clocks/resets, FSMs, dep-graph) |
| 3 | Grounding | Spec ↔ RTL signal mapping (exact / fuzzy / graph / memory) |
| 4 | Context | Budget-aware priority slicing of relevant RTL for each rule |
| 5 | Sufficiency | Structural + semantic checks; targeted clarifications |
| 6 | Generation | LLM SVA generation; Tier-1 syntax + Tier-2 semantic checks |
| 7 | Memory | Grounding updates, CogniGraph, staleness / Weibull decay |
| 8 | Multi-agent | Specialist agents (horizontal/vertical); used on escalation |
| 9 | Relay cache | Multi-granularity prompt cache (implemented; not on default path) |
| 10 | Self-refine | iGRPO-style drafts + SELF-REFINE loop; can escalate to Phase 8 |

**Entry points**

- `main.py` — Phases 0–7 (baseline end-to-end)
- `feed.py` — Phases 0–5 + Phase 10 (self-refine) + formal eval via `eval.py`

## Setup

**Requirements:** Python 3.11+, [Ollama](https://ollama.com/) with a pulled model (default: `llama3.1:latest` in `config.py`). Optional: Graphviz (`dot`), SymbiYosys (`sby`) for `feed.py` formal checks.

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Unix:    source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:latest
```

Model and thresholds live in `config.py`. Override host with `OLLAMA_HOST` if needed.

## Usage

```bash
python main.py <design_name> [max_rules]
python feed.py <design_name> [max_rules]
```

```bash
python main.py spi_master 5
python feed.py ethmac 10
```

Registered designs (`config.py`): `ethmac`, `ethernet`, `i2c`, `spi_master`, `sockit`, `openMSP430`, `crypto_bridge`.

Artifacts: `results/<design>/…` | Caches: `cache/` (both gitignored)

## Repository layout

```
config.py, main.py, feed.py, eval.py
phase0_session.py … phase10_self_refine.py
designs/    # RTL
specs/      # PDF / TXT specifications
```

## Design note

The accompanying system design document describes Groq (`llama-3.3-70b-versatile`) as the LLM backend; this codebase uses **Ollama** locally. Phases 8 and 10 are wired through `feed.py`; Phase 9 is present for integration but not invoked by the default orchestrators.
