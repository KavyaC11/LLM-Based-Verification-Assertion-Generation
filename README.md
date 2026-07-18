# LLM-Based RTL Assertion Generation

Automatic generation of SystemVerilog Assertions (SVA) from RTL sources and natural-language hardware specifications. The system manages long context, persistent memory across runs, multi-agent coordination, prompt caching, and iterative refinement of LLM outputs.

**Phases 0–10** · Core path: **0–7** · Advanced modules: **8–10**

## Overview

Given a design’s RTL and its specification, the pipeline extracts behavioral rules, maps them to real signals, assembles a minimal relevant context, and generates validated SVA properties. Outputs and caches are written under `results/` and `cache/`.

## Pipeline

| Phase | Implementation |
|-------|----------------|
| **0 — Session & cache** | Hashes RTL + spec content; reuses L1 RTL IR, L2 context packages, and L3 validated assertions when the design is unchanged. |
| **1 — Spec processing** | Extracts text (PDF/TXT), strips boilerplate, ranks chunks with TF-IDF + hardware keywords, then uses the LLM to emit structured rules (trigger, obligation, timing). |
| **2 — RTL processing** | Regex-parses Verilog/SystemVerilog/VHDL into a unified IR: ports, internals, clocks/resets, FSMs, and a signal dependency graph. |
| **3 — Grounding** | Aligns spec names to RTL signals via exact match, fuzzy matching, dependency-graph bonuses, and historical grounding memory with staleness filtering. |
| **4 — Context** | Builds a per-rule package with a budget-aware priority slicer (`confidence × 1/(hop+1)`), always including clocks/resets within a token budget. |
| **5 — Sufficiency** | Scores structural and semantic completeness; emits targeted clarification requests and can iterate before escalating for review. |
| **6 — Generation** | Prompts the LLM for SVA, then applies Tier-1 syntax checks and Tier-2 signal-validity checks with limited retries. |
| **7 — Memory** | Updates grounding confidence, assertion store, and CogniGraph; applies Weibull decay and prunes stale mappings for later runs. |
| **8 — Multi-agent** | Runs specialist roles (trigger, obligation, timing, synthesizer, reviewer) with redundancy/synergy controls; selectable horizontal or vertical flow. |
| **9 — Relay cache** | Multi-granularity prompt/chunk/reasoning reuse with similarity gating and validation before serving cached results. |
| **10 — Self-refine** | Samples multiple drafts, scores them, and refines with multi-aspect feedback; may escalate to multi-agent when single-shot refinement is insufficient. |

**Entry points:** `main.py` runs phases 0–7 end-to-end. `feed.py` uses self-refinement (phase 10) and optional formal evaluation via `eval.py`.

## Setup

- Python 3.11+
- Local LLM via [Ollama](https://ollama.com/) (model set in `config.py`, default `llama3.1:latest`)
- Optional: Graphviz (`dot`) for dependency-graph images; SymbiYosys (`sby`) for formal checks in `feed.py`

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Unix:     source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:latest
```

Thresholds, model name, and design paths are configured in `config.py`. The Ollama host defaults to `http://localhost:11434` (`OLLAMA_HOST`).

## Usage

```bash
python main.py <design_name> [max_rules]
python feed.py <design_name> [max_rules]
```

Examples:

```bash
python main.py spi_master 5
python feed.py ethmac 10
```

Designs registered in `config.py`: `ethmac`, `ethernet`, `i2c`, `spi_master`, `sockit`, `openMSP430`, `crypto_bridge`.

## Layout

```
config.py                 Model, thresholds, design registry
main.py                   Baseline orchestrator (phases 0–7)
feed.py                   Self-refine orchestrator + formal hook
eval.py                   Isolated SymbiYosys evaluation
phase0_session.py …       Session / cache through self-refine
phase10_self_refine.py
designs/                  RTL sources
specs/                    Specification PDFs / text
```

`venv/`, `cache/`, and `results/` are gitignored. To add a design, place RTL under `designs/`, a spec under `specs/`, and register it in `DESIGNS`.
