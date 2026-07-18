"""
Phase 2 — RTL Processing
"""  

import os
import re
import json
from pathlib import Path
from collections import defaultdict

RTL_EXTS = {".v", ".sv", ".vhd", ".vhdl"}


def get_rtl_files(rtl_dir: str) -> list[str]:
    files = []
    for root, _, fnames in os.walk(rtl_dir):
        for f in fnames:
            if Path(f).suffix.lower() in RTL_EXTS:
                files.append(os.path.join(root, f))
    return sorted(files)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_width(hi_str, lo_str) -> int:
    try:
        return abs(int(hi_str) - int(lo_str)) + 1
    except Exception:
        return 1


def _strip_comments(text: str) -> str:
    """Remove // line comments and /* block */ comments."""
    text = re.sub(r"//[^\n]*",        "",  text)
    text = re.sub(r"/\*.*?\*/",       "",  text, flags=re.DOTALL)
    return text


def _lhs_signal(lhs_raw: str) -> str:
    """
    Extract the base signal name from an LHS token.
    Handles:  csr[7]  ->  csr
              shift_reg[6:0]  ->  shift_reg
              plain_sig  ->  plain_sig
    """
    return re.sub(r"\s*\[.*?\]", "", lhs_raw).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Verilog / SystemVerilog parser
# ─────────────────────────────────────────────────────────────────────────────

# Matches:  input [1:0] bus_adr
#           output reg [7:0] bus_rdt
#           input wire clk
_PORT_RE = re.compile(
    r"\b(input|output|inout)\s+"
    r"(?:wire|reg|logic)?\s*"
    r"(?:signed\s+)?"
    r"(?:\[(\d+)\s*:\s*(\d+)\]\s+)?"
    r"(\w+)"
)

# Matches internal reg/wire/logic declarations
# We intentionally skip matches whose name is already a port
_SIG_RE = re.compile(
    r"\b(reg|wire|logic)\s+"
    r"(?:signed\s+)?"
    r"(?:\[(\d+)\s*:\s*(\d+)\]\s+)?"
    r"(\w+)"
)

# localparam / parameter  — used as FSM state candidates
_PARAM_RE = re.compile(
    r"\b(?:localparam|parameter)\b\s+"
    r"(?:\w+\s+)?"          # optional type
    r"(\w+)\s*="
)

def _is_reset_like(sig: str) -> bool:
    s = sig.lower()

    # direct/common reset forms
    if s in {"rst", "reset", "arst", "rst_n", "reset_n", "arst_n"}:
        return True

    # catch embedded forms too: TxReset, RxReset, wb_rst_i, async_reset, etc.
    if "reset" in s or "arst" in s:
        return True

    # catch rst as a token-ish fragment, but avoid matching unrelated words blindly
    if re.search(r"(?:^|_|[a-z])rst(?:$|_|[a-z0-9])", s):
        return True

    return False

def parse_module_verilog(filepath: str) -> dict:
    raw_text = Path(filepath).read_text(errors="replace", encoding="utf-8")
    text     = _strip_comments(raw_text)

    # Module name
    mm = re.search(r"\bmodule\s+(\w+)", text)
    module_name = mm.group(1) if mm else Path(filepath).stem

    # Ports
    ports: list[dict] = []
    port_names: set[str] = set()
    for m in _PORT_RE.finditer(text):
        name = m.group(4)
        if name in port_names:
            continue
        port_names.add(name)
        ports.append({
            "name":      name,
            "direction": m.group(1),
            "width":     _infer_width(m.group(2) or "0", m.group(3) or "0"),
        })

    # Internal signals — skip anything already declared as a port
    signals: list[dict] = []
    seen_sigs: set[str] = set(port_names)
    for m in _SIG_RE.finditer(text):
        name = m.group(4)
        if name in seen_sigs:
            continue
        seen_sigs.add(name)
        signals.append({
            "name":  name,
            "type":  m.group(1),
            "width": _infer_width(m.group(2) or "0", m.group(3) or "0"),
        })

    # Clocks / resets
    _EDGE_RE = re.compile(r"\b(posedge|negedge)\s+([A-Za-z_][A-Za-z0-9_$]*)", re.IGNORECASE)
    _RST_IF_RE = re.compile(r"\bif\s*\(\s*[!~]?\s*([A-Za-z_][A-Za-z0-9_$]*)\s*\)", re.IGNORECASE)

    clocks_pos = sorted({
        m.group(2) for m in _EDGE_RE.finditer(text)
        if m.group(1).lower() == "posedge" and not _is_reset_like(m.group(2))
    })

    clocks_neg = sorted({
        m.group(2) for m in _EDGE_RE.finditer(text)
        if m.group(1).lower() == "negedge"
        and not _is_reset_like(m.group(2))
        and m.group(2) not in clocks_pos
    })

    resets = sorted({
        m.group(2) for m in _EDGE_RE.finditer(text)
        if _is_reset_like(m.group(2))
    } | {
        m.group(1) for m in _RST_IF_RE.finditer(text)
        if _is_reset_like(m.group(1))
    })

    # Parameters / localparams (used as FSM state name candidates)
    params = [m.group(1) for m in _PARAM_RE.finditer(text)]

    # Submodule instantiations
    _INST_RE  = re.compile(r"^\s*(\w+)\s+(\w+)\s*\(", re.MULTILINE)
    sv_kw = {
        "module","endmodule","input","output","inout","assign","always",
        "initial","begin","end","wire","reg","logic","parameter","localparam",
        "if","else","case","casex","casez","endcase","for","function",
        "endfunction","task","endtask","generate","endgenerate","posedge","negedge",
    }
    submodules = sorted({
        m.group(1) for m in _INST_RE.finditer(text)
        if m.group(1) not in sv_kw
    })

    return {
        "module_name": module_name,
        "filepath":    filepath,
        "ports":       ports,
        "signals":     signals,
        "clocks":      clocks_pos + clocks_neg,
        "resets":      resets,
        "params":      params,
        "submodules":  submodules,
        "raw_text":    raw_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.1  VHDL parser 
# ─────────────────────────────────────────────────────────────────────────────

def parse_module_vhdl(filepath: str) -> dict:
    raw_text = Path(filepath).read_text(errors="replace", encoding="utf-8")
    text     = _strip_comments(raw_text)

    em = re.search(r"\bentity\s+(\w+)\s+is", text, re.IGNORECASE)
    module_name = em.group(1) if em else Path(filepath).stem

    ports: list[dict] = []
    for m in re.finditer(
        r"(\w+)\s*:\s*(in|out|inout)\s+"
        r"(?:std_logic_vector\s*\(\s*(\d+)\s+downto\s+(\d+)\s*\)|std_logic)",
        text, re.IGNORECASE
    ):
        ports.append({
            "name":      m.group(1),
            "direction": m.group(2).lower(),
            "width":     _infer_width(m.group(3) or "0", m.group(4) or "0"),
        })

    clocks = sorted({
        m.group(1) for m in re.finditer(r"rising_edge\s*\(\s*(\w+)\s*\)", text, re.IGNORECASE)
    })
    resets = sorted({
        m.group(1) for m in re.finditer(r"\bif\s+(\w+)\s*=\s*['\"]0['\"]", text, re.IGNORECASE)
        if re.search(r"rst|reset", m.group(1), re.IGNORECASE)
    })

    return {
        "module_name": module_name,
        "filepath":    filepath,
        "ports":       ports,
        "signals":     [],
        "clocks":      clocks,
        "resets":      resets,
        "params":      [],
        "submodules":  [],
        "raw_text":    raw_text,
    }


def parse_module(filepath: str) -> dict:
    ext = Path(filepath).suffix.lower()
    return parse_module_vhdl(filepath) if ext in (".vhd", ".vhdl") else parse_module_verilog(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# 2.2  FSM detection
# ─────────────────────────────────────────────────────────────────────────────

_STATE_REG_RE = re.compile(
    r"\b(state|current_state|c_state|cs|st|cur_state|nstate|next_state|ps)\b",
    re.IGNORECASE,
)


def detect_fsms(filepath: str, module_name: str, params: list[str]) -> list[dict]:
    text = _strip_comments(Path(filepath).read_text(errors="replace", encoding="utf-8"))

    state_regs = sorted({m.group(0) for m in _STATE_REG_RE.finditer(text)})
    if not state_regs:
        return []

    # State name candidates:
    #   (a) ALL-CAPS identifiers in the file
    #   (b) localparam / parameter names
    allcaps = {
        s for s in re.findall(r"\b([A-Z][A-Z0-9_]{1,30})\b", text)
        if s not in {"TX","RX","OR","AND","NOT","IN","OUT","XOR","NOR","NAND",
                     "HIGH","LOW","TRUE","FALSE","IDLE"}
    }
    # Add IDLE back — it's a valid state name
    allcaps.add("IDLE")
    state_name_candidates = allcaps | set(params)

    # Confirm only candidates actually assigned to a state register
    confirmed: set[str] = set()
    for reg in state_regs:
        for m in re.finditer(
            rf"\b{re.escape(reg)}\s*<=\s*(\w+)", text, re.IGNORECASE
        ):
            target = m.group(1)
            if target in state_name_candidates:
                confirmed.add(target)

    if not confirmed:
        # Fallback: any candidate that appears in a case branch
        for cand in state_name_candidates:
            if re.search(rf"\b{re.escape(cand)}\s*:", text):
                confirmed.add(cand)

    if not confirmed:
        return []

    # Transitions: from_state : begin ... state <= to_state
    transitions: list[dict] = []
    seen_trans: set[tuple] = set()
    for m in re.finditer(
        r"\b(\w+)\s*:\s*begin(.*?)end",
        text, re.DOTALL
    ):
        from_s = m.group(1)
        if from_s not in confirmed:
            continue
        block = m.group(2)
        for reg in state_regs:
            for am in re.finditer(
                rf"\b{re.escape(reg)}\s*<=\s*(\w+)", block, re.IGNORECASE
            ):
                to_s = am.group(1)
                if to_s in confirmed and (from_s, to_s) not in seen_trans:
                    seen_trans.add((from_s, to_s))
                    transitions.append({"from": from_s, "to": to_s})

    fsm_name = re.sub(r"[^a-z0-9_]", "", state_regs[0].lower()) + "_fsm"
    return [{
        "fsm_name":       fsm_name,
        "module":         module_name,
        "state_register": state_regs[0],
        "states":         sorted(confirmed),
        "transitions":    transitions[:60],
    }]


# ─────────────────────────────────────────────────────────────────────────────
# 2.4  Dependency graph (DAG)
# ─────────────────────────────────────────────────────────────────────────────

def build_dependency_graph(filepath: str, all_signal_names: set[str]) -> dict:
    """
    Build a signal-level DAG from one RTL file.

    Edge types
    ──────────
    data     : signal A is read to compute signal B  (B depends_on A)
    control  : signal A gates a branch that assigns B (B depends_on A, type=control)
    self     : B <= B + 1  (kept, marks a register with feedback)

    Each node also stores:
      clock   : clock signal name (from posedge/negedge sensitivity)
      reset   : reset signal name (from actual reset branch assignment — B4)
      reset_val: value assigned on reset
    """
    raw_text = Path(filepath).read_text(errors="replace", encoding="utf-8")
    text     = _strip_comments(raw_text)

    # node template
    def _new_node():
        return {
            "depends_on":  [],   # list of {signal, edge_type}
            "drives":      [],   # signal names this signal drives
            "clock":       None,
            "reset":       None,
            "reset_val":   None,
        }

    graph: dict[str, dict] = defaultdict(_new_node)

    def _add_data_edge(lhs: str, rhs_sigs: list[str]):
        for s in rhs_sigs:
            if s not in all_signal_names or s == lhs:
                continue
            entry = {"signal": s, "type": "data"}
            if entry not in graph[lhs]["depends_on"]:
                graph[lhs]["depends_on"].append(entry)
            if lhs not in graph[s]["drives"]:
                graph[s]["drives"].append(lhs)

    def _add_ctrl_edge(lhs: str, ctrl_sigs: list[str]):
        for s in ctrl_sigs:
            if s not in all_signal_names or s == lhs:
                continue
            entry = {"signal": s, "type": "control"}
            if entry not in graph[lhs]["depends_on"]:
                graph[lhs]["depends_on"].append(entry)
            if lhs not in graph[s]["drives"]:
                graph[s]["drives"].append(lhs)

    def _rhs_signals(expr: str) -> list[str]:
        return [s for s in re.findall(r"\b([a-zA-Z_]\w*)\b", expr)
                if s in all_signal_names]

    # ── Continuous assignments ─────────────────────────────────────────────
    for m in re.finditer(r"\bassign\s+(\w+(?:\[[\d:]+\])?)\s*=\s*(.+?);", text, re.DOTALL):
        lhs = _lhs_signal(m.group(1))
        if lhs not in all_signal_names:
            continue
        rhs = _rhs_signals(m.group(2))
        _add_data_edge(lhs, rhs)

    # ── Sequential / combinational always blocks ───────────────────────────
    # Split the file into always blocks first so we can tag clock/reset per block
    _ALWAYS_RE = re.compile(
        r"always\s*@\s*\((.*?)\)(.*?)(?=\balways\b|\bendmodule\b|$)",
        re.DOTALL,
    )

    for blk in _ALWAYS_RE.finditer(text):
        sensitivity = blk.group(1)
        body        = blk.group(2)

        # Clock from sensitivity list
        clk_m = re.search(r"(?:posedge|negedge)\s+(\w+)", sensitivity)
        clk   = clk_m.group(1) if clk_m else None

        # ── Reset branch extraction ──────────────────────────────────
        # Find the outermost if(!rst / if(~rst condition and extract body
        rst_branch_re = re.compile(
            r"if\s*\(\s*[!~]?\s*(\w+)\s*\)\s*begin(.*?)end",
            re.DOTALL,
        )
        rst_signal = None
        for rm in rst_branch_re.finditer(body):
            candidate = rm.group(1)
            if re.search(r"rst|reset|arst", candidate, re.IGNORECASE):
                rst_signal   = candidate
                rst_body     = rm.group(2)
                # Tag every signal assigned inside this reset branch
                for am in re.finditer(r"(\w+(?:\[\d+\])?)\s*<=\s*(.+?);", rst_body):
                    sig = _lhs_signal(am.group(1))
                    if sig in all_signal_names:
                        graph[sig]["reset"]     = rst_signal
                        graph[sig]["reset_val"] = am.group(2).strip()
                        if clk:
                            graph[sig]["clock"] = clk
                break   # only the first/outermost reset branch per block

        # ── Data / control edges from all assignments in this block ───────
        # We walk every non-blocking and blocking assignment.
        # For each one we also look at the enclosing if/case condition
        # to extract control edges.

        # Simple strategy: find all LHS <= RHS pairs
        for am in re.finditer(r"(\w+(?:\[\d+\])?)\s*<=\s*(.+?);", body):
            lhs = _lhs_signal(am.group(1))
            if lhs not in all_signal_names:
                continue
            rhs = _rhs_signals(am.group(2))
            _add_data_edge(lhs, rhs)
            if clk:
                graph[lhs]["clock"] = clk

        # Blocking assignments in combinational blocks
        for am in re.finditer(r"(\w+(?:\[\d+\])?)\s*=\s*(.+?);", body):
            lhs = _lhs_signal(am.group(1))
            if lhs not in all_signal_names:
                continue
            # Skip if/for/while keywords mistaken for assignments
            if lhs in {"if", "for", "while", "case"}:
                continue
            rhs = _rhs_signals(am.group(2))
            _add_data_edge(lhs, rhs)

        # Control edges: if (condition) → extract signals from condition
        # and add control edge to everything assigned inside that if block
        for if_m in re.finditer(
            r"if\s*\((.+?)\)\s*begin(.*?)end", body, re.DOTALL
        ):
            cond_sigs  = _rhs_signals(if_m.group(1))
            inner_lhss = [
                _lhs_signal(am.group(1))
                for am in re.finditer(r"(\w+(?:\[\d+\])?)\s*<=\s*", if_m.group(2))
            ]
            for lhs in inner_lhss:
                if lhs in all_signal_names:
                    _add_ctrl_edge(lhs, cond_sigs)

    return {k: dict(v) for k, v in graph.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Main entry
# ─────────────────────────────────────────────────────────────────────────────

def process_rtl(rtl_dir: str, top_module: str = None) -> dict:
    print(f"[Phase 2] Processing RTL: {rtl_dir}")
    rtl_files = get_rtl_files(rtl_dir)
    print(f"  Found {len(rtl_files)} file(s): {[Path(f).name for f in rtl_files]}")

    modules:          dict = {}
    all_fsms:         dict = {}
    signal_to_module: dict = {}
    dep_graph:        dict = {}

    for fpath in rtl_files:
        mod  = parse_module(fpath)
        name = mod["module_name"]
        if not name:
            continue

        modules[name] = {
            "ports":      mod["ports"],
            "signals":    mod["signals"],
            "clocks":     mod["clocks"],
            "resets":     mod["resets"],
            "params":     mod["params"],
            "submodules": mod["submodules"],
            "filepath":   fpath,
        }

        # signal_to_module: ports take priority; internals fill in
        if top_module is None or name == top_module:
            for sig in mod["ports"]:
                signal_to_module.setdefault(sig["name"], name)
            for sig in mod["signals"]:
                signal_to_module.setdefault(sig["name"], name)

        for fsm in detect_fsms(fpath, name, mod["params"]):
            all_fsms[fsm["fsm_name"]] = fsm

    # Build dependency graph over full signal universe
    all_sig_names = set(signal_to_module.keys())
    for name, meta in modules.items():
        if top_module is None or name == top_module:
            g = build_dependency_graph(meta["filepath"], all_sig_names)
            dep_graph.update(g)

    rtl_ir = {
        "modules":          modules,
        "fsms":             all_fsms,
        "signal_to_module": signal_to_module,
        "dep_graph":        dep_graph,
        "all_signals":      sorted(signal_to_module.keys()),
    }

    print(f"  → {len(modules)} module(s)  |  {len(all_fsms)} FSM(s)  |  "
          f"{len(all_sig_names)} signal(s)  |  {len(dep_graph)} dep-graph node(s)")
    return rtl_ir


# ─────────────────────────────────────────────────────────────────────────────
# Intermediate output dump
# ─────────────────────────────────────────────────────────────────────────────

def dump_rtl_ir(rtl_ir: dict, out_dir: str = "cache") -> dict[str, Path]:
    """
    Write the RTL IR to human-readable files under `out_dir/rtl_ir/`.

    Files produced
    ──────────────
    rtl_ir_full.json      — complete IR (machine-readable)
    ports.txt             — all ports per module with direction + width
    signals.txt           — all internal signals per module
    fsms.txt              — detected FSMs with states + transitions
    dep_graph.txt         — DAG edges: signal → depends_on / drives
    dep_graph_dot.dot     — Graphviz DOT format for visual inspection
    """
    base = Path(out_dir) / "rtl_ir"
    base.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # 1. Full JSON
    full_path = base / "rtl_ir_full.json"
    full_path.write_text(json.dumps(rtl_ir, indent=2, default=str), encoding="utf-8")
    written["full_json"] = full_path

    # 2. Ports per module
    ports_path = base / "ports.txt"
    lines = ["RTL IR — Ports", "=" * 60, ""]
    for mod_name, mod in rtl_ir["modules"].items():
        lines.append(f"Module: {mod_name}  ({mod['filepath']})")
        lines.append(f"  Clocks : {mod['clocks']}")
        lines.append(f"  Resets : {mod['resets']}")
        lines.append(f"  Params : {mod['params'][:10]}")
        lines.append(f"  {'Signal':<20} {'Dir':<10} {'Width'}")
        lines.append(f"  {'-'*45}")
        for p in mod["ports"]:
            lines.append(f"  {p['name']:<20} {p['direction']:<10} {p['width']}")
        lines.append("")
    ports_path.write_text("\n".join(lines), encoding="utf-8")
    written["ports"] = ports_path

    # 3. Internal signals per module
    sigs_path = base / "signals.txt"
    lines = ["RTL IR — Internal Signals", "=" * 60, ""]
    for mod_name, mod in rtl_ir["modules"].items():
        lines.append(f"Module: {mod_name}")
        lines.append(f"  {'Signal':<20} {'Type':<8} {'Width'}")
        lines.append(f"  {'-'*35}")
        for s in mod["signals"]:
            lines.append(f"  {s['name']:<20} {s['type']:<8} {s['width']}")
        lines.append("")
    sigs_path.write_text("\n".join(lines), encoding="utf-8")
    written["signals"] = sigs_path

    # 4. FSMs
    fsm_path = base / "fsms.txt"
    lines = ["RTL IR — FSMs", "=" * 60, ""]
    if not rtl_ir["fsms"]:
        lines.append("  (no FSMs detected)")
    for fsm_name, fsm in rtl_ir["fsms"].items():
        lines.append(f"FSM: {fsm_name}  (module: {fsm['module']})")
        lines.append(f"  State register : {fsm['state_register']}")
        lines.append(f"  States         : {fsm['states']}")
        lines.append(f"  Transitions ({len(fsm['transitions'])}):")
        for t in fsm["transitions"]:
            lines.append(f"    {t['from']:15s} ──→  {t['to']}")
        lines.append("")
    fsm_path.write_text("\n".join(lines), encoding="utf-8")
    written["fsms"] = fsm_path

    # 5. Dependency graph — human readable
    dep_path = base / "dep_graph.txt"
    lines = ["RTL IR — Dependency Graph", "=" * 60,
             "Format:  signal  [clk=X  rst=Y]",
             "           depends_on: signal (edge_type)",
             "           drives    : signal", ""]
    dep = rtl_ir["dep_graph"]
    for sig in sorted(dep.keys()):
        node = dep[sig]
        deps  = node.get("depends_on", [])
        drvs  = node.get("drives", [])
        if not deps and not drvs:
            continue
        clk_tag = f"clk={node['clock']}" if node.get("clock") else ""
        rst_tag = f"rst={node['reset']}={node.get('reset_val','?')}" \
                  if node.get("reset") else ""
        meta = "  ".join(filter(None, [clk_tag, rst_tag]))
        lines.append(f"  {sig:<20}  {meta}")
        for d in deps:
            if isinstance(d, dict):
                lines.append(f"    ← {d['signal']:<20} ({d['type']})")
            else:
                lines.append(f"    ← {d:<20}")
        for d in drvs:
            lines.append(f"    → {d}")
        lines.append("")
    dep_path.write_text("\n".join(lines), encoding="utf-8")
    written["dep_graph"] = dep_path

    # 6. DOT file for Graphviz  (open with: dot -Tpng dep_graph_dot.dot -o graph.png)
    dot_path = base / "dep_graph_dot.dot"
    dot_lines = [
        "digraph RTL_DAG {",
        '  rankdir=LR;',
        '  node [shape=box fontname="Courier" fontsize=10];',
        "",
    ]
    # Colour nodes by type
    for mod_name, mod in rtl_ir["modules"].items():
        port_names = {p["name"] for p in mod["ports"]}
        for p in mod["ports"]:
            colour = "lightblue" if p["direction"] == "input" else "lightyellow"
            dot_lines.append(f'  "{p["name"]}" [style=filled fillcolor={colour} label="{p["name"]}\\n({p["direction"]},{p["width"]}b)"];')
        for s in mod["signals"]:
            dot_lines.append(f'  "{s["name"]}" [style=filled fillcolor=lightgrey label="{s["name"]}\\n({s["type"]},{s["width"]}b)"];')

    dot_lines.append("")
    for sig, node in rtl_ir["dep_graph"].items():
        for d in node.get("depends_on", []):
            src = d["signal"] if isinstance(d, dict) else d
            etype = d.get("type", "data") if isinstance(d, dict) else "data"
            style = 'style=dashed color=red' if etype == "control" else 'color=black'
            dot_lines.append(f'  "{src}" -> "{sig}" [{style}];')

    dot_lines += ["}", ""]
    dot_path.write_text("\n".join(dot_lines), encoding="utf-8")
    written["dep_graph_dot"] = dot_path

    print(f"\n[dump] RTL IR written to: {base}/")
    for key, p in written.items():
        print(f"  {key:20s} → {p.name}")
    return written

def run_rtl_parser():
    rtl_dir = "designs/ethmac/rtl/verilog"   # change this to your RTL folder path

    rtl_ir = process_rtl(rtl_dir)

    # Pretty print summary
    print("\n=== RTL SUMMARY ===")
    print("Modules:", list(rtl_ir["modules"].keys()))
    print("Total signals:", len(rtl_ir["all_signals"]))
    print("FSMs:", list(rtl_ir["fsms"].keys()))

    # Dump outputs (like spec phase)
    dump_rtl_ir(rtl_ir)


if __name__ == "__main__":
    run_rtl_parser()