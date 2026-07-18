"""
Phase 1 — Spec Processing
"""

import re
import json
import time
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import ollama
from config import (
    OLLAMA_HOST, OLLAMA_MODEL,
    TFIDF_TOP_K, TFIDF_SIM_THRESHOLD,
)


_client = ollama.Client(host=OLLAMA_HOST)


# ─────────────────────────────────────────────────────────────────────────────
# 1.1  Text extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    if not HAS_PDF:
        raise ImportError("Install pdfplumber: pip install pdfplumber")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def extract_text(spec_file: str) -> str:
    p = Path(spec_file)
    if not p.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_file}")
    if p.suffix.lower() == ".pdf":
        return extract_text_from_pdf(spec_file)
    return p.read_text(errors="replace", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Boilerplate stripper
# ─────────────────────────────────────────────────────────────────────────────

_BOILERPLATE_SECTION_RE = re.compile(
    r"^\s*("
    r"table of contents|list of (tables|figures|drawings|abbreviations)|"
    r"references|license|copyright|demo (hardware|software)|"
    r"(altera|sopc|nios|quartus|bsp|hal)\b.*integration|"
    r"adding (support|the component)|configuring the component|"
    r"software driver|public domain kit|testing todo|"
    r"abbreviations.*terminology|index of|drawing index"
    r")",
    re.IGNORECASE,
)
_NOISE_LINE_RE = re.compile(
    r"^\s*("
    r"\d+(\.\d+)*\s+\w.*\.*\d+\s*$"
    r"|table\s+\d+\s*:"
    r"|drawing\s+\d+\s*:"
    r"|figure\s+\d+\s*:"
    r"|\d{1,3}\s*$"
    r"|https?://\S+"
    r")",
    re.IGNORECASE,
)


def _strip_boilerplate(text: str) -> str:
    lines  = text.splitlines()
    output = []
    skip   = False
    for line in lines:
        stripped = line.strip()
        if _BOILERPLATE_SECTION_RE.match(stripped):
            skip = True
            continue
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", stripped) and \
                not _BOILERPLATE_SECTION_RE.match(stripped):
            skip = False
        if skip:
            continue
        if _NOISE_LINE_RE.match(stripped):
            continue
        output.append(line)
    return "\n".join(output)


# ─────────────────────────────────────────────────────────────────────────────
# Strict RTL-signal lexical filter 
# ─────────────────────────────────────────────────────────────────────────────

_ENGLISH_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "her", "was", "one", "our", "out", "day", "get", "has", "him",
    "his", "how", "its", "may", "new", "now", "old", "see", "two",
    "way", "who", "did", "let", "put", "say", "she", "too", "use",
    "acknowledge", "acknowledgement", "address", "addresses", "addressed",
    "enable", "enables", "enabled",
    "ready", "receive", "receiver", "received", "reception",
    "register", "registers", "registered",
    "request", "requests", "requested",
    "transmit", "transmitter", "transmitted", "transmission",
    "output", "outputs", "input", "inputs",
    "select", "selected", "selection",
    "write", "writes", "written", "read", "reads",
    "error", "errors", "start", "started", "starting",
    "done", "busy", "valid", "ready", "idle",
    "data", "clock", "reset", "signal", "signals",
    "cycle", "cycles", "state", "states", "mode",
    "byte", "bit", "word", "frame", "packet", "block",
    "high", "low", "active", "inactive",
    "master", "slave", "device", "system",
    "interrupt", "interrupts",
    "when", "then", "upon", "after", "before", "during",
    "must", "shall", "should", "will", "can", "may",
    "this", "that", "with", "from", "into", "over",
    "each", "both", "same", "next", "last", "first",
    "only", "also", "thus", "note", "see",
    "operation", "interface", "access", "transfer", "port",
    "internal", "external", "initial", "default",
    "configuration", "control", "status", "information",
    "answer", "between", "lower", "newer", "package", "powered",
    "slower", "slowest", "stack", "testbench", "weak", "web", "were",
    "wire", "wires", "width", "value", "values", "version",
    "base", "time", "period", "rate", "module", "modules",
}

_RTL_PREFIXES = re.compile(
    r"^(i_|o_|w_|r_|s_|u_|n_|m_|g_|c_|d_|q_|z_|p_|b_|f_|v_)",
    re.IGNORECASE,
)
_RTL_SUFFIXES = re.compile(
    r"(_n|_b|_p|_q|_d|_r|_s|_z|_en|_we|_re|_oe|_ie|_clk|_rst|_req|"
    r"_ack|_vld|_rdy|_stb|_cyc|_sel|_dat|_adr|_err|_rty|_int|_irq|"
    r"_out|_in|_sig|_reg|_mem|_ff|_lat|_bus|_gate|_mux|_cnt|_ctr|"
    r"_sync|_async|_fifo|_buf|_pipe|_flag|_bit|_lo|_hi|_msb|_lsb|"
    r"_wen|_ren|_wdt|_rdt|_i|_e|_o)$",
    re.IGNORECASE,
)
_RTL_FRAGMENT_RE = re.compile(
    r"(clk|rst|mclk|smclk|aclk|irq|nmi|"
    r"txd|rxd|wen|ren|wdt|rdt|adr|"
    r"stb|cyc|ack|err|rty|owr|cdr|ovd)",
    re.IGNORECASE,
)
_RTL_FRAGMENT_PARTS = {
    "clk", "rst", "en", "ack", "req",
    "adr", "wen", "ren", "wdt", "rdt",
    "irq", "int", "we", "stb", "cyc",
    "err", "rty", "vld", "rdy", "owr",
    "cdr", "ovd", "pwr", "sel", "dat",
    "ien", "pls", "n", "o", "e", "p", "i",
}

# heuristically decides whether a word looks like an RTL signal/port name
def _is_rtl_signal_token(tok: str) -> bool:
    tok_lo = tok.lower()
    if len(tok) < 2 or tok.isdigit():
        return False
    if tok_lo in _ENGLISH_STOPWORDS:
        return False
    if tok_lo.isalpha() and "_" not in tok_lo and len(tok) <= 7:
        if not _RTL_FRAGMENT_RE.search(tok_lo):
            return False
    if _RTL_PREFIXES.match(tok) or _RTL_SUFFIXES.search(tok_lo):
        return True
    if _RTL_FRAGMENT_RE.search(tok_lo):
        return True
    parts = tok_lo.split("_")
    if len(parts) >= 2 and any(p in _RTL_FRAGMENT_PARTS for p in parts):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 — Port vocabulary extraction from spec text
# ─────────────────────────────────────────────────────────────────────────────

_NON_PORT = {
    "the", "and", "for", "but", "not", "all", "can",
    "must", "shall", "should", "will", "may",
    "this", "that", "with", "from", "into", "over",
    "each", "both", "same", "next", "last", "first",
    "also", "note", "see", "only",
    "base", "time", "period", "mode", "rate", "value",
    "register", "registers", "module", "modules",
    "signal", "signals", "cycle", "cycles",
    "clock", "reset", "data", "state",
}

# pulls likely port/signal names out of the spec using table patterns and repeated RTL-like tokens
def extract_port_names_from_spec(text: str) -> list[str]:
    return list(extract_ports_with_widths_from_spec(text).keys())

# ─────────────────────────────────────────────────────────────────────────────
# Port/signal name + width extraction from spec text
# ─────────────────────────────────────────────────────────────────────────────

def extract_ports_with_widths_from_spec(text: str) -> dict[str, int]:
    """
    Extract likely signal/port names and their bit-widths from spec text only.

    Returns:
        {
            "CLK_I": 1,
            "RST_I": 1,
            "ADDR_I": 32,
            "DATA_I": 32,
            "DATA_O": 32,
            "SEL_I": 4,
            "MTxD": 4,
            "MRxD": 4,
            "MDIO": 1,
            ...
        }
    """

    ports: dict[str, int] = {}

    def norm_sig(tok: str) -> str:
        tok = tok.strip().strip(",.;:()")
        tok = re.sub(r"\[\s*\d+\s*:\s*\d+\s*\]$", "", tok)   
        tok = re.sub(r"\[\s*\d+\s*\]$", "", tok)           
        return tok

    def width_from_token(tok: str) -> int | None:
        m = re.search(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]$", tok)
        if m:
            msb = int(m.group(1))
            lsb = int(m.group(2))
            return abs(msb - lsb) + 1
        m = re.search(r"\[\s*(\d+)\s*\]$", tok)
        if m:
            return 1
        return None

    def good_sig(tok: str) -> bool:
        base = norm_sig(tok)
        if len(base) < 2:
            return False
        lo = base.lower()
        if lo in _NON_PORT or lo in _ENGLISH_STOPWORDS:
            return False

        if "_" in base:
            return True
        if re.search(r"[a-z][A-Z]|[A-Z][a-z]", base):   
            return True
        if re.fullmatch(r"[A-Z0-9_]{2,}", base):        
            return True

        return False

    table_row_re = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9_\[\]:]{0,60})\s+"
        r"(\d+|\*)\s+"
        r"(I/O|I|O|Input|Output|Inout|In|Out)\b",
        re.MULTILINE,
    )

    for m in table_row_re.finditer(text):
        raw_name = m.group(1).strip()
        raw_width = m.group(2).strip()

        if not good_sig(raw_name):
            continue

        name = norm_sig(raw_name)

        # width from explicit width column has highest priority
        width = None
        if raw_width.isdigit():
            width = int(raw_width)
        else:
            width = width_from_token(raw_name)

        if width is None:
            width = 1

        ports[name] = width

    bus_token_re = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*\[\d+\s*:\s*\d+\])\b")
    for raw_name in bus_token_re.findall(text):
        if not good_sig(raw_name):
            continue
        name = norm_sig(raw_name)
        width = width_from_token(raw_name)
        if width is not None and name not in ports:
            ports[name] = width

    token_re = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{1,50}\b")
    counts = Counter()

    for raw_name in token_re.findall(text):
        if not good_sig(raw_name):
            continue
        name = norm_sig(raw_name)
        counts[name] += 1

    for name, cnt in counts.items():
        if cnt >= 2 and name not in ports:
            ports[name] = 1

    return dict(sorted(ports.items()))

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF chunk ranker
# ─────────────────────────────────────────────────────────────────────────────

_HW_STRONG_KW = {
    "shall", "must", "posedge", "negedge", "assert", "deassert",
    "handshake", "latch", "flip-flop",
    "synchronous", "asynchronous", "combinational", "sequential",
    "pulse", "sampled", "stable", "transition", "edge",
    "wishbone", "avalon", "apb", "ahb", "axi",
    "reset", "interrupt",
}
_HW_WEAK_KW = {
    "when", "upon", "if", "cycle", "clock", "enable", "disable",
    "receive", "transmit", "state", "idle", "active", "inactive",
}

# to estimate whether a spec chunk is worth looking at
def _keyword_relevance_score(chunk: str, known_ports: set) -> float:
    lower = chunk.lower()
    score  = sum(1.0 for kw in _HW_STRONG_KW if kw in lower)
    score += sum(0.4 for kw in _HW_WEAK_KW   if kw in lower)
    score += sum(2.0 for p in known_ports if p in lower)
    return score

# selects the most likely behavior-rich chunks before calling the LLM, reducing cost and noise
def rank_chunks_tfidf(
    chunks: list[str],
    known_ports: list[str],
    top_k: int = TFIDF_TOP_K,
    sim_threshold: float = TFIDF_SIM_THRESHOLD,
) -> list[tuple[int, float, str]]:
    if not HAS_SKLEARN or not chunks:
        # Fallback: return all chunks with keyword score, capped at top_k
        scored = [
            (i, _keyword_relevance_score(c, set(known_ports)), c)
            for i, c in enumerate(chunks)
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # Build query document
    hw_kw_str = " ".join(_HW_STRONG_KW | _HW_WEAK_KW)
    port_str  = " ".join(known_ports)
    # Repeat ports 3x to upweight them in TF-IDF space
    query_doc = f"{port_str} {port_str} {port_str} {hw_kw_str}"

    corpus = [query_doc] + chunks
    try:
        vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[a-zA-Z_][a-zA-Z0-9_]+",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf_matrix = vec.fit_transform(corpus)
        query_vec    = tfidf_matrix[0]
        chunk_vecs   = tfidf_matrix[1:]
        sims         = cosine_similarity(query_vec, chunk_vecs).flatten()
    except Exception:
        # sklearn failure → fall back to keyword scoring
        scored = [
            (i, _keyword_relevance_score(c, set(known_ports)), c)
            for i, c in enumerate(chunks)
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # Combine TF-IDF similarity with keyword score for final ranking
    # Hybrid: 0.6 × tfidf_sim (normalised to [0,1]) + 0.4 × kw_norm
    kw_scores  = [_keyword_relevance_score(c, set(known_ports)) for c in chunks]
    max_kw     = max(kw_scores) if max(kw_scores) > 0 else 1.0
    ranked = []
    for i, (sim, kw) in enumerate(zip(sims, kw_scores)):
        hybrid = 0.6 * float(sim) + 0.4 * (kw / max_kw)
        if float(sim) >= sim_threshold or kw >= 2.0:
            ranked.append((i, round(hybrid, 4), chunks[i]))

    ranked.sort(key=lambda x: -x[1])
    return ranked[:top_k]

# a quick keyword-only relevance gate
def is_relevant_chunk(chunk: str, known_ports: set, threshold: float = 2.0) -> bool:
    """Keyword-only gate — used as fast pre-filter before TF-IDF ranking."""
    return _keyword_relevance_score(chunk, known_ports) >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# Chunking with overlap  
# ─────────────────────────────────────────────────────────────────────────────

# breaks the spec into overlapping chunks, preferably by paragraph
def split_into_chunks(text: str, max_chars: int = 3000,
                      overlap_paras: int = 1) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)
                  if len(p.strip()) > 30]

    if len(paragraphs) >= 5:
        chunks: list[str] = []
        buf  = ""
        tail: list[str] = []
        for p in paragraphs:
            candidate = ("\n\n".join(tail) + "\n\n" + p).strip() if tail else p
            if len(buf) + len(candidate) + 2 <= max_chars:
                buf += "\n\n" + candidate if buf else candidate
            else:
                if buf.strip():
                    chunks.append(buf.strip())
                tail_paras = buf.strip().split("\n\n")
                tail = tail_paras[-overlap_paras:] if overlap_paras else []
                buf  = candidate
        if buf.strip():
            chunks.append(buf.strip())
        return chunks

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf, overlap = [], "", []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(buf) + len(sent) + 1 <= max_chars:
            buf += " " + sent
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = " ".join(overlap[-2:]) + " " + sent
        if len(sent) > 20:
            overlap.append(sent)
    if buf.strip():
        chunks.append(buf.strip())

    if not chunks:
        step = max_chars - 200
        for i in range(0, len(text), step):
            c = text[i: i + max_chars].strip()
            if c:
                chunks.append(c)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Semantic dedup  
# ─────────────────────────────────────────────────────────────────────────────

# simplifies a rule into a normalized trigger+obligation text form
def _normalise_rule_text(r: dict) -> str:
    trig = re.sub(r"\b(on|when|after|upon|if)\b", "",
                  r["trigger"]["expression"], flags=re.IGNORECASE)
    obli = re.sub(r"\b(is|are|will|shall|must|be)\b", "",
                  r["obligation"]["expression"], flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", (trig + " " + obli).lower().strip())


def _trigrams(s: str) -> set:
    tokens = re.findall(r"\w+", s)
    return {" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

# checks whether a newly extracted rule is basically the same as one already accepted
def _is_duplicate(new_rule: dict, existing_rules: list,
                  threshold: float = 0.60) -> bool:
    new_text = _normalise_rule_text(new_rule)
    for r in existing_rules:
        if _jaccard(new_text, _normalise_rule_text(r)) >= threshold:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Rule quality gate  
# ─────────────────────────────────────────────────────────────────────────────

_VAGUE_TRIGGER_RE = re.compile(
    r"^(on\s+reset$|when\s+enable\s+is\s+(high|low)$|"
    r"on\s+(rising|falling)\s+edge\s+of\s+clk$|"
    r"on\s+bit\s+level|on\s+timed|on\s+overdrive|"
    r"power\s+delivery$|clock\s+divider$|base\s+time\s+period$|"
    r"hardware\s+\d|software\s+\d)",
    re.IGNORECASE,
)
_VAGUE_OBLIGATION_RE = re.compile(
    r"^(reset\s+(behavior|all|internal)|update\s+internal\s+state|"
    r"allow\s+1-wire|idle\s+state\s+is\s+maintained|"
    r"data\s+(is\s+transferred|transfer\s+rate)|"
    r"deliver\s+power|update\s+(clock|base)|enable\s+overdrive|"
    r"wait\s+for\s+\d+ms$)",
    re.IGNORECASE,
)

# rejects vague, circular, weakly grounded, or too-short/underspecified rules
def _is_low_quality_rule(rule: dict) -> bool:
    trig = rule["trigger"]["expression"].strip()
    obli = rule["obligation"]["expression"].strip()
    if _VAGUE_TRIGGER_RE.match(trig) and _VAGUE_OBLIGATION_RE.match(obli):
        return True
    if _jaccard(trig.lower(), obli.lower()) > 0.70:
        return True
    if not rule.get("rtl_signals") and rule.get("confidence", 1.0) < 0.65:
        return True
    orig = rule.get("original_text", "")
    if len(orig) < 35 and not re.search(
        r"\b(is|are|will|shall|must|when|if|after|set|cleared|"
        r"written|read|asserted|deasserted|driven|sampled)\b",
        orig, re.IGNORECASE
    ):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic state-name detection  
# ─────────────────────────────────────────────────────────────────────────────

_STATE_RE_STATIC = re.compile(
    r"\b(IDLE|TRANSMIT|RECEIVE|RESET|ACTIVE|WAIT|BUSY|DATA|"
    r"PREAMBLE|SFD|CRC|DONE|HALT|FETCH|DECODE|EXECUTE|RUN|STOP|"
    r"INIT|READY|START|LOAD|STORE|PAUSE|ERROR|FLUSH|ACK)\b"
)

_ALLCAPS_STOPWORDS = {
    "I", "II", "III", "IV", "VI", "VII", "VIII", "IX",
    "OK", "DC", "AC", "IC", "PC", "IO", "ID", "IP",
    "OR", "AND", "NOT", "XOR", "NOR", "NAND",
    "HIGH", "LOW", "TRUE", "FALSE", "NULL", "NOP", "EOF", "SOF",
    "NOTE", "TABLE", "FIGURE", "SECTION", "REG", "BIT",
    "LSB", "MSB", "GPIO", "UART", "SPI", "I2C", "AHB", "APB",
    "CPU", "DMA", "ROM", "RAM", "FIFO",
    "MHz", "kHz", "GHz", "Mbps", "ns", "us", "ms",
    "RTL", "HDL", "SoC", "FPGA", "ASIC", "CPLD", "BSP", "HAL",
    "SOPC", "TCL", "EDS", "JTAG", "OS", "REC", "LGPL", "TODO",
    "GUI", "PS", "BAW", "BDW", "OWN", "OVD", "CDR", "BTP",
}

# detects likely FSM state names from repeated all-caps tokens in the spec
def _dynamic_state_names(text: str, min_occurrences: int = 2) -> list[str]:
    candidates = re.findall(r"\b([A-Z][A-Z0-9_]{2,30})\b", text)
    counts = Counter(candidates)
    return [
        tok for tok, cnt in counts.items()
        if cnt >= min_occurrences
        and tok not in _ALLCAPS_STOPWORDS
        and "_" not in tok
        and len(tok) <= 10
    ]

# ─────────────────────────────────────────────────────────────────────────────
# 1.2  LLM rule extraction — grounded prompt  
# ─────────────────────────────────────────────────────────────────────────────
# sends one spec chunk to the LLM and asks it to return structured behavioral rules in JSON
_EXTRACTION_PROMPT = """\
You are an expert hardware verification engineer.

Your task is to extract ONLY explicit RTL-relevant behavioral rules from the given hardware specification text.

KNOWN PORT / SIGNAL NAMES:
{port_list}

A good behavioral rule must describe:
- a concrete trigger condition
- a concrete obligation/result
- optional guards / preconditions
- optional timing if the spec explicitly states timing

Allowed rule categories:
- reset behavior
- read/write behavior
- enable/disable behavior
- handshake / valid-ready / busy-ack behavior
- interrupt / status bit set-clear behavior
- mode-dependent behavior
- state-dependent behavior
- error / exception behavior
- automatic clear / automatic set behavior
- protocol timing explicitly stated in the text

Do NOT extract:
- headings, captions, section titles, register names by themselves
- pure structural descriptions ("module consists of...", "contains...", "is connected to...")
- vague summaries without a signal-level effect
- inferred behavior not explicitly stated in the text
- obligations that do not mention at least one KNOWN PORT / SIGNAL
- rules where trigger and obligation are semantically the same statement

Important grounding rules:
1. Prefer exact signal names from the KNOWN PORT / SIGNAL list.
2. Only use timing if the text explicitly states it.
3. `original_text` must be copied verbatim from the spec text.
4. If a sentence is descriptive but does not specify a behavioral consequence, do not emit a rule.
5. If a chunk contains no explicit behavioral rules, return [].

SPEC TEXT:
{text}

Return ONLY a raw JSON array:
[
  {{
    "id": "R001",
    "trigger": "<explicit condition>",
    "obligation": "<explicit effect>",
    "timing_type": "immediate|next_cycle|within_cycles|after_cycles|eventually|unspecified",
    "timing_value": <integer or null>,
    "guards": ["<explicit guard condition>"],
    "original_text": "<verbatim sentence from SPEC TEXT>",
    "confidence": <0.0-1.0>,
    "rtl_signals": ["<signal names from KNOWN PORT / SIGNAL list>"]
  }}
]
"""

def _call_llm_extract(chunk: str, id_offset: int, chunk_idx: int,
                      known_ports: list[str]) -> list:
    port_list = ", ".join(known_ports) if known_ports else "(none identified yet)"
    prompt    = _EXTRACTION_PROMPT.format(
        text=chunk[:3500], port_list=port_list
    )

    for attempt in range(4):
        try:
            resp = _client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert hardware verification engineer. "
                            "Respond with valid JSON only. No markdown, no commentary."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.05, "num_predict": 2048},
            )
            raw = resp['message']['content'].strip()
            raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"^```\s*",     "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```$",        "", raw.strip())

            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                return []
            rules_raw = json.loads(m.group(0))
            if not isinstance(rules_raw, list):
                rules_raw = [rules_raw]

            known_lower = {p.lower() for p in known_ports}
            rules = []

            for r in rules_raw:
                if not isinstance(r, dict):
                    continue
                trigger    = str(r.get("trigger", "")).strip()
                obligation = str(r.get("obligation", "")).strip()
                if not trigger and not obligation:
                    continue

                timing_type  = r.get("timing_type", "unspecified")
                timing_value = r.get("timing_value")
                if timing_value is not None:
                    try:
                        timing_value = int(timing_value)
                    except (ValueError, TypeError):
                        timing_value = None

                raw_signals: list = r.get("rtl_signals", [])
                if known_ports:
                    validated_signals = [
                        s.strip() for s in raw_signals
                        if isinstance(s, str) and s.strip().lower() in known_lower
                    ]
                else:
                    validated_signals = [
                        s.strip() for s in raw_signals
                        if isinstance(s, str) and _is_rtl_signal_token(s.strip())
                    ]

                rule_num  = id_offset + len(rules) + 1
                ambiguity = []
                if timing_type in ("eventually", "unspecified", "soon"):
                    ambiguity.append("TIMING_AMBIGUOUS")

                rules.append({
                    "id": f"R{rule_num:03d}",
                    "trigger":    {"expression": trigger,    "type": "llm_extracted"},
                    "obligation": {"expression": obligation, "type": "llm_extracted"},
                    "timing": {
                        "type":  timing_type,
                        "value": timing_value,
                        "confidence": (
                            "HIGH"   if timing_value is not None else
                            "MEDIUM" if timing_type not in ("unspecified", "eventually") else
                            "LOW"
                        ),
                    },
                    "guards":          r.get("guards", []),
                    "original_text":   str(r.get("original_text", ""))[:400],
                    "ambiguity_flags": ambiguity,
                    "confidence":      float(r.get("confidence", 0.75)),
                    "rtl_signals":     validated_signals,
                    "provenance":      {"chunk_idx": chunk_idx},
                })
            return rules

        except json.JSONDecodeError as e:
            print(f"    [chunk {chunk_idx} attempt {attempt+1}] JSON error: {e}")
            time.sleep(2)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 30
                m2 = re.search(r"retry.after.(\d+)", err, re.IGNORECASE)
                if m2:
                    wait = int(m2.group(1)) + 5
                print(f"    [rate limit] waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"    [chunk {chunk_idx} error] {err[:200]}")
                return []

    return []


# ─────────────────────────────────────────────────────────────────────────────
# 1.3  Main entry
# ─────────────────────────────────────────────────────────────────────────────
# main Phase 1 driver
def process_spec(spec_file: str, rtl_ir: dict | None = None) -> dict:
    print(f"[Phase 1] Processing spec: {spec_file}")
    raw_text = extract_text(spec_file)
    clean_text = _strip_boilerplate(raw_text)
    print(f"  Text: {len(raw_text):,} → {len(clean_text):,} chars after boilerplate strip")

    spec_ports = extract_port_names_from_spec(clean_text)
    spec_port_widths = extract_ports_with_widths_from_spec(clean_text)
    print(f"  Pass 1 widths sample: {list(spec_port_widths.items())[:25]}")
    print(f"  Pass 1 — {len(spec_ports)} port candidates: {spec_ports[:25]}")

    rtl_vocab: set[str] = set()
    if rtl_ir:
        rtl_vocab  = set(rtl_ir.get("all_signals", []))
        rtl_lower  = {s.lower(): s for s in rtl_vocab}
        validated  = []
        for p in spec_ports:
            if p.lower() in rtl_lower:
                validated.append(rtl_lower[p.lower()])
        text_lower = clean_text.lower()
        for rtl_sig_lo, rtl_sig in rtl_lower.items():
            if len(rtl_sig_lo) >= 3 and rtl_sig_lo in text_lower:
                validated.append(rtl_sig)
        known_ports = sorted(set(validated))
        print(f"  RTL cross-check: {len(known_ports)} ports survive")
    else:
        known_ports = spec_ports
        print(f"  No RTL IR — using {len(known_ports)} spec-extracted ports")

    known_ports_set = set(known_ports)
    chunks = split_into_chunks(clean_text, max_chars=3000, overlap_paras=1)
    print(f"  Split into {len(chunks)} chunks")

    # ── TF-IDF ranking replaces flat keyword filter ──────────────
    ranked_chunks = rank_chunks_tfidf(chunks, known_ports, top_k=TFIDF_TOP_K,
                                      sim_threshold=TFIDF_SIM_THRESHOLD)
    skipped_by_tfidf = len(chunks) - len(ranked_chunks)
    print(f" TF-IDF ranking: {len(ranked_chunks)}/{len(chunks)} chunks "
          f"selected (skipped {skipped_by_tfidf} below threshold "
          f"sim={TFIDF_SIM_THRESHOLD})")

    all_rules:        list     = []
    all_rule_signals: set[str] = set()
    skipped_kw = 0

    for rank_pos, (orig_idx, tfidf_score, chunk) in enumerate(ranked_chunks):
        # Secondary keyword gate (fast path) — still applied per chunk
        if not is_relevant_chunk(chunk, known_ports_set, threshold=0.5):
            skipped_kw += 1
            continue

        print(f"  Chunk {orig_idx+1} (rank {rank_pos+1}, tfidf={tfidf_score:.3f}, "
              f"{len(chunk)} chars) …", end=" ", flush=True)

        rules = _call_llm_extract(chunk, id_offset=len(all_rules),
                                  chunk_idx=orig_idx, known_ports=known_ports)
        print(f"{len(rules)} raw", end="")

        accepted = 0
        for rule in rules:
            if _is_duplicate(rule, all_rules):
                continue
            if _is_low_quality_rule(rule):
                continue
            all_rules.append(rule)
            for sig in rule.get("rtl_signals", []):
                all_rule_signals.add(sig)
            accepted += 1
        print(f" → {accepted} accepted")

        if rank_pos < len(ranked_chunks) - 1:
            time.sleep(2)

    print(f"  Chunks skipped (TF-IDF): {skipped_by_tfidf}  "
          f"(keyword gate): {skipped_kw}  total saved: {skipped_by_tfidf + skipped_kw}")

    for idx, rule in enumerate(all_rules, start=1):
        rule["id"] = f"R{idx:03d}"

    text_lower   = clean_text.lower()
    all_signals: set[str] = set(all_rule_signals)
    for p in known_ports:
        if p.lower() in text_lower:
            all_signals.add(p)

    static_states  = set(_STATE_RE_STATIC.findall(raw_text))
    dynamic_states = set(_dynamic_state_names(raw_text, min_occurrences=2))
    state_mentions = sorted(static_states | dynamic_states)

    spec_ir = {
        "rules":             all_rules,
        "mentioned_signals": sorted(all_signals)[:300],
        "state_mentions":    state_mentions,
        "source":            spec_file,
        "known_ports":       known_ports,
        "port_widths": spec_port_widths,
        "chunk_stats": {
            "total_chunks":        len(chunks),
            "tfidf_selected":      len(ranked_chunks),
            "skipped_tfidf":       skipped_by_tfidf,
            "skipped_keyword_gate": skipped_kw,
            "llm_calls_made":      len(ranked_chunks) - skipped_kw,
        },
    }
    print(
        f"\n  ─── Summary ─────────────────────────────────────\n"
        f"  Rules          : {len(all_rules)}\n"
        f"  Unique signals : {len(all_signals)}\n"
        f"  State names    : {len(state_mentions)}\n"
        f"  Known ports    : {len(known_ports)}\n"
        f"  LLM calls saved: {skipped_by_tfidf + skipped_kw} "
        f"(of {len(chunks)} total chunks)"
    )
    return spec_ir


# ─────────────────────────────────────────────────────────────────────────────
# Intermediate output dump  
# ─────────────────────────────────────────────────────────────────────────────

def dump_spec_ir(spec_ir: dict, out_dir: str = "cache") -> dict[str, Path]:
    base = Path(out_dir) / "spec_ir"
    base.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    full_path = base / "spec_ir_full.json"
    full_path.write_text(json.dumps(spec_ir, indent=2, default=str), encoding="utf-8")
    written["full_json"] = full_path

    rules_path = base / "rules_summary.txt"
    lines = [
        "Spec IR — Rules Summary",
        f"Source : {spec_ir['source']}",
        f"Total  : {len(spec_ir['rules'])} rules",
        "=" * 70, "",
    ]
    for r in spec_ir["rules"]:
        timing = r["timing"]
        flags  = r.get("ambiguity_flags", [])
        sigs   = r.get("rtl_signals", [])
        lines += [
            f"┌─ {r['id']}  (confidence: {r['confidence']:.2f})"
            + (f"  ⚠ {flags}" if flags else ""),
            f"│  Trigger    : {r['trigger']['expression']}",
            f"│  Obligation : {r['obligation']['expression']}",
            f"│  Timing     : {timing['type']}"
            + (f" = {timing['value']} cycles" if timing.get('value') else "")
            + f"  [{timing['confidence']}]",
            f"│  Guards     : {r.get('guards', [])}",
            f"│  RTL signals: {sigs if sigs else '(none — review)'}",
            f"│  Chunk      : {r['provenance']['chunk_idx']}",
            f"└─ Original   : {r['original_text'][:120]}",
            "",
        ]
    rules_path.write_text("\n".join(lines), encoding="utf-8")
    written["rules_summary"] = rules_path

    # Chunk stats summary
    stats = spec_ir.get("chunk_stats", {})
    if stats:
        stats_path = base / "chunk_stats.txt"
        stats_path.write_text(
            "Chunk Selection Stats (TF-IDF)\n" + "=" * 50 + "\n" +
            "\n".join(f"  {k}: {v}" for k, v in stats.items()),
            encoding="utf-8"
        )
        written["chunk_stats"] = stats_path

    sig_path = base / "signals.txt"
    sig_path.write_text("\n".join(
        [f"# Validated RTL signals  ({len(spec_ir['mentioned_signals'])} total)", ""]
        + spec_ir["mentioned_signals"]
    ), encoding="utf-8")
    written["signals"] = sig_path

    state_path = base / "state_names.txt"
    state_path.write_text("\n".join(
        [f"# FSM state names  ({len(spec_ir['state_mentions'])} total)", ""]
        + spec_ir["state_mentions"]
    ), encoding="utf-8")
    written["state_names"] = state_path

    conf_path = base / "rules_by_confidence.txt"
    sorted_rules = sorted(spec_ir["rules"], key=lambda r: r["confidence"])
    conf_lines = ["Rules sorted by confidence (lowest first)", "=" * 70, ""]
    for r in sorted_rules:
        sigs = r.get("rtl_signals", [])
        conf_lines += [
            f"[{r['confidence']:.2f}]  {r['id']}",
            f"  T: {r['trigger']['expression'][:100]}",
            f"  O: {r['obligation']['expression'][:100]}",
            f"  Timing : {r['timing']['type']}  Signals: {sigs}",
            f"  Flags  : {r.get('ambiguity_flags', [])}",
            f"  Origin : {r.get('original_text', '')[:80]}",
            "",
        ]
    conf_path.write_text("\n".join(conf_lines), encoding="utf-8")
    written["rules_by_confidence"] = conf_path

    ports_path = base / "known_ports.txt"
    kp = spec_ir.get("known_ports", [])
    ports_path.write_text("\n".join(
        [f"# Port vocabulary ({len(kp)} ports)", ""] + kp
    ), encoding="utf-8")
    written["known_ports"] = ports_path

    widths_path = base / "port_widths.txt"
    pw = spec_ir.get("port_widths", {})
    widths_path.write_text("\n".join(
        [f"# Port widths ({len(pw)} entries)", ""] +
        [f"{k}: {v}" for k, v in pw.items()]
    ), encoding="utf-8")
    written["port_widths"] = widths_path
    
    print(f"\n[dump] Spec IR written to: {base}/")
    for key, p in written.items():
        print(f"  {key:25s} → {p.name}")
    return written

# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    SPEC_PATH = "specs/eth_speci.pdf"   
    OUTPUT_PATH = "output_rules.json"

    print("Running rule extraction...")

    try:
        rules = process_spec(SPEC_PATH)

        # Save output
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2)

        print(f"Done! Extracted {len(rules)} rules")
        print(f"Saved to {OUTPUT_PATH}")

    except Exception as e:
        print("Error occurred:")
        print(e)