import os


# ── Model Selection (Ollama) ───────────────────────────────────────────────────
# Set the host for your local Ollama instance.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Set the model to use with Ollama.
# llama3.1:8b is a good starting point. For higher quality, consider llama3.1:70b
# if your hardware supports it.
OLLAMA_MODEL = "llama3.1:latest"

# ── Grounding Thresholds ──────────────────────────────────────────────────────
GROUNDING_HIGH      = 0.85   # auto-accept, no disambiguation needed
GROUNDING_MED       = 0.60   # accept with flag for review
FUZZY_THRESHOLD     = 70     # rapidfuzz score (0-100) to consider a candidate

# ── Slicing ───────────────────────────────────────────────────────────────────
MAX_SLICE_DEPTH     = 5      # BFS depth for dependency traversal (fallback only)
MAX_FWD_DEPTH       = 3      # Forward slice depth (fallback only)
MAX_ITERATIONS      = 3      # Sufficiency refinement iterations

# ── Budget-Aware Slicer ──────────────────────────────────────────
# Priority-queue slicer 
# Replaces fixed BFS depth with a ranked signal selection that stops when the
# assembled snippet budget is exhausted.  Relevance score per signal is:
#   score = grounding_confidence × (1 / (hop_distance + 1))
# Signals are added highest-score-first until TOKEN_BUDGET is reached.
SLICE_TOKEN_BUDGET      = 350    # max tokens of RTL snippet text to assemble
SLICE_MIN_SIGNALS       = 5      # always include at least this many signals
SLICE_SCORE_FLOOR       = 0.05   # prune signals below this relevance score

# ── TF-IDF Chunk Relevance ───────────────────────────────────────
# Chunk scoring now uses TF-IDF cosine similarity between each chunk and a
# query built from known ports + rule keywords 
# The keyword scoring path is kept as a fast pre-filter; TF-IDF is used when
# the keyword score alone is uncertain (between threshold/2 and threshold).
TFIDF_TOP_K             = 8      # keep at most this many chunks per spec pass
TFIDF_SIM_THRESHOLD     = 0.07   # cosine similarity floor for a chunk to pass

# ── Sufficiency Weights ───────────────────────────────────────────────────────
L1_WEIGHT           = 0.40
L2_WEIGHT           = 0.40
L3_WEIGHT           = 0.20
OVERALL_THRESHOLD   = 0.70

# ── SVA Generation ────────────────────────────────────────────────────────────
DEFAULT_TIMING_WINDOW = "[1:10]"
MAX_SNIPPET_CHARS   = 1500

# ── Ranked Snippet Compression ───────────────────────────────────
# Before hard-truncating to MAX_SNIPPET_CHARS, lines are ranked by signal
# relevance (does this line contain a trigger/obligation signal?) and
# context lines (±SNIPPET_CONTEXT_LINES around each hit) are kept first.
# Lossless-safe: only the ordering changes, not the content
SNIPPET_CONTEXT_LINES   = 3      # lines of context to keep around each hit line
SNIPPET_RELEVANCE_BONUS = 2.0    # score multiplier for lines containing core signals

# ── Active Staleness Pruning ─────────────────────────────────────
# Stale grounding entries (Weibull-decayed below floor or signal absent from
# RTL) are filtered OUT during Phase 3 retrieval rather than merely flagged.
# After STALE_PRUNE_AFTER_RUNS pipeline runs a hard-prune removes them from
# disk entirely
STALE_CONFIDENCE_FLOOR  = 0.30   # entries below this are excluded from grounding
STALE_PRUNE_AFTER_RUNS  = 5      # prune after this many pipeline runs

# ── Tier-2.5 Port-Direction Check ────────────────────────────────
# After Tier-2 semantic lint, validate that the generated assertion's
# obligation signals are consistent with their RTL port directions.
# Asserting that an `input` port goes high is architecturally unsound
TIER25_ENABLED          = True   # set False to skip the direction check
TIER25_WARN_ON_INOUT    = True   # treat inout as a warning (not a hard fail)

# ── Cache and Results ─────────────────────────────────────────────────────────────────────
CACHE_DIR = "cache"
RESULTS_DIR = "results"

# ── Design Registry ──────────────────────────────────────────────────────────
DESIGNS = {
    "ethmac": {
        "rtl_dir":    "designs/ethmac/rtl",
        "spec_file":  "specs/eth_speci.pdf",
        "top_module": "eth_top",
    },
    "sockit": {
        "rtl_dir":    "designs/sockit",
        "spec_file":  "specs/sockit.pdf",
        "top_module": "sockit",
    },
    "openMSP430": {
        "rtl_dir":    "designs/openMSP430",
        "spec_file":  "specs/openMSP430.pdf",
        "top_module": "openMSP430",
    },
    "spi_master": {
        "rtl_dir":    "designs/spi_master/rtl",
        "spec_file":  "specs/spi_master.txt",
        "top_module": "spi_master",
    },
    "crypto_bridge": {
        "rtl_dir":    "designs/crypto_dma_bridge/rtl",
        "spec_file":  "specs/crypto_dma_bridge.txt",
        "top_module": "crypto_dma_bridge",
    },
    "i2c": {
        "rtl_dir":    "designs/i2c/rtl",
        "spec_file":  "specs/i2c.pdf",
        "top_module": "i2c_master_top",
    },
    "ethernet": {
        "rtl_dir":    "designs/ethernet/rtl",
        "spec_file":  "specs/ethernet.pdf",
        "top_module": "eth_top",
    },
}