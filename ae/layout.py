"""Row layouts of the paper's tables (names and order as printed in the paper).

These are the source of truth for ``fig``: every table is rendered with exactly
these rows, in this order, so a reproduced table lines up with the paper even
though the raw result files use the benchmark registry's own names.
"""

from __future__ import annotations

# ── Table 1: recurrences in the wild ────────────────────────────────────────
# (paper row label, specs.registry name)
WILD_ROWS: list[tuple[str, str]] = [
    ("First-Order High Pass Filter", "first-order-high-pass"),
    ("Second-Order High Pass Filter", "second-order-high-pass"),
    ("Second-Order Band-Pass Filter", "second-order-low-pass"),
    ("Box Filter", "box-filter-sat"),
    ("Repeated Box Filter", "repeated-box-filter"),
    ("Recursive Gaussian Filter", "recursive-gaussian"),
    ("Non-uniform Recursive Filter", "nonuniform-recursive-filter"),
    ("Distance Transform", "distance-transform"),
    ("Exponential Smoothing", "exponential-smoothing"),
    ("Double Exponential Smoothing", "double-exp-smoothing"),
    ("Welford Variance", "welford-variance"),
    ("Continued Fraction", "continued-fraction"),
]

# ── Table 2: LeetCode recurrences ───────────────────────────────────────────
LEETCODE_ROWS: list[tuple[str, str]] = [
    ("10.~Regular Expression Matching", "leetcode-10-regular-expression-matching"),
    ("91.~Decode Ways", "leetcode-91-decode-ways"),
    ("121.~Buy/Sell Stock", "leetcode-121-best-time-stock"),
    ("188.~Buy/Sell Stock IV", "leetcode-188-best-time-stock-iv"),
    ("198.~House Robber", "leetcode-198-house-robber"),
    ("221.~Maximal Square", "leetcode-221-maximal-square"),
    ("304.~Range Sum Query 2D", "leetcode-304-range-sum-2d"),
    ("309.~Buy/Sell Stock with Cooldown", "leetcode-309-best-time-stock-cooldown"),
    ("746.~Min Cost Climbing Stairs", "leetcode-746-min-cost-climbing-stairs"),
    ("1143.~Longest Common Subsequence", "leetcode-1143-lcs"),
    ("1458.~Max Dot Product of Two Subsequences", "leetcode-1458-max-dot-product"),
    ("3259.~Max Energy Boost From Two Drinks", "leetcode-3259-max-energy-two-drinks"),
]

MICRO_GROUPS: dict[str, list[tuple[str, str]]] = {
    "wild": WILD_ROWS,
    "leetcode": LEETCODE_ROWS,
}

# ── Table 3: bioinformatics kernels vs libraries (2048 x 2048) ──────────────
# Each benchmark group: paper name, harness case name, the baseline rows
# (paper label, case the row comes from, impl name in that case's results),
# and the WFA2 85%-identity row (dagger: reports WFA2's speedup over VecRec).
# The VecRec row is the fastest *correct* int16/int32 schedule of the main case.
BIO_SIZE = 2048


class BioGroup:
    def __init__(self, name: str, case: str, baselines: list[tuple[str, str, str]],
                 wfa85: tuple[str, str] | None = None) -> None:
        self.name = name
        self.case = case
        self.baselines = baselines
        self.wfa85 = wfa85


BIO_GROUPS: list[BioGroup] = [
    BioGroup("Edit Distance", "edit-distance", [
        ("edlib", "edit-distance", "edlib"),
    ]),
    BioGroup("Needleman--Wunsch", "needleman-wunsch", [
        ("parasail\\_scan", "needleman-wunsch", "parasail_nw_scan_16"),
        ("WFA2 (65\\%)", "needleman-wunsch-sim65", "wfa2_nw"),
        ("parasail\\_striped", "needleman-wunsch", "parasail_nw_striped_16"),
        ("parasail\\_diag", "needleman-wunsch", "parasail_nw_diag_16"),
        ("ksw2\\_gg2\\_sse", "needleman-wunsch", "ksw2_gg2_sse"),
        ("ksw2\\_extf2\\_sse", "needleman-wunsch", "ksw2_extf2_sse"),
        ("seqan3", "needleman-wunsch", "seqan3_nw"),
    ], wfa85=("needleman-wunsch-sim85", "wfa2_nw")),
    BioGroup("Smith--Waterman", "smith-waterman", [
        ("parasail\\_striped", "smith-waterman", "parasail_sw_striped_16"),
        ("parasail\\_scan", "smith-waterman", "parasail_sw_scan_16"),
        ("parasail\\_diag", "smith-waterman", "parasail_sw_diag_16"),
        ("seqan3", "smith-waterman", "seqan3_sw"),
    ]),
    BioGroup("Affine Gap", "affine-gap", [
        ("parasail\\_scan", "affine-gap", "parasail_affine_scan_16"),
        ("parasail\\_striped", "affine-gap", "parasail_affine_striped_16"),
        ("WFA2 (65\\%)", "affine-gap-sim65", "wfa2_affine"),
        ("parasail\\_diag", "affine-gap", "parasail_affine_diag_16"),
        ("ksw2\\_extd2\\_sse", "affine-gap", "ksw2_extd2_sse"),
        ("ksw2\\_extz2\\_sse", "affine-gap", "ksw2_extz2_sse"),
        ("ksw2\\_gg2\\_sse", "affine-gap", "ksw2_gg2_sse"),
    ], wfa85=("affine-gap-sim85", "wfa2_affine")),
]

# VecRec element widths the paper picks the best of ("int16 and int32 variants").
BIO_VECREC_WIDTHS = ("_i16_", "_i32_")

# ── Table 4: minimap2 end-to-end on ONT workloads ───────────────────────────
# (paper dataset label, scenario name in end_to_end_baselines/minimap2/scenarios.py)
E2E_ROWS: list[tuple[str, str]] = [
    ("ecoli-ultralong", "ablation-noapprox-ecoli-ultralong"),
    ("ecoli-strains", "ablation-noapprox-ecoli-strains-mapont"),
    ("pa-strains", "ablation-noapprox-pa-strains"),
]

# ── Table 5: Reptile vs VecRec on Smith--Waterman ───────────────────────────
REPTILE_KIND = "sw"
REPTILE_SIZES = (1024, 2048, 4096, 8192)
REPTILE_TILES = (4, 8, 16, 32, 64, 128, 256, 512)
