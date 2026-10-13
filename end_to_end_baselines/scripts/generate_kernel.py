"""Generate the vecrec affine kernel header + minimap2 ksw2 replacement source.

Usage:
    uv run python end_to_end_baselines/scripts/generate_kernel.py [schedule_index]

Produces (in end_to_end_baselines/minimap2/generated/):
  vecrec_affine_kernel.h    — the vecrec kernel struct (arch-optimised SIMD),
                              post-patched so the affine gap penalties are
                              RUNTIME constructor parameters (not baked-in
                              literals). The vectorised inner loops are
                              untouched VecRec output.
  ksw2_vecrec_ext.cpp       — drop-in ksw_extz2_sse / ksw_extd2_sse replacements
                              with runtime gap + runtime band, full ksw_extz_t
                              outputs, and a real CIGAR traceback. ksw_exts2_sse
                              is deliberately NOT defined here so minimap2 keeps
                              ksw2's original splice kernel.

By default, run VecRec's schedule and codegen search for each affine/element-width
variant. A positional schedule_index or VECREC_SCHED_* environment variable
explicitly bypasses search for that variant.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import pathlib
import re
import sys
import tempfile

# Ensure repo root is on the path so we can import from bio_kernel_evals / evals_common.
_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from vecrec.codegen import (
    CodeGen, MAX_SWEEP_SCHEDULES, generate_and_run_benchmark,
    instantiate_kernels, sweep_codegen_setup,
)
from vecrec.optimize import get_candidate_schedules

# ── legacy element-width default for direct helper calls ──────────────────────
# The generated minimap2 wrapper always includes both I8 and I16 kernels.
# I8 is schedule-sensitive: VecRec's dilated F-gap "prefix-scan"
# tower holds `multiplier * gap_penalty` in the SAME element type as the score
# itself (a compile-time literal, or -- after patch_runtime_gaps -- a runtime
# `static_cast<elt_type>(mult * param)`). At I8 the multiplier grows with the
# schedule's dilation depth (roughly doubling per schedule index), so any
# schedule whose required multiplier times the largest real gap magnitude
# exceeds 127 will either fail to compile (literal narrowing) or silently
# wrap (runtime cast) -- NOT saturate. For minimap2 map-ont's default 2-piece
# gap (q2+e2=25), schedule indices >= 12 overflow (mult >= 8 -> 200 > 127);
# indices 0-11 are the search range for I8. A runtime guard below also checks
# the actual generated tap multipliers, which can grow with codegen's epi_mult.
ELEMENT_WIDTH = int(os.environ.get("VECREC_ELEMENT_WIDTH", "16"))
if ELEMENT_WIDTH not in (8, 16):
    raise ValueError(f"VECREC_ELEMENT_WIDTH must be 8 or 16, got {ELEMENT_WIDTH}")
ELT_CTYPE = "int8_t" if ELEMENT_WIDTH == 8 else "int16_t"

# ── runtime-gap sentinels ─────────────────────────────────────────────────────
# The expression is generated with these distinctive (prime) sentinel gap values
# so that every gap-derived literal in the generated header — including the
# dilation-scaled tap values (m * gap_extend) produced by the schedule — can be
# unambiguously detected and rewritten into runtime constructor parameters.
# They must be distinct primes that don't collide with structural constants, and
# small enough that m*sentinel stays within int16 for m up to epi.
GE_S      = -101   # gap_extend          (piece 1)
GO_GE_S   = -103   # gap_open+gap_extend (piece 1)
GE2_S     = -107   # gap_extend          (piece 2)
GO2_GE2_S = -109   # gap_open+gap_extend (piece 2)

# map sentinel magnitude -> runtime parameter name (kernel ctor argument)
_SENTINEL_TO_PARAM = {
    abs(GE_S): "ge",
    abs(GO_GE_S): "go_ge",
    abs(GE2_S): "ge2",
    abs(GO2_GE2_S): "go2_ge2",
}


# Build dual-affine (2-piece affine / Suzuki-Kasahara) expression matching the
# recurrence minimap2 runs (ksw_extd2_sse). The penalties are SENTINELS here;
# they are rewritten to runtime ctor params by patch_runtime_gaps() below.
#   gap(L) = min(q + L*e,  q2 + L*e2)
def _build_minimap2_dual_affine_expr(elt=None, gaps=None):
    from vecrec.expr import (
        Convolve, Impulse, Ith, Num, PointwiseMul,
        Recurse, RepeaterVec, RVec2D, SAdd, SignalVector, SignalVectorExpr,
        TIKernel, Type, Var2D, with_types,
    )
    from vecrec.util import ElementType

    GE, GO_GE, GE2, GO2_GE2 = gaps or (GE_S, GO_GE_S, GE2_S, GO2_GE2_S)

    if elt is None:
        elt = ElementType.I8 if ELEMENT_WIDTH == 8 else ElementType.I16
    with with_types(Type.TropMax, elt):
        s = Var2D("s")
        neginf = Type.TropMax.zero()
        one    = Type.TropMax.one()
        z      = TIKernel.z()

        def body(prev: RVec2D) -> SignalVectorExpr:
            prev_h  = prev.at(0).at(0)   # H(i-1, j)
            prev_e1 = prev.at(1).at(0)   # E1(i-1, j)
            prev_e2 = prev.at(2).at(0)   # E2(i-1, j)
            s_row   = Ith(s, 0)

            # E1(i,j) = max(H(i-1,j) + q+e,   E1(i-1,j) + e)
            E1 = SAdd(PointwiseMul(prev_h,  Num(GO_GE)),   PointwiseMul(prev_e1, Num(GE)))
            # E2(i,j) = max(H(i-1,j) + q2+e2, E2(i-1,j) + e2)
            E2 = SAdd(PointwiseMul(prev_h,  Num(GO2_GE2)), PointwiseMul(prev_e2, Num(GE2)))

            # diagonal: H(i-1,j-1) + s(i,j), seeded at (0,0) with Impulse
            diag = SAdd(
                PointwiseMul(Convolve(z, prev_h), s_row),
                PointwiseMul(Impulse(one), s_row),
            )

            # Ht = max(diag, E1, E2)
            Ht = SAdd(SAdd(diag, E1), E2)

            # F1(i,j) = max(Ht(i,j-1) + q+e,   F1(i,j-1) + e)
            F1 = Recurse(TIKernel([neginf, GE]),   Convolve(TIKernel([neginf, GO_GE]),   Ht))
            # F2(i,j) = max(Ht(i,j-1) + q2+e2, F2(i,j-1) + e2)
            F2 = Recurse(TIKernel([neginf, GE2]),  Convolve(TIKernel([neginf, GO2_GE2]), Ht))

            # H(i,j) = max(Ht, F1, F2)
            H = SAdd(SAdd(Ht, F1), F2)
            return SignalVector([H, E1, E2])

        return Ith(RepeaterVec(3, body).at(0), 0)


# Build 1-piece (Gotoh) affine expression: the genuinely cheaper kernel used for
# ksw_extz2_sse (minimap2's 1-piece entry, taken when q==q2 && e==e2). Only ONE
# gap-open/extend pair, so a single E and single F gap state — no E2/F2 work.
#   gap(L) = q + L*e
def _build_minimap2_single_affine_expr(elt=None, gaps=None):
    from vecrec.expr import (
        Convolve, Impulse, Ith, Num, PointwiseMul,
        Recurse, RepeaterVec, RVec2D, SAdd, SignalVector, SignalVectorExpr,
        TIKernel, Type, Var2D, with_types,
    )
    from vecrec.util import ElementType

    GE, GO_GE = gaps or (GE_S, GO_GE_S)

    if elt is None:
        elt = ElementType.I8 if ELEMENT_WIDTH == 8 else ElementType.I16
    with with_types(Type.TropMax, elt):
        s = Var2D("s")
        neginf = Type.TropMax.zero()
        one    = Type.TropMax.one()
        z      = TIKernel.z()

        def body(prev: RVec2D) -> SignalVectorExpr:
            prev_h  = prev.at(0).at(0)   # H(i-1, j)
            prev_e1 = prev.at(1).at(0)   # E1(i-1, j)
            s_row   = Ith(s, 0)

            # E1(i,j) = max(H(i-1,j) + q+e,   E1(i-1,j) + e)
            E1 = SAdd(PointwiseMul(prev_h,  Num(GO_GE)),   PointwiseMul(prev_e1, Num(GE)))

            # diagonal: H(i-1,j-1) + s(i,j), seeded at (0,0) with Impulse
            diag = SAdd(
                PointwiseMul(Convolve(z, prev_h), s_row),
                PointwiseMul(Impulse(one), s_row),
            )

            # Ht = max(diag, E1)
            Ht = SAdd(diag, E1)

            # F1(i,j) = max(Ht(i,j-1) + q+e,   F1(i,j-1) + e)
            F1 = Recurse(TIKernel([neginf, GE]),   Convolve(TIKernel([neginf, GO_GE]),   Ht))

            # H(i,j) = max(Ht, F1)
            H = SAdd(Ht, F1)
            return SignalVector([H, E1])

        return Ith(RepeaterVec(2, body).at(0), 0)


def _build_expr():
    return _build_minimap2_dual_affine_expr()


# ── runtime-gap header post-patch ─────────────────────────────────────────────

_RUNTIME_TAP_KERNEL = """
// ── injected by generate_kernel.py: runtime-coefficient tap kernel ───────────
// Identical hot-loop body to TimeInvariantKernel, but the tap values are held in
// a constructor-initialised member array instead of compile-time template
// constants. This lets the affine gap penalties be supplied at RUNTIME without
// touching any vectorised inner loop.
template <int _taps, typename vec_type, const int indices[_taps]>
struct RuntimeTapKernel {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static const int *idxs = indices;
    constexpr static int taps = _taps;
    elt_type vals[_taps];

    template <typename... Args>
    inline __attribute__((always_inline)) RuntimeTapKernel(Args... args)
        : vals{ static_cast<elt_type>(args)... } {}

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        _Pragma("unroll") for (int i = 0; i < taps; i++) {
            reg[base + i] = vals[i];
        }
    }

    void reset_and_next_row() {}
    void seek_col(int /* col_start */) {}
};
"""


def _decode_element(val: int, max_mult: int):
    """Return (kind, param, mult) for a single int16 tap/constant literal.

    kind is 'zero' (structural 0), 'gap' (a runtime gap penalty, possibly scaled
    by a dilation multiple), or 'other' (structural non-gap)."""
    if val == 0:
        return ("zero", None, 0)
    a = abs(val)
    hits = []
    for sval, param in _SENTINEL_TO_PARAM.items():
        if a % sval == 0:
            m = a // sval
            if 1 <= m <= max_mult:
                hits.append((param, m))
    if len(hits) == 1:
        return ("gap", hits[0][0], hits[0][1])
    if len(hits) == 0:
        return ("other", None, 0)
    raise RuntimeError(f"ambiguous gap literal {val}: matches {hits}")


def _element_expr(kind: str, param: str, mult: int, ctype: str = None) -> str:
    ctype = ctype or ELT_CTYPE
    if kind == "zero":
        return "0"
    if mult == 1:
        return param
    return f"static_cast<{ctype}>({mult} * {param})"


def patch_runtime_gaps(hdr: str, max_mult: int, kernel_name: str,
                       required: set[str], ctype: str = None) -> str:
    """Rewrite one generated kernel struct so its affine gap penalties become
    runtime constructor parameters. `kernel_name` selects which struct to patch
    (e.g. "k_vecrec" for the 2-piece kernel, "k_vecrec_1p" for the 1-piece one);
    `required` is the exact set of gap params that must end up wired (all four
    for 2-piece, {ge, go_ge} for 1-piece). Fails loudly if the codegen output no
    longer matches the expected shape. Does NOT inject the RuntimeTapKernel
    template (that is added once by the caller so this can run per kernel)."""
    ctype = ctype or ELT_CTYPE
    # Ordered ctor params to append, filtered to those this kernel needs.
    all_params = ["ge", "go_ge", "ge2", "go2_ge2"]
    ctor_params = [p for p in all_params if p in required]

    start = hdr.index("struct %s {" % kernel_name)
    end = hdr.index("using %s_vector_type" % kernel_name, start)
    head, struct, tail = hdr[:start], hdr[start:end], hdr[end:]

    # The real correctness invariant (schedule-independent): every one of the
    # four runtime gap penalties must end up wired into the kernel. We assert
    # `used_params` covers all four at the end; the per-element ambiguity/mixed
    # checks guard against silently misdecoding a folded literal.
    used_params: set[str] = set()

    # 1. find gap value arrays and their per-element runtime expressions
    gap_arrays: dict[str, list[str]] = {}
    for m in re.finditer(
        rf"constexpr static {re.escape(ctype)} (v\d+)\[\d+\]\s*=\s*\{{([^}}]*)\}};", struct
    ):
        name = m.group(1)
        vals = [int(x.strip()) for x in m.group(2).split(",")]
        decoded = [_decode_element(v, max_mult) for v in vals]
        if any(k == "gap" for (k, _, _) in decoded):
            if any(k == "other" for (k, _, _) in decoded):
                raise RuntimeError(f"array {name}={vals} mixes gap and structural literals")
            gap_arrays[name] = [_element_expr(k, p, mt, ctype) for (k, p, mt) in decoded]
            used_params.update(p for (k, p, _) in decoded if k == "gap")

    # Delete the now-dead value-array declarations: RuntimeTapKernel drops the
    # values-array template argument entirely (only keeps taps/vec_type/
    # indices), so these become unused. Harmless dead code at I16 (the
    # sentinel-scaled literal always fits int16), but at I8 a literal like
    # `16 * -101 = -1616` doesn't fit int8_t and is a hard narrowing-conversion
    # compile error if left in place — so always strip them, not just at I8.
    for arr_name in gap_arrays:
        struct, n = re.subn(
            rf"\s*constexpr static {re.escape(ctype)} {arr_name}\[\d+\]\s*=\s*\{{[^}}]*\}};\n",
            "\n", struct, count=1,
        )
        if n != 1:
            raise RuntimeError(f"could not delete dead value-array declaration for {arr_name}")

    if not gap_arrays:
        raise RuntimeError(
            "no gap tap arrays found; VecRec codegen output changed — "
            "update generate_kernel.py."
        )

    # 2. convert each gap TimeInvariantKernel decl -> RuntimeTapKernel and record
    #    the member name so we can populate its ctor args in the init list.
    member_init_exprs: dict[str, list[str]] = {}
    for arr_name, exprs in gap_arrays.items():
        pat = re.compile(
            rf"TimeInvariantKernel<\s*(\d+)\s*,\s*([^,]+?)\s*,\s*(v\d+)\s*,\s*{arr_name}\s*>\s*(v\d+);"
        )
        mm = pat.search(struct)
        if mm is None:
            raise RuntimeError(f"no TimeInvariantKernel decl uses gap array {arr_name}")
        member = mm.group(4)
        member_init_exprs[member] = exprs
        struct = pat.sub(
            rf"RuntimeTapKernel<\1, \2, \3> \4;", struct, count=1
        )

    # 3. split the ctor (signature + init list + body)
    ctor = re.search(rf"({kernel_name}\([^)]*)\)\s*:\s*(.*?)\{{\}}", struct, re.S)
    if ctor is None:
        raise RuntimeError(f"could not locate {kernel_name} constructor")
    sig, init_list = ctor.group(1), ctor.group(2)

    # 3a. append runtime gap params to the ctor signature
    new_sig = sig + "".join(f", {ctype} {p}" for p in ctor_params)

    # 3b. Signal1DConstant gap members: replace literal init arg with the param
    def _const_repl(m: re.Match) -> str:
        member, lit = m.group(1), int(m.group(2))
        dec = _decode_element(lit, max_mult)
        if dec[0] != "gap":
            return m.group(0)
        used_params.add(dec[1])
        return f"{member}({_element_expr(*dec, ctype=ctype)})"

    new_init = re.sub(r"(v\d+)\((-?\d+)\)", _const_repl, init_list)

    if used_params != required:
        raise RuntimeError(
            f"runtime gap wiring incomplete for {kernel_name}: wired {sorted(used_params)}, "
            f"need {sorted(required)}. VecRec codegen shape changed — "
            "update generate_kernel.py."
        )

    # 3c. converted RuntimeTapKernel members: fill their (currently empty) ctor args
    for member, exprs in member_init_exprs.items():
        token = re.compile(rf"\b{member}\(\)")
        if not token.search(new_init):
            raise RuntimeError(f"converted member {member} not found as empty init in ctor")
        new_init = token.sub(f"{member}({', '.join(exprs)})", new_init, count=1)

    struct = struct.replace(ctor.group(0), f"{new_sig}) :\n              {new_init}{{}}", 1)

    return head + struct + tail


# ── schedule selection ────────────────────────────────────────────────────────

from vecrec.util import ElementType

try:
    from vecrec.util import get_native_bits
    max_bits = get_native_bits()
except OSError:
    max_bits = 256
# Allow overriding the vector width (controls schedule epi) for experiments.
max_bits = int(os.environ.get("VECREC_MAX_BITS", max_bits))

# ── schedule selection ────────────────────────────────────────────────────────
# Search on a 256x256 matrix, close to minimap2's typical extension size. The
# search is VecRec's standard two-stage path: benchmark all candidate schedules,
# then sweep codegen settings for the top candidates with correctness checks.
# Use minimap2's map-ont gap penalties while searching: the prime sentinels
# above exist only to identify literals for the runtime-gap patch and would
# overflow I8 in optimized schedules. Schedule indices are independent of the
# coefficient values; the selected index is regenerated with sentinels below.
_legacy_idx = int(sys.argv[1]) if len(sys.argv) > 1 else None
TUNE_SIZE = int(os.environ.get("VECREC_TUNE_SIZE", "256"))
TUNE_ITERS = int(os.environ.get("VECREC_TUNE_ITERS", "100"))
if TUNE_SIZE <= 0 or TUNE_ITERS <= 0:
    raise ValueError("VECREC_TUNE_SIZE and VECREC_TUNE_ITERS must be positive")
TUNE_GAPS_2P = (-2, -6, -1, -25)  # options.c: e, q+e, e2, q2+e2
TUNE_GAPS_1P = TUNE_GAPS_2P[:2]


def _search_schedule(expr, cap):
    """Mirror find_best_schedule_and_emit_header with minimap2-sized inputs.

    The pinned helper fixes its benchmark at 1024 columns and does not expose
    an I8-safe candidate cap; both stages here call its public search primitives.
    """
    schedules = get_candidate_schedules(expr, max_bits)
    total_candidates = len(schedules)
    if cap is not None:
        schedules = schedules[:cap + 1]
    if not schedules:
        raise RuntimeError("VecRec generated no candidate schedules")
    bench = dict(row_size=TUNE_SIZE, input_size=TUNE_SIZE * TUNE_SIZE,
                 warmup_iterations=10, benchmark_iterations=TUNE_ITERS)
    raw = generate_and_run_benchmark(
        CodeGen(), schedules, include_correctness_check=True, **bench
    )
    output = raw["output"]
    timing_keys = [key for key in output if key != "validation"]
    n_top = min(len(schedules), max(2, math.ceil(0.1 * len(schedules))))
    top_indices = [int(key[1:]) for key in
                   sorted(timing_keys, key=lambda key: output[key], reverse=True)[:n_top]]
    if 0 not in top_indices:
        top_indices.append(0)
    non_baseline = [idx for idx in top_indices if idx != 0]
    if len(non_baseline) > MAX_SWEEP_SCHEDULES:
        kept = set(non_baseline[:MAX_SWEEP_SCHEDULES]) | {0}
        top_indices = [idx for idx in top_indices if idx in kept]
    sweep = sweep_codegen_setup([schedules[idx] for idx in top_indices], **bench)
    best_local, best = max(enumerate(sweep),
                           key=lambda pair: pair[1]["best_throughput_mbs"])
    if best["best_throughput_mbs"] <= 0:
        raise RuntimeError("VecRec schedule search found no valid candidate")
    return {"best_schedule_index": top_indices[best_local],
            "best_ele_per_iter_multiple": best["best_ele_per_iter_multiple"],
            "best_rows_per_iter": best["best_rows_per_iter"],
            "best_throughput_mbs": best["best_throughput_mbs"],
            "num_schedules": len(schedules), "total_candidates": total_candidates}


def _select_schedule(label, builder, elt, pieces, width):
    prefix = f"VECREC_SCHED_{pieces}P"
    specific = f"{prefix}_I8" if width == 8 else prefix
    specific_override = os.environ.get(specific)
    override = specific_override
    if override is None and width == 8:
        override = os.environ.get(prefix)
    if override is None and _legacy_idx is not None:
        override = str(_legacy_idx)
    mult_override = os.environ.get(f"VECREC_MULT_{pieces}P")
    # Limit I8 to the previously validated schedule range; the runtime guard
    # below checks actual tap scaling for the selected codegen setting.
    cap = 11 if width == 8 else None
    if override is not None:
        idx = int(override)
        if idx < 0 or (cap is not None and specific_override is not None and idx > cap):
            raise ValueError(f"{label}: schedule index {idx} is outside the safe range")
        if cap is not None:
            idx = min(idx, cap)
        # Keep the legacy positional/general override behavior: an index beyond
        # a variant's candidate list uses its last candidate.
        available = len(get_candidate_schedules(builder(elt), max_bits))
        if idx >= available:
            print(f"[schedule] {label}: k{idx} exceeds {available} candidates; "
                  f"using k{available - 1}", flush=True)
            idx = available - 1
        mult = int(mult_override) if mult_override is not None else 2
        rpi = int(os.environ.get(f"VECREC_RPI_{pieces}P", "1"))
        print(f"[schedule] {label}: explicit k{idx}, epi_mult={mult}, rpi={rpi}", flush=True)
        return idx, mult, rpi, {"selection": "explicit", "schedule_index": idx,
                                "ele_per_iter_multiple": mult, "rows_per_iter": rpi}

    gaps = TUNE_GAPS_2P if pieces == 2 else TUNE_GAPS_1P
    print(f"[schedule] {label}: searching VecRec candidates at {TUNE_SIZE}x{TUNE_SIZE}", flush=True)
    result = _search_schedule(builder(elt, gaps=gaps), cap)
    final_candidates = len(get_candidate_schedules(builder(elt), max_bits))
    if final_candidates != result["total_candidates"]:
        raise RuntimeError(f"{label}: tuning and runtime-gap expressions have "
                           "different schedule counts")
    idx = result["best_schedule_index"]
    mult = int(mult_override) if mult_override is not None else result["best_ele_per_iter_multiple"]
    rpi = int(os.environ.get(f"VECREC_RPI_{pieces}P", result["best_rows_per_iter"]))
    print(f"[schedule] {label}: k{idx}, epi_mult={mult}, "
          f"rpi={rpi}, "
          f"{result['best_throughput_mbs']:.1f} MB/s "
          f"({result['num_schedules']} candidates)", flush=True)
    return idx, mult, rpi, {"selection": "searched", "schedule_index": idx,
                       "ele_per_iter_multiple": mult,
                       "rows_per_iter": rpi,
                       "throughput_mbs": result["best_throughput_mbs"],
                       "num_schedules": result["num_schedules"],
                       "tune_gaps": gaps,
                       "tune_size": TUNE_SIZE, "tune_iterations": TUNE_ITERS}


SELECTIONS = {}
for _label, _builder, _elt, _pieces, _width in (
    ("2p_i16", _build_minimap2_dual_affine_expr, ElementType.I16, 2, 16),
    ("2p_i8", _build_minimap2_dual_affine_expr, ElementType.I8, 2, 8),
    ("1p_i16", _build_minimap2_single_affine_expr, ElementType.I16, 1, 16),
    ("1p_i8", _build_minimap2_single_affine_expr, ElementType.I8, 1, 8),
):
    SELECTIONS[_label] = _select_schedule(_label, _builder, _elt, _pieces, _width)

BAND_WIDTH = 500   # codegen ring-buffer sizing only; the band is runtime (arg w)


def _codegen(expr, name, sched_idx, mult, rpi):
    """Generate the searched schedule at its selected codegen setting."""
    schedules = get_candidate_schedules(expr, max_bits)
    if not 0 <= sched_idx < len(schedules):
        raise RuntimeError(f"{name}: schedule k{sched_idx} missing from {len(schedules)} candidates")
    idx = sched_idx
    cg = CodeGen(band_width=BAND_WIDTH, ele_per_iter_multiple=mult,
                 rows_per_iter=rpi)
    with contextlib.redirect_stdout(io.StringIO()):
        code = cg.generate(schedules[idx], name)
    return code, idx


# Dual element width: each (piece-count, width) is a distinct kernel struct so the
# wrapper can pick I8 vs I16 at runtime by the max achievable score. I8 doubles
# the lane count (faster) but SATURATES past ~127, so it is used only for
# provably-small extensions; I16 is the always-correct fallback.
_KSPEC = [
    # struct,        builder,                             elt,             sched,       mult,    pieces, ctype,     required
    ("k_vr_2p_i16", _build_minimap2_dual_affine_expr,   ElementType.I16, *SELECTIONS["2p_i16"][:3], 2, "int16_t", {"ge", "go_ge", "ge2", "go2_ge2"}),
    ("k_vr_2p_i8",  _build_minimap2_dual_affine_expr,   ElementType.I8,  *SELECTIONS["2p_i8"][:3],  2, "int8_t",  {"ge", "go_ge", "ge2", "go2_ge2"}),
    ("k_vr_1p_i16", _build_minimap2_single_affine_expr, ElementType.I16, *SELECTIONS["1p_i16"][:3], 1, "int16_t", {"ge", "go_ge"}),
    ("k_vr_1p_i8",  _build_minimap2_single_affine_expr, ElementType.I8,  *SELECTIONS["1p_i8"][:3],  1, "int8_t",  {"ge", "go_ge"}),
]

KMETA: dict = {}   # struct name -> dict(code, sched, epi, rpi, ctype, pieces, required)
for (kname, builder, elt, sidx, mult, rpi, pieces, ctype, required) in _KSPEC:
    _code, _used_idx = _codegen(builder(elt), kname, sidx, mult, rpi)
    KMETA[kname] = dict(code=_code, sched=_used_idx, epi=_code.element_per_iter,
                        rpi=_code.rows_per_iter, ctype=ctype, pieces=pieces,
                        required=required)

# ── output paths ─────────────────────────────────────────────────────────────

out_dir = _REPO / "end_to_end_baselines" / "minimap2" / "generated"
out_dir.mkdir(parents=True, exist_ok=True)

hdr_path = str(out_dir / "vecrec_affine_kernel.h")
src_path = str(out_dir / "ksw2_vecrec_ext.cpp")

# ── generate, then post-patch the kernel header for runtime gaps ──────────────

fd, tmp_hdr = tempfile.mkstemp(suffix=".h", prefix="vecrec_mm2_")
os.close(fd)
# All kernels share ONE common.h (instantiate_kernels emits the template once).
instantiate_kernels(tmp_hdr, [KMETA[n]["code"] for n in KMETA])

hdr = pathlib.Path(tmp_hdr).read_text()
pathlib.Path(tmp_hdr).unlink(missing_ok=True)

# Patch InputSignal2D → ProfileSignal2D (avoids materialising the O(N²) matrix).
# All kernels use Var2D("s"), so these global replaces cover every struct.
svar = "s"
hdr = hdr.replace("InputSignal2D<", "ProfileSignal2D<")
hdr = hdr.replace(f"({svar}, {svar}_stride,", f"({svar}, {svar}_sym, {svar}_stride,")
hdr = re.sub(rf"\*{svar}, int {svar}_stride",
             f"*{svar}, const int *{svar}_sym, int {svar}_stride", hdr)

# Inject the runtime-tap kernel template ONCE (before the first kernel struct),
# then rewrite each kernel's gap penalties into runtime ctor params separately.
_first = next(iter(KMETA))
hdr = hdr.replace(f"struct {_first} {{",
                  _RUNTIME_TAP_KERNEL + f"\nstruct {_first} {{", 1)
for _kname, _m in KMETA.items():
    hdr = patch_runtime_gaps(hdr, _m["epi"], _kname, _m["required"], _m["ctype"])


def _gap_multipliers(header: str, kernel_name: str, ctype: str,
                     params: set[str]) -> dict[str, int]:
    """Read the largest runtime scaling factor of each gap tap in a kernel."""
    start = header.index(f"struct {kernel_name} {{")
    end = header.index(f"using {kernel_name}_vector_type", start)
    body = header[start:end]
    return {param: max([1] + [int(v) for v in re.findall(
        rf"static_cast<{re.escape(ctype)}>\((\d+) \* {param}\)", body
    )]) for param in params}


for _kname, _m in KMETA.items():
    _m["gap_multipliers"] = _gap_multipliers(hdr, _kname, _m["ctype"], _m["required"])

pathlib.Path(hdr_path).write_text(hdr)

# ── generate the minimap2 ksw2 replacement source ─────────────────────────────

ksw2_h_rel = "../minimap2/ksw2.h"

SRC = r"""// Auto-generated by end_to_end_baselines/scripts/generate_kernel.py
// DO NOT EDIT — re-run the script to regenerate.
//
// Replaces minimap2's affine extension with vecrec FULL-MATRIX kernels:
//   ksw_extz2_sse -> vecrec_ext_core_1p  (dedicated 1-piece Gotoh: H,E1,F1)
//   ksw_extd2_sse -> vecrec_ext_core_2p  (dual-affine 2-piece: H,E1,E2,F1,F2)
// ksw_exts2_sse (splice) is intentionally NOT defined here so minimap2 links
// ksw2's original splice kernel — only the affine extension changes.
//
// Compile-time ablation switches (VR_*, default off) live at the top of the
// generated body; see CLAUDE.md.
//
// Scope: this drop-in is valid ONLY when
//   (1) banding does not apply  — the whole R×C matrix is computed (w ignored);
//       this is the measured 99.9% case for minimap2's map-ont extension calls.
//   (2) z-drop does not exist   — ez->zdropped is always 0 (caller runs -z big).
//
// Drop-in behaviour:
//   * runtime gap penalties  — q/e/q2/e2 from the caller are passed into the
//     kernel constructor (no baked-in penalties).
//   * w, zdrop               — ACCEPTED but UNUSED (full matrix, no z-drop).
//   * full ksw_extz_t outputs — max/max_q/max_t, mqe/mqe_t, mte/mte_q, reach_end.
//   * real CIGAR             — 2-piece-aware traceback with distinct gap-LEFT
//     and gap-RIGHT alignment paths (selected by KSW_EZ_RIGHT).
//   * memory                 — all scratch comes from minimap2's kalloc pool.
//
// Flag assertions (score-only / approx-max / z-drop output) live in align.c's
// mm_align_pair so BOTH the ksw2 and vecrec builds check identical conditions
// (fair comparison). This file only enforces the full-matrix/no-z-drop contract.
//
// schedules: 2-piece k@SCHED@ epi=@EPI@ rpi=@RPI@ | 1-piece k@SCHED1@ epi=@EPI1@ rpi=@RPI1@ (full matrix, no banding)

#ifdef USE_VECREC

#include "vecrec_affine_kernel.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cassert>
#include <vector>

// ── ablation switches (compile-time; default 0 = full/correct behavior) ──────
// Pass via VECREC_EXTRA_DEFS (forwarded by build_minimap2.sh), e.g.
//   VECREC_EXTRA_DEFS="-DVR_NO_TRACEBACK=1 -DVR_DUMMY_PROFILE=1"
// Flags marked (timing only) intentionally produce WRONG alignments; use them to
// attribute per-phase cost, never for a parity run. See CLAUDE.md.
#ifndef VR_FORCE_2P
#define VR_FORCE_2P 0          // route the 1-piece entry through the 2-piece core
#endif
#ifndef VR_NO_TRACEBACK
#define VR_NO_TRACEBACK 0      // (timing only) skip traceback; emit trivial CIGAR
#endif
#ifndef VR_DUMMY_PROFILE
#define VR_DUMMY_PROFILE 0     // (timing only) skip the score-profile build
#endif
#ifndef VR_NO_HSTORE
#define VR_NO_HSTORE 0         // (timing only) do not store H; implies no tb/endpoint
#endif
#ifndef VR_NO_ENDPOINT_SCAN
#define VR_NO_ENDPOINT_SCAN 0  // (timing only) skip mqe/mte last-row/col scans
#endif
#ifndef VR_CHECK
#define VR_CHECK 0             // 1: assert corner+gmax vs scalar ref; 2: every cell
#endif
#ifndef VR_I8_MAX
// Use the narrow (2x lanes, faster) I8 kernel only when the max achievable score
// provably cannot saturate int8: min(qlen,tlen)*max_match + end_bonus <= this.
// Otherwise fall back to the I16 kernel. Keep margin below 127. The negative
// side is bounded separately by VR_I8_SAFE_MAG / VR_I16_SAFE_MAG below.
#define VR_I8_MAX 120
#endif
#ifndef VR_BAND_PACKED
// Banded H storage: store only each row's in-band strip (O(R*band_stride)) instead
// of the full R*C matrix. Set 0 to restore the legacy full-matrix banded fill (for
// an A/B of the memory-traffic saving). No effect on the full-matrix (exact) path.
#define VR_BAND_PACKED 1
#endif

// Derived: NO_HSTORE forces traceback + endpoint scan off (both read H).
#if VR_NO_HSTORE
#  define VR_DO_HSTORE    0
#  define VR_DO_TRACEBACK 0
#  define VR_DO_ENDPOINT  0
#else
#  define VR_DO_HSTORE    1
#  define VR_DO_TRACEBACK (!VR_NO_TRACEBACK)
#  define VR_DO_ENDPOINT  (!VR_NO_ENDPOINT_SCAN)
#endif

extern "C" {
#include "@KSW2_H@"
}

// Original ksw2 affine kernels, linked in under renamed symbols (see
// build_minimap2.sh). Calls carrying KSW_EZ_APPROX_MAX are delegated to these:
// vecrec computes an EXACT max, it has no approximate-max mode, so the faithful
// behaviour for an approx-max call is to run stock ksw2.
extern "C" void ksw_extz2_sse_orig(void *km,
        int qlen, const uint8_t *query, int tlen, const uint8_t *target,
        int8_t m, const int8_t *mat,
        int8_t q, int8_t e, int w, int zdrop, int end_bonus, int flag,
        ksw_extz_t *ez);
extern "C" void ksw_extd2_sse_orig(void *km,
        int qlen, const uint8_t *query, int tlen, const uint8_t *target,
        int8_t m, const int8_t *mat,
        int8_t q, int8_t e, int8_t q2, int8_t e2,
        int w, int zdrop, int end_bonus, int flag,
        ksw_extz_t *ez);

// ── always-on phase instrumentation (init / DP / traceback) ──────────────────
// Mirrors the [extend-phases] line the ksw2 build prints from ksw2_extd2_sse.c,
// so the two builds are directly comparable. Thread-safe (atomic) for -t N.
namespace {
long vr_approx_delegated = 0;   // approx-max calls handed to stock ksw2 (opt-in)
// approx-max policy: by DEFAULT vecrec handles KSW_EZ_APPROX_MAX calls itself
// (its exact full-matrix max is a superset of ksw2's approximate max). Set
// VECREC_DELEGATE_APPROX=1 to restore delegation to stock ksw2 (for A/B timing).
static inline int vr_delegate_approx() {
    static int d = -1;
    if (d < 0) d = std::getenv("VECREC_DELEGATE_APPROX") ? 1 : 0;
    return d;
}
}  // namespace

// ── phase-timing instrumentation (OPT-IN: -DVECREC_PHASE_TIMING) ─────────────
// OFF by default. The per-call atomic accumulators contend on shared cache lines
// under -t N and badly distort multi-thread scaling, so timing must be a
// deliberate profiling build, never part of a perf comparison. When off, the
// VR_* macros expand to nothing (zero overhead).
#ifdef VECREC_PHASE_TIMING
namespace {
long vr_init_ns = 0, vr_dp_ns = 0, vr_tb_ns = 0, vr_calls = 0;
static inline long vr_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}
struct VRPhaseReport {
    ~VRPhaseReport() {
        long nc = vr_calls > 0 ? vr_calls : 1;
        std::fprintf(stderr,
            "[vecrec-phases] init=%.2f ms  dp=%.2f ms  traceback=%.2f ms"
            "  |  per call: init=%.2f us  dp=%.2f us  tb=%.2f us  calls=%ld"
            "  approx_delegated=%ld\n",
            vr_init_ns * 1e-6, vr_dp_ns * 1e-6, vr_tb_ns * 1e-6,
            vr_init_ns * 1e-3 / nc, vr_dp_ns * 1e-3 / nc, vr_tb_ns * 1e-3 / nc,
            vr_calls, vr_approx_delegated);
    }
};
static VRPhaseReport _vr_phase_report;
}  // namespace
#define VR_T0(x)      long x = vr_now_ns()
#define VR_ADD(acc,x) __atomic_fetch_add(&acc, vr_now_ns() - (x), __ATOMIC_RELAXED)
#define VR_CALLS()    __atomic_fetch_add(&vr_calls, 1, __ATOMIC_RELAXED)
#else
#define VR_T0(x)      ((void)0)
#define VR_ADD(acc,x) ((void)0)
#define VR_CALLS()    ((void)0)
#endif

namespace {

// Reconstruct predecessors from the stored full H matrix and greedily walk back
// to the origin. `right_align` selects tie preference: false => gaps placed left
// (diagonal preferred on ties), true => gaps placed right (gaps preferred). H is
// the true DP matrix, so at least one predecessor identity holds at every cell —
// the greedy walk always yields a valid path. op codes: 0=M, 1=I (consume
// query), 2=D (consume target). Returns nops (ops filled in reverse order).
// Traceback over the stored H matrix only (E/F are not stored). The kernel
// computes the ksw2 recurrence
//   E(i,j)  = max_k H(i-k,j)  - gap(k)          vertical gap (I), opens from H
//   Ht(i,j) = max(H(i-1,j-1) + s(i,j), E(i,j))
//   F(i,j)  = max_k Ht(i,j-k) - gap(k)          horizontal gap (D), opens from Ht
//   H(i,j)  = max(Ht(i,j), F(i,j))
// with gap(k) = min(q+k*e, q2+k*e2) (PIECES==2; q+k*e for PIECES==1). At each
// cell we pick a predecessor whose value plus the transition score equals the
// current cell's value, searching gap lengths k = 1, 2, ... so the emitted
// CIGAR re-scores to exactly H at the start cell. Because gap() is
// subadditive (positive open costs), checking H(i,j-k) for D runs is always
// sufficient; the Ht-based search below is only a safety net.
// Tie-breaking: gap-LEFT (default) prefers M, then I, then D; gap-RIGHT
// prefers a length-1 D, then a length-1 I, then M. Longer gaps are only
// searched when no length-1 step or M explains the cell.
// PIECES: 1 checks only the (ge,go_ge) gap costs; 2 also checks (ge2,go2_ge2).
template <int PIECES, typename HT>
static int vecrec_traceback(
        const HT *Hmat, int phys_w,
        const uint8_t *query, const uint8_t *target,
        int8_t m, const int8_t *mat,
        int ge, int go_ge, int ge2, int go2_ge2,
        int i, int j, uint8_t *ops, bool right_align,
        const int *cs_row, int band_stride, int neg) {
    (void)ge2; (void)go2_ge2;
    // Band-packed layout when cs_row != nullptr (phys_w then unused); else the
    // full-matrix layout with row stride phys_w. Out-of-band reads give `neg`.
    auto H = [&](int r, int c) -> int {
        if (cs_row) {
            int off = c - cs_row[r];
            if (off < 0 || off >= band_stride) return neg;
            return static_cast<int>(Hmat[static_cast<std::size_t>(r) * band_stride + off]);
        }
        return static_cast<int>(Hmat[static_cast<std::size_t>(r) * phys_w + c]);
    };
    // Score (<= 0) of a gap of length k >= 1.
    auto gap = [&](int k) -> int {
        int g = go_ge + (k - 1) * ge;
        if constexpr (PIECES == 2) {
            int g2 = go2_ge2 + (k - 1) * ge2;
            if (g2 > g) g = g2;
        }
        return g;
    };
    auto sub = [&](int r, int c) -> int {
        return static_cast<int>(mat[(int)query[r - 1] * m + (int)target[c - 1]]);
    };
    // Ht(r,c) recomputed from H (safety net only; O(r) per call).
    auto Ht = [&](int r, int c) -> int {
        if (r == 0) return c == 0 ? H(0, 0) : neg;
        int best = (c > 0) ? H(r - 1, c - 1) + sub(r, c) : neg;
        for (int k = 1; k <= r; ++k) {
            int v = H(r - k, c) + gap(k);
            if (v > best) best = v;
        }
        return best;
    };
    int nops = 0;
    int cur = H(i, j);      // value the path into (i,j) must explain
    bool at_ht = false;     // true: (i,j) is reached through Ht, so no D step here
    while (i > 0 && j > 0) {
        const bool can_M = (cur == H(i - 1, j - 1) + sub(i, j));
        const bool can_I1 = (cur == H(i - 1, j) + gap(1));
        const bool can_D1 = !at_ht && (cur == H(i, j - 1) + gap(1));
        int op = -1, len = 1;
        bool next_at_ht = false;
        if (!right_align) {
            if      (can_M)  op = 0;
            else if (can_I1) op = 1;
            else if (can_D1) op = 2;
        } else {
            if      (can_D1) op = 2;
            else if (can_I1) op = 1;
            else if (can_M)  op = 0;
        }
        for (int k = 2; op < 0 && k <= (i > j ? i : j); ++k) {
            if (k <= i && cur == H(i - k, j) + gap(k)) { op = 1; len = k; }
            else if (!at_ht && k <= j && cur == H(i, j - k) + gap(k)) { op = 2; len = k; }
        }
        for (int k = 1; op < 0 && !at_ht && k <= j; ++k) {
            if (cur == Ht(i, j - k) + gap(k)) { op = 2; len = k; next_at_ht = true; }
        }
        if (op < 0) {
#if VR_CHECK
            std::fprintf(stderr, "[VR_CHECK] traceback found no predecessor at (%d,%d)\n", i, j);
            assert(0 && "VR_CHECK: traceback found no predecessor");
#endif
            op = 0;   // keep the CIGAR span consistent
        }
        for (int k = 0; k < len; ++k) ops[nops++] = static_cast<uint8_t>(op);
        if (op == 0)      { --i; --j; }
        else if (op == 1) { i -= len; }
        else              { j -= len; }
        at_ht = next_at_ht;
        cur = at_ht ? Ht(i, j) : H(i, j);
    }
    while (i > 0) { ops[nops++] = 1; --i; }   // leading insertions
    while (j > 0) { ops[nops++] = 2; --j; }   // leading deletions
    return nops;
}

// ABLATION helper (VR_NO_TRACEBACK): emit a trivial path from the origin to
// (i,j) — a diagonal run plus the query/target remainder — so the emitted CIGAR
// still spans the requested endpoint (keeps mm_fix_cigar happy) at O(1) cost.
// The alignment is meaningless; this exists only to time-out the traceback.
static int vr_trivial_cigar(int i, int j, uint8_t *ops) {
    int nops = 0, d = i < j ? i : j;
    for (int k = 0; k < i - d; ++k) ops[nops++] = 1;   // extra query -> I
    for (int k = 0; k < j - d; ++k) ops[nops++] = 2;   // extra target -> D
    for (int k = 0; k < d;     ++k) ops[nops++] = 0;   // diagonal run -> M
    return nops;
}

#if VR_CHECK
// VR_CHECK verifier: an independent scalar int32 DP (Gotoh for PIECES==1,
// dual-affine for PIECES==2) recomputed from scratch — no SIMD, no difference
// encoding — so it catches codegen/encoding bugs the vectorised path could
// share. Matches the ksw2/vecrec recurrence exactly: F opens from Ht (not H).
// Aborts on any disagreement. VR_CHECK>=2 also compares every H cell.
template <int PIECES, typename HT>
static void vr_reference_check(
        int qlen, const uint8_t *query, int tlen, const uint8_t *target,
        int8_t m, const int8_t *mat, int q, int e, int q2, int e2,
        const HT *Hmat, int Hstride, int vr_gmax) {
    const int NEG = -(1 << 28);
    const int go_ge = -(q + e),  ge  = -e;
    const int go2   = -(q2 + e2), ge2 = -e2;
    const int R = qlen + 1, C = tlen + 1;
    std::vector<int> Hr((std::size_t)R * C, NEG);
    std::vector<int> E1(C, NEG), E2(C, NEG), E1n(C, NEG), E2n(C, NEG);
    int ref_max = NEG;
    for (int i = 0; i < R; ++i) {
        int F1 = NEG, F2 = NEG, Ht_left = NEG;   // running horizontal gap + Ht(i,j-1)
        for (int j = 0; j < C; ++j) {
            int diag;
            if (i == 0 && j == 0) diag = 0;
            else if (i > 0 && j > 0)
                diag = Hr[(std::size_t)(i-1)*C + (j-1)] +
                       (int)mat[(int)query[i-1]*m + (int)target[j-1]];
            else diag = NEG;
            int e1 = NEG, e2v = NEG;
            if (i > 0) {
                int hup = Hr[(std::size_t)(i-1)*C + j];
                e1 = std::max(hup + go_ge, E1[j] + ge);
                if (PIECES == 2) e2v = std::max(hup + go2, E2[j] + ge2);
            }
            int Ht = std::max(diag, e1);
            if (PIECES == 2) Ht = std::max(Ht, e2v);
            int f1 = NEG, f2 = NEG;
            if (j > 0) {
                f1 = std::max(Ht_left + go_ge, F1 + ge);
                if (PIECES == 2) f2 = std::max(Ht_left + go2, F2 + ge2);
            }
            int H = std::max(Ht, f1);
            if (PIECES == 2) H = std::max(H, f2);
            Hr[(std::size_t)i*C + j] = H;
            E1n[j] = e1; E2n[j] = e2v;
            F1 = f1; F2 = f2; Ht_left = Ht;
            if (H > ref_max) ref_max = H;
        }
        std::swap(E1, E1n); std::swap(E2, E2n);
    }
    int ref_corner = Hr[(std::size_t)qlen*C + tlen];
    int vr_corner  = (int)Hmat[(std::size_t)qlen*Hstride + tlen];
    if (ref_corner != vr_corner) {
        std::fprintf(stderr, "[VR_CHECK] corner mismatch qlen=%d tlen=%d ref=%d vr=%d\n",
                     qlen, tlen, ref_corner, vr_corner);
        assert(0 && "VR_CHECK: corner score mismatch");
    }
    if (ref_max != vr_gmax) {
        std::fprintf(stderr, "[VR_CHECK] gmax mismatch qlen=%d tlen=%d ref=%d vr=%d\n",
                     qlen, tlen, ref_max, vr_gmax);
        assert(0 && "VR_CHECK: global max mismatch");
    }
#if VR_CHECK >= 2
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j) {
            int rr = Hr[(std::size_t)i*C + j];
            int vv = (int)Hmat[(std::size_t)i*Hstride + j];
            if (rr != vv) {
                std::fprintf(stderr, "[VR_CHECK] cell (%d,%d) ref=%d vr=%d qlen=%d tlen=%d\n",
                             i, j, rr, vv, qlen, tlen);
                assert(0 && "VR_CHECK: matrix cell mismatch");
            }
        }
#endif
}
#endif  // VR_CHECK

@CORES@

}  // namespace

// ── public ksw2 extension API (affine only; splice stays original ksw2) ───────

// Safety guard: BOTH vecrec kernel widths encode their DP state as
// *differences* from neighboring cells, with a "tropical -infinity" sentinel
// baked into the SIMD template library (common.h's SRecurseOp / SConvolveOp
// Sat variants: zero_init). Every SConvolve/SRecurse node's running value
// starts at this sentinel and combines via max(), so the sentinel acts as
// that reduction's identity element. If a call's genuinely-achievable (very
// negative) score can approach the sentinel's own magnitude, a real score and
// the "no contribution yet" identity become numerically indistinguishable and
// the reduction wrongly picks the sentinel, corrupting the result.
//
// The pinned vec-rec's saturating I8/I16 kernels use `lowest(elt_type)/2`
// (-64 for int8_t, -16384 for int16_t) as their tropical "-infinity": the
// identity of every Sat-variant max() reduction (vec-rec/src/vecrec/util.py's
// val_to_str, templates/common.h's SConvolveOp/SRecurseOp/SRecurseVOp Sat
// instantiations). A real score more negative than that sentinel would lose
// to it in a max() and silently corrupt the result. Bound the worst-case
// achievable magnitude of a call and delegate to stock ksw2 -- which has no
// such sentinel -- whenever it could approach the chosen width's limit.
//
// Bound: reaching corner (qlen,tlen) is always achievable via min(qlen,tlen)
// diagonal (match-or-mismatch) steps plus |qlen-tlen| steps of one gap piece,
// so the true optimum is NEVER worse (more negative) than
// `min(qlen,tlen)*worst_mismatch + |qlen-tlen|*worst_gap_extend + gap_open`
// -- the optimizer can only do at least as well as that one concrete path.
// This is a genuine upper bound on |score| (unlike a gap-extend-only estimate,
// which is NOT valid here: reaching a `qlen==tlen` corner needs `qlen`
// diagonal steps too, and a real divergent region's optimal path can be
// dominated by mismatch cost, not gap cost -- confirmed empirically to
// undershoot the true magnitude by ~2x).
#ifndef VR_I8_SAFE_MAG
#define VR_I8_SAFE_MAG 50      // safety margin under |INT8_MIN/2|=64
#endif
#ifndef VR_I16_SAFE_MAG
#define VR_I16_SAFE_MAG 14000  // safety margin under |INT16_MIN/2|=16384
#endif

extern "C" void ksw_extz2_sse(void *km,
        int qlen, const uint8_t *query, int tlen, const uint8_t *target,
        int8_t m, const int8_t *mat,
        int8_t q, int8_t e, int w, int zdrop, int end_bonus, int flag,
        ksw_extz_t *ez) {
    // approx-max: vecrec handles it exactly by default; delegate only if opted in.
    if ((flag & KSW_EZ_APPROX_MAX) && vr_delegate_approx()) {
        __atomic_fetch_add(&vr_approx_delegated, 1, __ATOMIC_RELAXED);
        ksw_extz2_sse_orig(km, qlen, query, tlen, target, m, mat, q, e, w, zdrop, end_bonus, flag, ez);
        return;
    }
    // Band-collapse early-out: if |qlen-tlen| > w the diagonal band cannot reach
    // the corner (stock ksw2 detects this mid-sweep as st>en and sets zdropped,
    // since wl=wr=w). Predict it up front from lengths alone and BAIL WITHOUT
    // running the vecrec kernel -- report a dropped extension, matching stock's
    // collapse. (The vecrec banded kernel would otherwise waste work / be wrong.)
    if (w > 0 && (qlen > tlen ? qlen - tlen : tlen - qlen) > w) {
        ksw_reset_extz(ez);
        ez->zdropped = 1;
        return;
    }
    // Element-width dispatch: one pass over `mat` for both the max (match) and
    // min (worst mismatch) entries, feeding both the I8-saturation bound
    // (score-magnitude, positive side) and the sentinel-collision bound
    // (score-magnitude, negative side; see VR_I8_SAFE_MAG/VR_I16_SAFE_MAG above).
    int _mmax = 0, _mmin = 0;
    for (int _k = 0; _k < (int)m * (int)m; ++_k) {
        if (mat[_k] > _mmax) _mmax = mat[_k];
        if (mat[_k] < _mmin) _mmin = mat[_k];
    }
    long _minlen = (qlen < tlen ? qlen : tlen);
    long _dlen   = (qlen > tlen ? qlen - tlen : tlen - qlen);
    long _worst_mag = _minlen * (long)(-_mmin) + _dlen * (long)e + (long)q + (long)e;
    if (_worst_mag > VR_I16_SAFE_MAG) {
        ksw_extz2_sse_orig(km, qlen, query, tlen, target, m, mat, q, e, w, zdrop, end_bonus, flag, ez);
        return;
    }
    long _ub = _minlen * _mmax + (end_bonus > 0 ? end_bonus : 0);
    // A searched schedule may scale gap taps inside an int8 register. Check the
    // generated factors before choosing it, since narrowing casts would wrap.
    const bool _use_i8 = (_ub <= VR_I8_MAX) && (_worst_mag <= VR_I8_SAFE_MAG)
        && (long)e * @I8_1P_GE_MULT@ <= 127
        && ((long)q + e) * @I8_1P_GO_GE_MULT@ <= 127;
    // 1-piece affine entry (q==q2 && e==e2). Feed piece 2 == piece 1; the DEDICATED
    // 1-piece core (H,E1,F1 only — no E2/F2 work) is used by default. VR_FORCE_2P
    // routes through the 2-piece core instead, to A/B the dedicated-kernel win.
#if VR_FORCE_2P
    if (_use_i8) vecrec_ext_core_2p_i8 (km, qlen, query, tlen, target, m, mat, q, e, q, e, w, end_bonus, flag, ez);
    else         vecrec_ext_core_2p_i16(km, qlen, query, tlen, target, m, mat, q, e, q, e, w, end_bonus, flag, ez);
#else
    if (_use_i8) vecrec_ext_core_1p_i8 (km, qlen, query, tlen, target, m, mat, q, e, q, e, w, end_bonus, flag, ez);
    else         vecrec_ext_core_1p_i16(km, qlen, query, tlen, target, m, mat, q, e, q, e, w, end_bonus, flag, ez);
#endif
}

extern "C" void ksw_extd2_sse(void *km,
        int qlen, const uint8_t *query, int tlen, const uint8_t *target,
        int8_t m, const int8_t *mat,
        int8_t q, int8_t e, int8_t q2, int8_t e2,
        int w, int zdrop, int end_bonus, int flag,
        ksw_extz_t *ez) {
    // approx-max: vecrec handles it exactly by default; delegate only if opted in.
    if ((flag & KSW_EZ_APPROX_MAX) && vr_delegate_approx()) {
        __atomic_fetch_add(&vr_approx_delegated, 1, __ATOMIC_RELAXED);
        ksw_extd2_sse_orig(km, qlen, query, tlen, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag, ez);
        return;
    }
    // Band-collapse early-out (see ksw_extz2_sse): |qlen-tlen| > w cannot reach
    // the corner within the band; bail without running the vecrec kernel.
    if (w > 0 && (qlen > tlen ? qlen - tlen : tlen - qlen) > w) {
        ksw_reset_extz(ez);
        ez->zdropped = 1;
        return;
    }
    // Score-magnitude safety guard + I8/I16 dispatch prep (see ksw_extz2_sse
    // for the derivation; 2-piece uses whichever gap piece is cheaper for the
    // |qlen-tlen| term).
    int _mmax = 0, _mmin = 0;
    for (int _k = 0; _k < (int)m * (int)m; ++_k) {
        if (mat[_k] > _mmax) _mmax = mat[_k];
        if (mat[_k] < _mmin) _mmin = mat[_k];
    }
    long _minlen = (qlen < tlen ? qlen : tlen);
    long _dlen   = (qlen > tlen ? qlen - tlen : tlen - qlen);
    long _min_e  = (e < e2 ? e : e2);
    long _min_go = (((long)q + e) < ((long)q2 + e2) ? ((long)q + e) : ((long)q2 + e2));
    long _worst_mag = _minlen * (long)(-_mmin) + _dlen * _min_e + _min_go;
    if (_worst_mag > VR_I16_SAFE_MAG) {
        ksw_extd2_sse_orig(km, qlen, query, tlen, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag, ez);
        return;
    }
    long _ub = _minlen * _mmax + (end_bonus > 0 ? end_bonus : 0);
    const bool _use_i8 = (_ub <= VR_I8_MAX) && (_worst_mag <= VR_I8_SAFE_MAG)
        && (long)e * @I8_2P_GE_MULT@ <= 127
        && ((long)q + e) * @I8_2P_GO_GE_MULT@ <= 127
        && (long)e2 * @I8_2P_GE2_MULT@ <= 127
        && ((long)q2 + e2) * @I8_2P_GO2_GE2_MULT@ <= 127;
    if (_use_i8)
        vecrec_ext_core_2p_i8 (km, qlen, query, tlen, target, m, mat, q, e, q2, e2, w, end_bonus, flag, ez);
    else
        vecrec_ext_core_2p_i16(km, qlen, query, tlen, target, m, mat, q, e, q2, e2, w, end_bonus, flag, ez);
}

#endif  // USE_VECREC
"""

# ── core body template, formatted once per piece-count ────────────────────────
# Tokens (@X@) are substituted per core BEFORE insertion, so the two cores can
# carry different EPI/RPI/kernel-type without colliding with the header-comment
# substitution below.
CORE_TEMPLATE = r"""
// Core: full-matrix vecrec @PIECES@-piece affine DP + endpoint stats + gap-aware
// traceback. gapo/gape = piece 1, gapo2/gape2 = piece 2 (positive magnitudes, as
// ksw2 passes them). w/zdrop accepted but unused. Optional phases gated by VR_*.
static void @NM@(
        void *km,
        int qlen, const uint8_t *query,
        int tlen, const uint8_t *target,
        int8_t m, const int8_t *mat,
        int8_t gapo, int8_t gape, int8_t gapo2, int8_t gape2,
        int w, int end_bonus, int flag,
        ksw_extz_t *ez) {
    assert(!(flag & KSW_EZ_SCORE_ONLY));
    // Banding (APPROXIMATE): honor the caller's band `w` as a two-sided diagonal
    // band via seek_col. KNOWN LIMITATION: seek_col does NOT prime the SConvolve
    // diagonal carry, so cells at the extreme LEFT band edge can be scored
    // approximately (the documented diagonal-carry init issue); the result is
    // exact only when the optimum stays band-interior (roughly w >= 4*epi). Full
    // matrix is used when w<=0 or w>=len-1 (the common minimap2 case).
    const int _maxlen = (qlen > tlen ? qlen : tlen);
    const bool banded = (w > 0 && w < _maxlen - 1);

    constexpr int EPI = @CEPI@;
    constexpr int RPI = @CRPI@;
    constexpr int PIECES = @PIECES@;

    const int C = tlen + 1;
    const int phys_w = ((C + EPI - 1) / EPI) * EPI;
    const int R = qlen + 1;
    const int N = R * phys_w;

    using elt_type = @ELTYPE@;
    constexpr int VR_NEG = @VRNEG@;  // tropical -inf sentinel (per element width)
    VR_T0(_t_init0);

    // Runtime tropical gap parameters (negative penalties) for the kernel ctor.
    const @ELTYPE@ ge       = static_cast<@ELTYPE@>(-(int)gape);
    const @ELTYPE@ go_ge    = static_cast<@ELTYPE@>(-((int)gapo + (int)gape));
    const @ELTYPE@ ge2      = static_cast<@ELTYPE@>(-(int)gape2);
    const @ELTYPE@ go2_ge2  = static_cast<@ELTYPE@>(-((int)gapo2 + (int)gape2));
    (void)ge2; (void)go2_ge2;

    // Per-symbol profile (O(alphabet × phys_w)). Column 0 stays 0 (boundary).
    elt_type *profile = static_cast<elt_type*>(
        kcalloc(km, static_cast<std::size_t>(m) * phys_w, sizeof(elt_type)));
#if VR_DUMMY_PROFILE
    (void)mat; (void)target;   // ABLATION: skip real profile build; zero-fill only
#else
    for (int c = 0; c < m; ++c)
        for (int j = 0; j < tlen; ++j)
            profile[static_cast<std::size_t>(c) * phys_w + (j + 1)] =
                static_cast<elt_type>(mat[c * m + target[j]]);
#endif

    // A row-tiled kernel may evaluate up to RPI-1 padded rows in its last
    // block, and the sweep below calls reset_and_next_row() RPI times after
    // EVERY block, including the last: the ProfileSignal2D anchored at block
    // row RPI-1 then reads sym[] up to index (RPI-1) + RPI*ceil(R/RPI)
    // <= R + 2*RPI - 2. Give it a zero-filled halo of 2*RPI entries; a
    // shorter halo (R+RPI+1) over-read by up to RPI-1 ints and segfaulted
    // whenever the buffer happened to end on a page boundary (seen on macOS).
    int *sym = static_cast<int*>(kcalloc(km, static_cast<std::size_t>(R + 2 * RPI), sizeof(int)));
    for (int i = 0; i < qlen; ++i) sym[i + 1] = static_cast<int>(query[i]);

    // H matrix, TIGHT stride C (padding cols [C,phys_w) are never read).
    const int Hstride = C;
#if VR_DO_HSTORE
    // BAND-PACKED storage (VR_BAND_PACKED, banded only): store just each row's ~2w
    // in-band cells in a packed strip of `band_stride` columns starting at
    // cs_row[i], cutting H alloc/fill/store traffic from O(R*C) to O(R*band_stride).
    // The full-matrix path (banded=false) is unchanged. cs_row[i] is the row's
    // EPI-aligned band start (== the sweep's _cs); out-of-band reads give VR_NEG.
    int band_stride = Hstride;
    int *cs_row = nullptr;
    if (banded && VR_BAND_PACKED) {
        int _bs = 0;   // widest per-block band span (aligned), matching the sweep
        for (int _rb = 0; _rb < N; _rb += RPI * phys_w) {
            int _i   = _rb / phys_w;
            int _cs0 = (std::max(0, _i - w) / EPI) * EPI;
            int _ce0 = std::min(phys_w, (((_i + (RPI - 1) + w + 1) + EPI - 1) / EPI) * EPI);
            if (_ce0 - _cs0 > _bs) _bs = _ce0 - _cs0;
        }
        if (_bs > 0 && _bs < phys_w) {   // real saving only when the band < matrix
            band_stride = _bs;
            cs_row = static_cast<int*>(kcalloc(km, static_cast<std::size_t>(R), sizeof(int)));
        }
    }
    const bool packed = (cs_row != nullptr);
    const int row_stride = packed ? band_stride : Hstride;
    elt_type *Hmat = static_cast<elt_type*>(
        kmalloc(km, static_cast<std::size_t>(R) * row_stride * sizeof(elt_type)));
    // Out-of-band cells must read as tropical -inf for the endpoint scans and
    // traceback; only in-band cells are written by the sweep. Packed fills the
    // strip (R*band_stride); legacy banded fills the full matrix; full-matrix
    // (non-banded) writes every cell so needs no fill.
    if (packed)
        std::fill(Hmat, Hmat + static_cast<std::size_t>(R) * band_stride,
                  static_cast<elt_type>(VR_NEG));
    else if (banded)
        std::fill(Hmat, Hmat + static_cast<std::size_t>(R) * Hstride,
                  static_cast<elt_type>(VR_NEG));
#else
    elt_type *Hmat = nullptr;   // ABLATION: H not stored (timing only)
#endif

    auto kernel = @KVEC@(profile, sym, phys_w, phys_w@CTORARGS@);
    constexpr int vec_lanes = sizeof(@KVECVT@) / sizeof(elt_type);
    VR_ADD(vr_init_ns, _t_init0);
    VR_T0(_t_dp0);

    int gmax = VR_NEG, gmax_i = 0, gmax_j = 0;
    int row_base = 0;
    while (row_base < N) {
        int _log_row_here = row_base / phys_w;
        // Column range for this block-row: full [0,phys_w) unless banded, in which
        // case a two-sided diagonal band [i-w, i+w] aligned to EPI. seek_col()
        // repositions the profile/column state (left carry is NOT primed -- see
        // the banding note above).
        int _cs = 0, _ce = phys_w;
        if (banded) {
            _cs = (std::max(0, _log_row_here - w) / EPI) * EPI;
            _ce = std::min(phys_w,
                           (((_log_row_here + (RPI - 1) + w + 1) + EPI - 1) / EPI) * EPI);
            kernel.seek_col(_cs);
        }
        int col = _cs;
        while (col < _ce) {
            @KVECVT@ out_buf[RPI][EPI / vec_lanes];
            @KVECVT@* out_per_row[RPI] = { @COUT@ };
            kernel.run(out_per_row);
            for (int rr = 0; rr < RPI; rr++) {
                int log_r = _log_row_here + rr;
                if (log_r >= R) break;
                int copy_b = std::min(EPI, C - col);   // real columns only
                if (copy_b <= 0) continue;
                const elt_type* chunk = reinterpret_cast<const elt_type*>(out_buf[rr]);
#if VR_DO_HSTORE
                if (packed) {
                    cs_row[log_r] = _cs;   // band start for this row (== block _cs)
                    assert((col - _cs) + copy_b <= band_stride);
                    std::memcpy(Hmat + (static_cast<std::size_t>(log_r) * band_stride + (col - _cs)),
                                chunk, sizeof(elt_type) * copy_b);
                } else {
                    std::memcpy(Hmat + (static_cast<std::size_t>(log_r) * Hstride + col),
                                chunk, sizeof(elt_type) * copy_b);
                }
#endif
                // Fused global-max: vectorized horizontal max, scalar locate only
                // when a new maximum actually appears (padding cols j>=C skipped).
                @KVECVT@ vmax = out_buf[rr][0];
                for (int s = 1; s < EPI / vec_lanes; ++s)
                    vmax = __builtin_elementwise_max(vmax, out_buf[rr][s]);
                elt_type cmax = __builtin_reduce_max(vmax);
                if (cmax > gmax) {
                    for (int t = 0; t < copy_b; ++t) {
                        int j = col + t;
                        if (j >= C) break;
                        if (chunk[t] > gmax) { gmax = chunk[t]; gmax_i = log_r; gmax_j = j; }
                    }
                }
            }
            col += EPI;
        }
        for (int rr = 0; rr < RPI; rr++) kernel.reset_and_next_row();
        row_base += RPI * phys_w;
    }
    VR_ADD(vr_dp_ns, _t_dp0);
    VR_T0(_t_tb0);

#if VR_DO_HSTORE
    auto H_get = [&](int i, int j) -> int {
        if (packed) {
            int off = j - cs_row[i];
            if (off < 0 || off >= band_stride) return VR_NEG;
            return static_cast<int>(Hmat[static_cast<std::size_t>(i) * band_stride + off]);
        }
        return static_cast<int>(Hmat[static_cast<std::size_t>(i) * Hstride + j]);
    };
#endif

    // mqe/mte from the last row and last column (O(C + R)). gmax found inline.
    int mqe = VR_NEG, mqe_t = 0;                 // query exhausted (last row)
    int mte = VR_NEG, mte_q = 0;                 // target exhausted (last col)
#if VR_DO_ENDPOINT
    for (int j = 0; j < C; ++j) { int v = H_get(qlen, j); if (v > mqe) { mqe = v; mqe_t = j; } }
    for (int i = 0; i < R; ++i) { int v = H_get(i, tlen); if (v > mte) { mte = v; mte_q = i; } }
#endif
    if (gmax <= VR_NEG) { gmax = 0; gmax_i = 0; gmax_j = 0; }

#if VR_CHECK && VR_DO_HSTORE
    // VR_CHECK validates the EXACT kernel against a full-matrix scalar reference.
    // The banded path is approximate by design (seek_col does not prime the
    // diagonal carry at the left band edge), and band-packed storage has no full
    // matrix to compare, so the check only runs on the full-matrix path. Band
    // storage correctness is covered by the VR_BAND_PACKED=0/1 A/B (identical
    // alignments) instead.
    if (!packed && !banded)
        vr_reference_check<PIECES>(qlen, query, tlen, target, m, mat,
                                  (int)gapo, (int)gape, (int)gapo2, (int)gape2,
                                  Hmat, Hstride, gmax);
#endif

    uint8_t *ops = static_cast<uint8_t*>(
        kmalloc(km, static_cast<std::size_t>(qlen + tlen + 1)));
    int ti, tj, reach_end;
    if (!(flag & KSW_EZ_EXTZ_ONLY)) {
        reach_end = 0; ti = qlen; tj = tlen;
    } else if (mqe > VR_NEG && mqe + end_bonus > gmax) {
        reach_end = 1; ti = qlen; tj = mqe_t;
    } else {
        reach_end = 0; ti = gmax_i; tj = gmax_j;
    }
    int nops;
#if VR_DO_TRACEBACK
    if (flag & KSW_EZ_RIGHT)
        nops = vecrec_traceback<PIECES>(Hmat, Hstride, query, target, m, mat,
                                ge, go_ge, ge2, go2_ge2, ti, tj, ops, /*right_align=*/true,
                                packed ? cs_row : nullptr, band_stride, VR_NEG);
    else
        nops = vecrec_traceback<PIECES>(Hmat, Hstride, query, target, m, mat,
                                ge, go_ge, ge2, go2_ge2, ti, tj, ops, /*right_align=*/false,
                                packed ? cs_row : nullptr, band_stride, VR_NEG);
#else
    nops = vr_trivial_cigar(ti, tj, ops);   // ABLATION: meaningless span (timing only)
#endif

    const int seq_max_q = gmax_i > 0 ? gmax_i - 1 : 0;
    const int seq_max_t = gmax_j > 0 ? gmax_j - 1 : 0;

#if VR_DO_HSTORE
    ez->score = (flag & KSW_EZ_EXTZ_ONLY) ? gmax : H_get(qlen, tlen);
#else
    ez->score = gmax;   // corner unavailable without H store (timing only)
#endif
    ez->max   = static_cast<uint32_t>(gmax > 0 ? (gmax & 0x7fffffff) : 0);
    ez->max_q = seq_max_q;
    ez->max_t = seq_max_t;
    ez->mqe   = (mqe <= VR_NEG) ? KSW_NEG_INF : mqe;
    ez->mqe_t = (mqe <= VR_NEG) ? -1 : (mqe_t > 0 ? mqe_t - 1 : 0);
    ez->mte   = (mte <= VR_NEG) ? KSW_NEG_INF : mte;
    ez->mte_q = (mte <= VR_NEG) ? -1 : (mte_q > 0 ? mte_q - 1 : 0);
    ez->reach_end = reach_end;
    ez->zdropped  = 0;

#ifdef VR_DEBUG_CALLS
    static int _vrdbg = -1;
    if (_vrdbg < 0) _vrdbg = std::getenv("VECREC_DEBUG_CALLS") ? 1 : 0;
    if (_vrdbg) {
        std::fprintf(stderr, "VRDBG pieces=%d qlen=%d tlen=%d score=%d gmax=%d\n",
                     PIECES, qlen, tlen, (int)ez->score, gmax);
    }
#endif

    // ops[] holds the traceback in end-to-start order. Like ksw_backtrack(),
    // emit it reversed (start-to-end) unless the caller asked for a reversed
    // CIGAR: minimap2's left extension aligns reversed sequences and passes
    // KSW_EZ_REV_CIGAR so the CIGAR comes back in forward-strand order.
    ez->n_cigar = 0;
    if (flag & KSW_EZ_REV_CIGAR) {
        for (int k = 0; k < nops; ++k)
            ez->cigar = ksw_push_cigar(km, &ez->n_cigar, &ez->m_cigar, ez->cigar, ops[k], 1);
    } else {
        for (int k = nops - 1; k >= 0; --k)
            ez->cigar = ksw_push_cigar(km, &ez->n_cigar, &ez->m_cigar, ez->cigar, ops[k], 1);
    }

    kfree(km, ops);
#if VR_DO_HSTORE
    kfree(km, Hmat);
    if (cs_row) kfree(km, cs_row);
#endif
    kfree(km, sym);
    kfree(km, profile);
    VR_ADD(vr_tb_ns, _t_tb0);
    VR_CALLS();
}
"""


def _make_core(name, kvec, epi, rpi, pieces, ctor_args, elt_ctype, neg):
    cout = ", ".join(f"out_buf[{r}]" for r in range(rpi))
    return (CORE_TEMPLATE
            .replace("@NM@", name)
            .replace("@KVECVT@", f"{kvec}_vector_type")
            .replace("@KVEC@", kvec)
            .replace("@CEPI@", str(epi))
            .replace("@CRPI@", str(rpi))
            .replace("@PIECES@", str(pieces))
            .replace("@CTORARGS@", ctor_args)
            .replace("@COUT@", cout)
            .replace("@ELTYPE@", elt_ctype)
            .replace("@VRNEG@", neg))


def _neg(ctype: str) -> str:
    # Must match the kernel-internal tropical "-infinity" sentinel that the pinned
    # vec-rec emits (val_to_str for -inf; the Sat-variant zero_init in
    # templates/common.h). Taken from vec-rec itself so the two cannot drift.
    from vecrec.expr.base import Type
    from vecrec.util import ElementType
    elt = ElementType.I8 if ctype == "int8_t" else ElementType.I16
    return elt.val_to_str(float("-inf"), Type.TropMax)


# One core per (piece-count, width). The public entry points dispatch to the
# matching *_i8 / *_i16 core by max achievable score (see VR_I8_MAX).
_CORE_SPECS = [
    ("vecrec_ext_core_2p_i16", "k_vr_2p_i16", 2, ", ge, go_ge, ge2, go2_ge2"),
    ("vecrec_ext_core_2p_i8",  "k_vr_2p_i8",  2, ", ge, go_ge, ge2, go2_ge2"),
    ("vecrec_ext_core_1p_i16", "k_vr_1p_i16", 1, ", ge, go_ge"),
    ("vecrec_ext_core_1p_i8",  "k_vr_1p_i8",  1, ", ge, go_ge"),
]
_cores = []
for (_cname, _kname, _pieces, _ctorargs) in _CORE_SPECS:
    _mk = KMETA[_kname]
    _cores.append(_make_core(_cname, _kname, _mk["epi"], _mk["rpi"], _pieces,
                             _ctorargs, _mk["ctype"], _neg(_mk["ctype"])))

SRC = (SRC
       .replace("@CORES@", "\n".join(_cores))
       .replace("@I8_1P_GE_MULT@", str(max(
           KMETA["k_vr_1p_i8"]["gap_multipliers"]["ge"],
           KMETA["k_vr_2p_i8"]["gap_multipliers"]["ge"],
           KMETA["k_vr_2p_i8"]["gap_multipliers"]["ge2"])))
       .replace("@I8_1P_GO_GE_MULT@", str(max(
           KMETA["k_vr_1p_i8"]["gap_multipliers"]["go_ge"],
           KMETA["k_vr_2p_i8"]["gap_multipliers"]["go_ge"],
           KMETA["k_vr_2p_i8"]["gap_multipliers"]["go2_ge2"])))
       .replace("@I8_2P_GE_MULT@", str(KMETA["k_vr_2p_i8"]["gap_multipliers"]["ge"]))
       .replace("@I8_2P_GO_GE_MULT@", str(KMETA["k_vr_2p_i8"]["gap_multipliers"]["go_ge"]))
       .replace("@I8_2P_GE2_MULT@", str(KMETA["k_vr_2p_i8"]["gap_multipliers"]["ge2"]))
       .replace("@I8_2P_GO2_GE2_MULT@", str(KMETA["k_vr_2p_i8"]["gap_multipliers"]["go2_ge2"]))
       .replace("@SCHED1@", str(KMETA["k_vr_1p_i16"]["sched"]))
       .replace("@EPI1@", str(KMETA["k_vr_1p_i16"]["epi"]))
       .replace("@RPI1@", str(KMETA["k_vr_1p_i16"]["rpi"]))
       .replace("@SCHED@", str(KMETA["k_vr_2p_i16"]["sched"]))
       .replace("@EPI@", str(KMETA["k_vr_2p_i16"]["epi"]))
       .replace("@RPI@", str(KMETA["k_vr_2p_i16"]["rpi"]))
       .replace("@KSW2_H@", ksw2_h_rel))

pathlib.Path(src_path).write_text(SRC)
meta_path = out_dir / "vecrec_schedule_search.json"
meta_path.write_text(json.dumps({
    "max_bits": max_bits,
    "kernels": {label: {**selection[3],
                        "element_per_iter": KMETA[f"k_vr_{label}"]["epi"],
                        "rows_per_iter": KMETA[f"k_vr_{label}"]["rpi"],
                        "gap_multipliers": KMETA[f"k_vr_{label}"]["gap_multipliers"]}
                for label, selection in SELECTIONS.items()},
}, indent=2) + "\n")

print("Generated:")
print(f"  {hdr_path}")
print(f"  {src_path}")
print(f"  {meta_path}")
print(f"  dual-width kernels (max_bits={max_bits}, I8 dispatch when max score <= VR_I8_MAX):")
for _kname, _m in KMETA.items():
    print(f"    {_kname:12s} sched={_m['sched']:2d} epi={_m['epi']:2d} rpi={_m['rpi']} ({_m['ctype']})")
