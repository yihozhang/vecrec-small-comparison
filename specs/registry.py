"""Registry of recurrence specs to benchmark.

Each entry maps a benchmark name to the C++ kernel/struct name and a builder that
returns the *unscheduled* VecRec expression. ``scripts/gen_kernels.py`` feeds the
expression to ``vecrec.codegen.find_best_schedule_and_emit_header`` to find the best
schedule and emit ``generated/<kernel_name>.h``.

Expressions are reconstructed inline (with a pointer to the upstream spec under
``vec-rec/benchmarks/*`` where one exists) so this project depends only on the
``vecrec`` library, not on vec-rec's Python benchmark harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from vecrec.expr import (
    Convolve,
    Convolve2D,
    Impulse,
    KernelMatrix,
    Num,
    PointwiseDiv,
    Recurse,
    Recurse2D,
    RecurseV,
    SignalVector,
    TIKernel,
    TIKernel2D,
    TVKernel,
    TVKernel2D,
    Type,
    Var,
    Var2D,
    VectorAt,
    with_types,
)
from vecrec.expr.base import SignalExpr, SignalExpr2D
from vecrec.expr.signal import ComputedSignal2D
from vecrec.expr.signal_ops import Ith, PointwiseMul, SSub
from vecrec.semiring import CustomSemiring
from vecrec.util import ElementType

INF = float("inf")

# Boolean semiring: OR = plus, AND = times (on {0,1}: OR=max, AND=min). Idempotent (OR), so the
# tropical-style dilation applies. Distinct from the built-in semirings (times is min, not +).
BoolOrAnd = CustomSemiring(
    name="BoolOrAnd",
    zero=0.0,   # False (identity for OR)
    one=1.0,    # True  (identity for AND)
    plus=lambda a, b: max(a, b),   # OR
    times=lambda a, b: min(a, b),  # AND
    is_zero_fn=lambda v: abs(v) < 1e-6,
    is_one_fn=lambda v: abs(v - 1.0) < 1e-6,
    cpp_zero_init="(elt_type)0",
    cpp_mul_op="[](auto a, auto b) { return __builtin_elementwise_min(a, b); }",
    cpp_add_op="[](auto acc, auto x) { return __builtin_elementwise_max(acc, x); }",
    cpp_sadd_op="[](auto left, auto right) { return __builtin_elementwise_max(left, right); }",
    cpp_pmul_op="[](auto left, auto right) { return __builtin_elementwise_min(left, right); }",
    add_idempotent=True,
)


@dataclass(frozen=True)
class Spec:
    name: str          # benchmark name / slug (hyphenated) — dict key, CLI arg, CSV key
    kernel_name: str   # C++ struct / header name (underscores)
    dim: str           # "1d" or "2d" — informational
    semiring: str      # "arith" | "tropmin" | "tropmax" — informational
    build_expr: Optional[Callable[[], "SignalExpr | SignalExpr2D"]]
    display_name: str = ""  # human-readable name for display (e.g. "Exponential Smoothing")
    description: str = ""
    note: str = ""     # e.g. why a benchmark is unsupported


# ── 1D arithmetic IIR filters ─────────────────────────────────────────────────

def _exponential_smoothing(a: float = 0.8) -> SignalExpr:
    # f(x) = a*f(x-1) + (1-a)*g(x)
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(TIKernel([0, a]), Convolve(TIKernel([1 - a]), Var("g")))


def _double_exp_smoothing(alpha: float = 0.5, beta: float = 0.3) -> SignalExpr:
    # Holt's linear (double) exponential smoothing, rewritten so the trend b_t no
    # longer depends on the *current* level l_t (VecRec recurrence kernels only allow
    # state at t to depend on state at t-1):
    #   l_t = (1-a) l_{t-1} + (1-a) b_{t-1} + a    g_t
    #   b_t = -a*B l_{t-1} + (1-a*B) b_{t-1} + a*B g_t
    # Encoded as a 2x2 mutual recurrence (precedent: arith/mutual_recurse.py).
    # Output = the level l_t. Coupled kernel is stable for these (alpha, beta).
    with with_types(Type.Arith, ElementType.Float):
        oma = 1.0 - alpha          # (1 - alpha)
        ab = alpha * beta          # alpha*beta
        A = KernelMatrix(
            [
                [TIKernel([0.0, oma]), TIKernel([0.0, oma])],
                [TIKernel([0.0, -ab]), TIKernel([0.0, 1.0 - ab])],
            ]
        )
        G = SignalVector(
            [
                Convolve(TIKernel([alpha]), Var("g")),  # a*g_t  -> level
                Convolve(TIKernel([ab]), Var("g")),     # a*B*g_t -> trend
            ]
        )
        F = RecurseV(A, G)
        return VectorAt(F, 0)  # output the smoothed level l_t


def _first_order_high_pass(a: float = 0.8) -> SignalExpr:
    # vec-rec/benchmarks/arith/first_order_high_pass.py
    # f(x) = a*(f(x-1) + g(x) - g(x-1))
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(TIKernel([0, a]), Convolve(TIKernel([a, -a]), Var("g")))


def _second_order_high_pass() -> SignalExpr:
    # vec-rec/benchmarks/arith/second_order_high_pass.py
    # f(x) = 1.6*f(x-1) - 0.64*f(x-2) + g(x) - 2*g(x-1) + g(x-2)
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(TIKernel([0, 1.6, -0.64]), Convolve(TIKernel([1, -2, 1]), Var("g")))


def _second_order_low_pass() -> SignalExpr:
    # vec-rec/benchmarks/arith/second_order_ti.py
    # f(x) = 1.8*f(x-1) - 0.9*f(x-2) + g(x)
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(TIKernel([0, 1.8, -0.9]), Var("g"))


# Recursive Gaussian (Young & van Vliet 1995, causal), sigma=2.0.
# vec-rec/benchmarks/arith/recursive_gaussian.py
_SIGMA = 2.0
_q = 1.31564 * (math.sqrt(1.0 + 0.490811 * _SIGMA * _SIGMA) - 1.0)  # sigma < 2.5
_b0 = 1.57825 + 2.44413 * _q + 1.4281 * _q**2 + 0.422205 * _q**3
_b1 = 2.44413 * _q + 2.85619 * _q**2 + 1.26661 * _q**3
_b2 = -1.4281 * _q**2 - 1.26661 * _q**3
_b3 = 0.422205 * _q**3
GAUSS_A1 = _b1 / _b0
GAUSS_A2 = _b2 / _b0
GAUSS_A3 = _b3 / _b0
GAUSS_B = 1.0 - (GAUSS_A1 + GAUSS_A2 + GAUSS_A3)


def _recursive_gaussian() -> SignalExpr:
    # f(x) = a1*f(x-1) + a2*f(x-2) + a3*f(x-3) + B*g(x)
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(
            TIKernel([0, GAUSS_A1, GAUSS_A2, GAUSS_A3]),
            Convolve(TIKernel([GAUSS_B]), Var("g")),
        )


# ── 1D time-varying (non-uniform) recurrences ─────────────────────────────────

def _nonuniform_recursive_filter() -> SignalExpr:
    # vec-rec/benchmarks/arith/second_order_tv.py
    # f(x) = a(x)*f(x-1) + b(x)*f(x-2) + g(x)  — coefficients vary per sample.
    # Driven by an input stream g so it is a genuine (BIBO-stable) filter; the
    # benchmark keeps |a|+|b| < 1 so the time-varying recurrence stays bounded.
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(TVKernel([Num(0.0), Var("a"), Var("b")]), Var("g"))


# w(x) = 1/x is a pure function of stream position, so instead of feeding it as a
# materialized input array we compute it on the fly with a ComputedSignal2D whose
# C++ struct tracks the column position and emits a fully-vectorized reciprocal
# (one vdivps/vdivpd per run() — no memory traffic for w). Instantiated by codegen
# as WelfordWeight2D<vec_type, n_rows, block_size, block_row>; used via Ith(_,0) as
# a 1D signal. x is 1-based (element i => x = i+1).
WELFORD_W_PRELUDE = r"""
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct WelfordWeight2D {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    int my_col;
    WelfordWeight2D() : my_col(0) {}
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        vec_type wv;
        if constexpr (lanes == 1) {
            wv = (elt_type)1 / (elt_type)(my_col + 1);
        } else {
            vec_type posv;
            _Pragma("unroll") for (int k = 0; k < lanes; k++) posv[k] = (elt_type)(my_col + 1 + k);
            wv = (elt_type)1 / posv;   // fully-vectorized reciprocal
        }
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) reg[base + i] = wv;
        my_col += lanes;
    }
    void reset_and_next_row() { my_col = 0; }
};
"""


def _welford_mean() -> SignalExpr:
    # Welford running mean: M(x) = M(x-1) + (a(x) - M(x-1)) / x
    #                            = (1 - w(x))*M(x-1) + w(x)*a(x),  w(x) = 1/x.
    with with_types(Type.Arith, ElementType.Float):
        w = Ith(ComputedSignal2D("w", cpp_struct="WelfordWeight2D",
                                 prelude=WELFORD_W_PRELUDE, setup_code="", make_args=""), 0)
        a = Var("a")
        return Recurse(TVKernel([Num(0.0), SSub(Num(1.0), w)]), PointwiseMul(w, a))


# Continued fraction via the Wallis–Euler recurrence:
#   h(x) = a(x)*h(x-1) + b(x)*h(x-2),   f(x) = h(x)/h(x-1)
# (skipping the final division would leave a rational approximant; see
# vec-rec/benchmarks/arith/continued_fraction.py).
#
# h grows/decays geometrically, so a single long stream overflows within ~100
# steps. We instead run many *independent* continued fractions of CF_SEGMENT
# steps each: one CF per harness "row" (row_size = CF_SEGMENT), relying on the
# recurrence state resetting at each row boundary. A seed input s injects h=1 at
# each segment's first step (after reset the history is zero, so h(0)=s(0)).
# Emitted in **double** (float64) so h stays finite across a segment; VecRec
# dilates this genuinely lag-1 recurrence to vectorize within each segment.
CF_SEGMENT = 512      # steps per continued fraction (== harness row_size)


def _continued_fraction() -> SignalExpr:
    with with_types(Type.Arith, ElementType.Double):
        # h(x) = a(x)*h(x-1) + b(x)*h(x-2) + s(x)
        h = Recurse(TVKernel([Num(0.0), Var("a"), Var("b")]), Var("s"))
        return PointwiseDiv(h, Convolve(TIKernel.z(), h))  # f = h(x)/h(x-1)


# ── 2D arithmetic ─────────────────────────────────────────────────────────────

def _box_filter_sat(D: int = 11) -> SignalExpr:
    # vec-rec/benchmarks/arith/box_filter_summed_area_table.py
    with with_types(Type.Arith, ElementType.Float):
        kernel = [[0.0] * D for _ in range(D)]
        kernel[0][0] = kernel[D - 1][D - 1] = 1.0
        kernel[D - 1][0] = kernel[0][D - 1] = -1.0
        fir = Convolve2D(TIKernel2D(kernel), Var2D("g"))
        sat_kernel = TIKernel2D([[0.0, 1.0], [1.0, -1.0]])
        return Recurse2D(sat_kernel, fir).at(0)


def _repeated_box_filter(D: int = 7) -> SignalExpr:
    # vec-rec/benchmarks/arith/repeated_box_filter.py — 3 cascaded SAT box passes.
    # Each pass divides by D*D, so it is a moving *average* (values stay bounded).
    with with_types(Type.Arith, ElementType.Float):
        def one_pass(sig: SignalExpr2D) -> SignalExpr2D:
            k = [[0.0] * D for _ in range(D)]
            k[0][0] = k[D - 1][D - 1] = 1.0
            k[D - 1][0] = k[0][D - 1] = -1.0
            df = Convolve2D(TIKernel2D(k), sig)
            df2 = PointwiseDiv(df, Num(float(D * D)))
            return Recurse2D(TIKernel2D([[0.0, 1.0], [1.0, -1.0]]), df2)

        b1 = one_pass(Var2D("g"))
        b2 = one_pass(b1)
        b3 = one_pass(b2)
        return b3.at(0)


# ── LeetCode dynamic-programming recurrences ──────────────────────────────────
# Node-weighted DPs carry the per-step reward/cost on the time-varying feedback taps;
# a seed input carries the base cases (the semiring zero elsewhere).

def _decode_ways() -> SignalExpr:
    # LeetCode 91: f(i) = a(i)*f(i-1) + b(i)*f(i-2), a(i),b(i) in {0,1} (valid single / pair).
    # Counts grow ~Fibonacci, so run short independent strings (one per row, reset per row);
    # seed s injects the f=1 empty-prefix base at each segment start.
    with with_types(Type.Arith, ElementType.I64):
        return Recurse(TVKernel([Num(0.0), Var("a"), Var("b")]), Var("s"))


# 1D ring counting recurrences (segmented per row to bound the geometric growth; int64).

def _count_good_strings(zero: int = 1, one: int = 3) -> SignalExpr:
    # LeetCode 2466: f(i) = f(i-zero) + f(i-one), seeded by s.
    with with_types(Type.Arith, ElementType.I64):
        data = [0.0] * (max(zero, one) + 1)
        data[zero] = 1.0
        data[one] = 1.0
        return Recurse(TIKernel(data), Var("s"))


def _min_cost_climbing_stairs() -> SignalExpr:
    # LeetCode 746: f(i) = cost[i] + min(f(i-1), f(i-2)).
    # (min, +): cost on both taps; seed s carries f(0), f(1).
    with with_types(Type.TropMin, ElementType.Double):
        c = Var("c")
        return Recurse(TVKernel([Num(Type.TropMin.zero()), c, c]), Var("s"))


def _house_robber() -> SignalExpr:
    # LeetCode 198: f(i) = max(f(i-1), f(i-2) + nums[i]).
    # (max, +): +0 tap on f(i-1), +nums tap on f(i-2); seed s carries the base cases.
    with with_types(Type.TropMax, ElementType.Double):
        return Recurse(TVKernel([Num(Type.TropMax.zero()), Num(0.0), Var("n")]), Var("s"))


def _delete_and_earn() -> SignalExpr:
    # LeetCode 740: dp(v) = max(dp(v-1), dp(v-2) + total[v]), total[v] = v * count(v).
    # House Robber over the value axis (same recurrence as 198).
    with with_types(Type.TropMax, ElementType.Double):
        return Recurse(TVKernel([Num(Type.TropMax.zero()), Num(0.0), Var("t")]), Var("s"))


def _max_energy_two_drinks() -> SignalExpr:
    # LeetCode 3259: dp_A[i] = A[i] + max(dp_A[i-1], dp_B[i-2]); dp_B symmetric.
    # (max, +) 2x2 mutual: reward on the taps (stay = lag-1, switch = lag-2 cross). Output dp_A.
    with with_types(Type.TropMax, ElementType.Double):
        a, b = Var("a"), Var("b")
        z = Num(Type.TropMax.zero())
        A = KernelMatrix(
            [
                [TVKernel([z, a, z]), TVKernel([z, z, a])],
                [TVKernel([z, z, b]), TVKernel([z, b, z])],
            ]
        )
        G = SignalVector([Var("sa"), Var("sb")])
        return VectorAt(RecurseV(A, G), 0)


def _range_sum_2d() -> SignalExpr:
    # LeetCode 304: 2D prefix sum (summed-area table)
    # f[i][j] = m[i][j] + f[i-1][j] + f[i][j-1] - f[i-1][j-1].
    with with_types(Type.Arith, ElementType.Float):
        return Recurse2D(TIKernel2D([[0.0, 1.0], [1.0, -1.0]]), Var2D("m").at(0)).at(0)


def _is_subsequence() -> SignalExpr:
    # LeetCode 392: is s a subsequence of t? f(i,j) = f(i,j-1) OR (s[i-1]==t[j-1] AND f(i-1,j-1)).
    # Boolean OR/AND, 2D: left tap passes through (coeff one), diagonal tap is the match mask; seeded
    # true at (0,0).
    with with_types(BoolOrAnd, ElementType.I32):
        z = Num(0.0)
        m = Var2D("m").at(0)   # diagonal: s[i-1] == t[j-1]
        kernel = TVKernel2D([[z, Num(1.0)], [z, m]])
        return Recurse2D(kernel, Var2D("s").at(0)).at(0)


def _lcs() -> SignalExpr:
    # LeetCode 1143 (also 1035; 516/3503 via a reversed axis). LCS max-form, a 2D (max,+) recurrence
    # dp(i,j) = max(dp(i-1,j), dp(i,j-1), dp(i-1,j-1) + m(i,j)); match reward m on the diagonal tap.
    with with_types(Type.TropMax, ElementType.I32):
        z = Num(Type.TropMax.zero())
        m = Var2D("m").at(0)
        kernel = TVKernel2D(
            [
                [z, Num(0.0)],        # same row:  dp(i, j-1) + 0
                [Num(0.0), m],        # row back:  dp(i-1, j) + 0,  dp(i-1, j-1) + m(i,j)
            ]
        )
        return Recurse2D(kernel, Impulse(0)).at(0)


def _max_dot_product() -> SignalExpr:
    # LeetCode 1458: dp(i,j) = max(dp(i-1,j), dp(i,j-1), dp(i-1,j-1) + p, p), p = nums1[i]*nums2[j].
    # LCS-shaped; the product p is both the diagonal tap and the standalone ⊕ input.
    with with_types(Type.TropMax, ElementType.I32):
        z = Num(Type.TropMax.zero())
        p = Var2D("p").at(0)
        kernel = TVKernel2D([[z, Num(0.0)], [Num(0.0), p]])
        return Recurse2D(kernel, Var2D("p").at(0)).at(0)


def _edit_distance() -> SignalExpr:
    # LeetCode 72 (also 583; 1312 via a reversed axis). Levenshtein, a 2D (min,+) recurrence
    # D(i,j) = min(D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1) + c(i,j)); c = per-cell substitution cost.
    with with_types(Type.TropMin, ElementType.I32):
        posinf = Num(Type.TropMin.zero())
        gap = Num(1.0)
        c = Var2D("c").at(0)
        kernel = TVKernel2D([[posinf, gap], [gap, c]])
        return Recurse2D(kernel, Impulse(0)).at(0)


def _maximal_square() -> SignalExpr:
    # LeetCode 221 (also 1277): side of the largest all-ones square ending at (i,j):
    #   dp(i,j) = min(dp(i-1,j), dp(i,j-1), dp(i-1,j-1)) + 1  if grid==1 else 0.
    # 2D TropMin with constant +1 taps (incl. the diagonal), like distance_transform. The mask is
    # the ⊕ input g: 0 where grid==0 (forces dp=0), 1 on a grid==1 boundary cell (a single-cell
    # square), and a large "+inf" sentinel on grid==1 interior cells (so the min(neighbors)+1 wins).
    with with_types(Type.TropMin, ElementType.Float):
        k = TIKernel2D([[INF, 1.0], [1.0, 1.0]])
        return Recurse2D(k, Var2D("g").at(0)).at(0)


# ── 2D tropical (min, +) ──────────────────────────────────────────────────────

_DT_KERNEL = [
    [INF, 1.0],
    [1.0, math.sqrt(2), math.sqrt(5)],
    [INF, math.sqrt(5)],
]


def _distance_transform() -> SignalExpr:
    # vec-rec/benchmarks/trop/distance_transform.py
    with with_types(Type.TropMin, ElementType.Float):
        g = Var2D("g")
        return Recurse2D(TIKernel2D(_DT_KERNEL), g.at(0)).at(0)


# ── Registry ──────────────────────────────────────────────────────────────────
SPECS: dict[str, Spec] = {
    s.name: s
    for s in [
        # ── Original repo benchmarks (signal-processing filters + reference DPs) ──
        Spec("exponential-smoothing", "exponential_smoothing", "1d", "arith",
             _exponential_smoothing, display_name="Exponential Smoothing",
             description="f(x)=0.8*f(x-1)+0.2*g(x)"),
        Spec("double-exp-smoothing", "double_exp_smoothing", "1d", "arith",
             _double_exp_smoothing, display_name="Double Exponential Smoothing",
             description="Holt's double exponential smoothing (level+trend) as a 2x2 mutual recurrence"),
        Spec("box-filter-sat", "box_filter_sat", "2d", "arith",
             _box_filter_sat, display_name="Box Filter (Summed Area Table)",
             description="Box filter (D=11) via 2D difference kernel + summed area table"),
        Spec("repeated-box-filter", "repeated_box_filter", "2d", "arith",
             _repeated_box_filter, display_name="Repeated Box Filter",
             description="3-pass box filter (D=7) approximating a Gaussian"),
        Spec("first-order-high-pass", "first_order_high_pass", "1d", "arith",
             _first_order_high_pass, display_name="First-Order High Pass Filter",
             description="f(x)=0.8*(f(x-1)+g(x)-g(x-1))"),
        Spec("recursive-gaussian", "recursive_gaussian", "1d", "arith",
             _recursive_gaussian, display_name="Recursive Gaussian Filter",
             description="Causal recursive Gaussian (Young & van Vliet, sigma=2): 3rd-order IIR"),
        Spec("second-order-high-pass", "second_order_high_pass", "1d", "arith",
             _second_order_high_pass, display_name="Second-Order High Pass Filter",
             description="f(x)=1.6*f(x-1)-0.64*f(x-2)+g(x)-2*g(x-1)+g(x-2)"),
        Spec("second-order-low-pass", "second_order_low_pass", "1d", "arith",
             _second_order_low_pass, display_name="Second-Order Low Pass Filter",
             description="f(x)=1.8*f(x-1)-0.9*f(x-2)+g(x)"),
        Spec("distance-transform", "distance_transform", "2d", "tropmin",
             _distance_transform, display_name="Distance Transform",
             description="2D distance transform (TropMin) with knight-move kernel"),
        Spec("nonuniform-recursive-filter", "nonuniform_recursive_filter", "1d", "arith",
             _nonuniform_recursive_filter, display_name="Non-uniform Recursive Filter",
             description="Time-varying 2nd-order IIR: f(x)=a(x)*f(x-1)+b(x)*f(x-2)+g(x)"),
        Spec("welford-mean", "welford_mean", "1d", "arith",
             _welford_mean, display_name="Welford Mean",
             description="Welford running mean M(x)=M(x-1)+(a(x)-M(x-1))/x as a 1st-order TV recurrence"),
        Spec("continued-fraction", "continued_fraction", "1d", "arith",
             _continued_fraction, display_name="Continued Fraction",
             description="Wallis-Euler continued fraction h(x)=a(x)h(x-1)+b(x)h(x-2), f=h(x)/h(x-1); "
                         "batched 512-step segments in float64 (see registry note)"),

        # ── LeetCode 2D sequence-alignment DPs (TropMax / TropMin) ──
        Spec("leetcode-1143-lcs", "leetcode_1143_lcs", "2d", "tropmax",
             _lcs, display_name="Longest Common Subsequence (LeetCode 1143)",
             description="dp(i,j)=max(dp(i-1,j),dp(i,j-1),dp(i-1,j-1)+match); 2D TropMax alignment"),
        Spec("leetcode-1035-uncrossed-lines", "leetcode_1035_uncrossed_lines", "2d", "tropmax",
             _lcs, display_name="Uncrossed Lines (LeetCode 1035)",
             description="Identical recurrence to LCS on two integer arrays; 2D TropMax alignment"),
        Spec("leetcode-1092-shortest-common-supersequence", "leetcode_1092_shortest_common_supersequence", "2d", "tropmax",
             _lcs, display_name="Shortest Common Supersequence (LeetCode 1092)",
             description="length = len(a)+len(b)-LCS(a,b); the DP is the LCS recurrence; 2D TropMax alignment"),
        Spec("leetcode-1216-valid-palindrome-iii", "leetcode_1216_valid_palindrome_iii", "2d", "tropmax",
             _lcs, display_name="Valid Palindrome III (LeetCode 1216)",
             description="k>=n-LPS(s); LPS = LCS-form over s vs its reverse (diagonal reward 2); 2D TropMax"),
        Spec("leetcode-516-longest-palindromic-subseq", "leetcode_516_longest_palindromic_subseq", "2d", "tropmax",
             _lcs, display_name="Longest Palindromic Subsequence (LeetCode 516)",
             description="LPS = LCS-form over one string vs its reverse (diagonal reward 2 on match); 2D TropMax"),
        Spec("leetcode-3503-longest-palindrome-concat", "leetcode_3503_longest_palindrome_concat", "2d", "tropmax",
             _lcs, display_name="Longest Palindrome After Concatenation (LeetCode 3503)",
             description="LPS-form (reward 2 on match) over a combined string; 2D TropMax alignment"),
        Spec("leetcode-1458-max-dot-product", "leetcode_1458_max_dot_product", "2d", "tropmax",
             _max_dot_product, display_name="Max Dot Product of Two Subsequences (LeetCode 1458)",
             description="dp(i,j)=max(dp(i-1,j),dp(i,j-1),dp(i-1,j-1)+p,p), p=nums1[i]*nums2[j]; 2D TropMax"),
        Spec("leetcode-72-edit-distance", "leetcode_72_edit_distance", "2d", "tropmin",
             _edit_distance, display_name="Edit Distance (LeetCode 72)",
             description="Levenshtein D(i,j)=min(D(i-1,j)+1,D(i,j-1)+1,D(i-1,j-1)+cost); 2D TropMin alignment"),
        Spec("leetcode-583-delete-two-strings", "leetcode_583_delete_two_strings", "2d", "tropmin",
             _edit_distance, display_name="Delete Operation for Two Strings (LeetCode 583)",
             description="Edit distance with insert/delete only (substitution forbidden); 2D TropMin alignment"),
        Spec("leetcode-1312-min-insertion-palindrome", "leetcode_1312_min_insertion_palindrome", "2d", "tropmin",
             _edit_distance, display_name="Minimum Insertion Steps to Make Palindrome (LeetCode 1312)",
             description="Min insertions = edit-distance form over one string vs its reverse; 2D TropMin"),

        # ── LeetCode 2D arithmetic (summed-area table) ──
        Spec("leetcode-304-range-sum-2d", "leetcode_304_range_sum_2d", "2d", "arith",
             _range_sum_2d, display_name="Range Sum Query 2D (LeetCode 304)",
             description="2D prefix sum / summed-area table: f[i][j]=m[i][j]+f[i-1][j]+f[i][j-1]-f[i-1][j-1]"),
        Spec("leetcode-1314-matrix-block-sum", "leetcode_1314_matrix_block_sum", "2d", "arith",
             _range_sum_2d, display_name="Matrix Block Sum (LeetCode 1314)",
             description="same 2D prefix-sum / summed-area-table recurrence as 304; 2D arith"),

        Spec("leetcode-221-maximal-square", "leetcode_221_maximal_square", "2d", "tropmin",
             _maximal_square, display_name="Maximal Square (LeetCode 221)",
             description="dp(i,j)=min(up,left,diag)+1 if grid==1 else 0; 2D TropMin, constant +1 taps"),
        Spec("leetcode-1277-count-square-submatrices", "leetcode_1277_count_square_submatrices", "2d", "tropmin",
             _maximal_square, display_name="Count Square Submatrices with All Ones (LeetCode 1277)",
             description="per-cell dp identical to 221 (largest all-ones square side); 2D TropMin"),

        # ── LeetCode 2D custom-semiring DP (boolean OR/AND) ──
        Spec("leetcode-392-is-subsequence", "leetcode_392_is_subsequence", "2d", "boolorand",
             _is_subsequence, display_name="Is Subsequence (LeetCode 392)",
             description="f(i,j)=f(i,j-1) OR (match AND f(i-1,j-1)); boolean OR/AND semiring, 2D"),

        # ── LeetCode 1D tropical time-varying DPs ──
        Spec("leetcode-746-min-cost-climbing-stairs", "leetcode_746_min_cost_climbing_stairs", "1d", "tropmin",
             _min_cost_climbing_stairs, display_name="Min Cost Climbing Stairs (LeetCode 746)",
             description="f(i)=cost[i]+min(f(i-1),f(i-2)); TropMin time-varying"),
        Spec("leetcode-198-house-robber", "leetcode_198_house_robber", "1d", "tropmax",
             _house_robber, display_name="House Robber (LeetCode 198)",
             description="f(i)=max(f(i-1), f(i-2)+nums[i]); TropMax time-varying"),
        Spec("leetcode-213-house-robber-ii", "leetcode_213_house_robber_ii", "1d", "tropmax",
             _house_robber, display_name="House Robber II (LeetCode 213)",
             description="Circular House Robber; same max(f(i-1), f(i-2)+nums[i]) recurrence as 198; TropMax TV"),
        Spec("leetcode-740-delete-and-earn", "leetcode_740_delete_and_earn", "1d", "tropmax",
             _delete_and_earn, display_name="Delete and Earn (LeetCode 740)",
             description="dp(v)=max(dp(v-1), dp(v-2)+total[v]); House Robber over the value axis; TropMax TV"),

        # ── LeetCode 1D tropical mutual recurrence ──
        Spec("leetcode-3259-max-energy-two-drinks", "leetcode_3259_max_energy_two_drinks", "1d", "tropmax",
             _max_energy_two_drinks, display_name="Maximum Energy Boost From Two Drinks (LeetCode 3259)",
             description="dp_A[i]=A[i]+max(dp_A[i-1],dp_B[i-2]); dp_B symmetric; TropMax 2x2 mutual"),

        # ── LeetCode 1D arithmetic time-varying counting ──
        Spec("leetcode-91-decode-ways", "leetcode_91_decode_ways", "1d", "arith",
             _decode_ways, display_name="Decode Ways (LeetCode 91)",
             description="f(i)=a(i)f(i-1)+b(i)f(i-2), a,b in {0,1}; arith time-varying, batched 64-step segments"),
        Spec("leetcode-639-decode-ways-ii", "leetcode_639_decode_ways_ii", "1d", "arith",
             _decode_ways, display_name="Decode Ways II (LeetCode 639)",
             description="same 2nd-order TV recurrence as 91 but with '*' multiplicities (coeffs > 1); batched segments"),

        # ── LeetCode 1D ring counting (segmented) ──
        Spec("leetcode-2466-count-good-strings", "leetcode_2466_count_good_strings", "1d", "arith",
             _count_good_strings, display_name="Count Ways to Build Good Strings (LeetCode 2466)",
             description="f(i)=f(i-1)+f(i-3); arith ring, batched 64-step segments"),
    ]
}
