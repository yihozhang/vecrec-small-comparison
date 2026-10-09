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
    RepeaterVec,
    SAdd,
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
from vecrec.expr.signal_ops import ChangeSemiring, Ith, PointwiseMul, SNeg, SSub
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


def _welford_variance() -> SignalExpr:
    with with_types(Type.Arith, ElementType.Float):
        w = Ith(ComputedSignal2D("w", cpp_struct="WelfordWeight2D",
                                 prelude=WELFORD_W_PRELUDE, setup_code="", make_args=""), 0)
        a = Var("a")
        wc = SSub(Num(1.0), w)                              # (x-1)/x
        M = Recurse(TVKernel([Num(0.0), wc]), PointwiseMul(w, a))
        delta1 = SSub(a, Convolve(TIKernel.z(), M))          # a(x) - M(x-1)  (M(0)=0)
        delta2 = SSub(a, M)                                  # a(x) - M(x)
        incr = PointwiseMul(delta1, delta2)  # (a(x) - M(x)) * (a(x) - M(x-1))
        M2 = Recurse(TIKernel([0.0, 1.0]), incr)             # M2(x) = M2(x-1) + incr(x)
        return PointwiseMul(w, M2)                           # var(x) = M2(x)/x


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


################################################################################
# ── LeetCode dynamic-programming recurrences ──────────────────────────────────
################################################################################

def _decode_ways() -> SignalExpr:
    # LeetCode 91: f(i) = a(i)*f(i-1) + b(i)*f(i-2), a(i),b(i) in {0,1} (valid single / pair).
    # Counts grow ~Fibonacci, so run short independent strings (one per row, reset per row);
    # seed s injects the f=1 empty-prefix base at each segment start.
    # YZ: I choose to not specialize a and b because the problem size is destined to be small 
    # due to the fibonacci like growth anyway, so the actual throughput on the actual problem
    # probably does not matter.
    # This is not true for decode ways 2, where the result is modulo 1e9+7.
    # TODO: make this problem work fully with decode ways 2.
    with with_types(Type.Arith, ElementType.I32):
        return Recurse(TVKernel([Num(0.0), Var("a"), Var("b")]), Var("s"))


# 1D ring counting recurrences (segmented per row to bound the geometric growth; int64).

# YZ: Not suitable because in the actual problem zero and one could be big.
# def _count_good_strings(zero: int = 1, one: int = 3) -> SignalExpr:
#     # LeetCode 2466: f(i) = f(i-zero) + f(i-one), seeded by s.
#     with with_types(Type.Arith, ElementType.I64):
#         data = [0.0] * (max(zero, one) + 1)
#         data[zero] = 1.0
#         data[one] = 1.0
#         return Recurse(TIKernel(data), Var("s"))


# The dp(0)=dp(1)=0 base case (you stand for free on step 0 or 1) is produced on the fly by a
# ComputedSignal2D whose C++ struct emits the TropMin one (0) at the first two stream positions
# and the TropMin zero (+inf) sentinel thereafter — no seed/impulse input signal. Instantiated by
# codegen as MinCostBase2D<vec_type, n_rows, block_size, block_row>; used via Ith(_, 0) as a 1D
# signal (same self-describing pattern as WelfordWeight2D).
MIN_COST_BASE_PRELUDE = r"""
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct MinCostBase2D {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    static constexpr elt_type ONE = (elt_type)0;                 // TropMin one
    static constexpr elt_type INF = (elt_type)(INT32_MAX / 8);   // TropMin zero (+inf)
    int my_col;
    MinCostBase2D() : my_col(0) {}
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        vec_type v = INF;                                        // common case: past the base window
        if constexpr (block_row == 0) {
            if (my_col < 2) {                                    // only the first call(s) touch the base
                if constexpr (lanes == 1) {
                    v = ONE;                                     // serial: first two calls return 0, 0
                } else {
                    _Pragma("unroll") for (int k = 0; k < lanes; k++) v[k] = (my_col + k < 2) ? ONE : INF;
                }
            }
        }
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) reg[base + i] = v;
        my_col += lanes;
    }
    void reset_and_next_row() { my_col = 0; }
};
"""


def _min_cost_climbing_stairs() -> SignalExpr:
    # LeetCode 746: dp(i) = min(dp(i-1)+cost[i-1], dp(i-2)+cost[i-2]); dp(0)=dp(1)=0.
    # (min, +): you stand for free on step 0 or 1 and pay cost[k] to *leave* step k, so the cost
    # rides on the source (lag) taps — cost[i-1] on the f(i-1) tap and cost[i-2] on the f(i-2) tap
    # (a 1- and 2-step delay of c). The dp(0)=dp(1)=0 base is emitted on the fly by a
    # ComputedSignal2D (MinCostBase2D above), so no seed input array is needed.
    with with_types(Type.TropMin, ElementType.I32):
        c = Var("c")
        kernel = TVKernel([
            Num(Type.TropMin.zero()),
            Convolve(TIKernel.z(), c),      # cost[i-1] on the f(i-1) tap
            Convolve(TIKernel.z(n=-2), c),  # cost[i-2] on the f(i-2) tap
        ])
        base = Ith(ComputedSignal2D("base", cpp_struct="MinCostBase2D",
                                    prelude=MIN_COST_BASE_PRELUDE, setup_code="", make_args=""), 0)
        return Recurse(kernel, base)


def _house_robber() -> SignalExpr:
    # LeetCode 198: f(i) = max(f(i-1), f(i-2) + nums[i]).
    # (max, +): +0 tap on f(i-1), +nums tap on f(i-2); seed s carries the base cases.
    with with_types(Type.TropMax, ElementType.I32):
        return Recurse(TVKernel([Num(Type.TropMax.zero()), Num(0.0), Var("n")]), Var("s"))

# Duplicate of house robber I think
# def _delete_and_earn() -> SignalExpr:
#     # LeetCode 740: dp(v) = max(dp(v-1), dp(v-2) + total[v]), total[v] = v * count(v).
#     # House Robber over the value axis (same recurrence as 198).
#     with with_types(Type.TropMax, ElementType.I32):
#         return Recurse(TVKernel([Num(Type.TropMax.zero()), Num(0.0), Var("t")]), Var("s"))


def _max_energy_two_drinks() -> SignalExpr:
    # LeetCode 3259: dp_A[i] = A[i] + max(dp_A[i-1], dp_B[i-2]); dp_B symmetric.
    # (max, +) 2x2 mutual: reward on the taps (stay = lag-1, switch = lag-2 cross). Output dp_A.
    with with_types(Type.TropMax, ElementType.I32):
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
    with with_types(Type.Arith, ElementType.I32):
        return Recurse2D(TIKernel2D([[0.0, 1.0], [1.0, -1.0]]), Var2D("m").at(0)).at(0)

# YZ: DP is not the most efficient solution for this one. Optimal solution uses two pointer.
# def _is_subsequence() -> SignalExpr:
#     # LeetCode 392: is s a subsequence of t? f(i,j) = f(i,j-1) OR (s[i-1]==t[j-1] AND f(i-1,j-1)).
#     # Boolean OR/AND, 2D: left tap passes through (coeff one), diagonal tap is the match mask; seeded
#     # true at (0,0).
#     with with_types(BoolOrAnd, ElementType.I32):
#         z = Num(0.0)
#         m = Var2D("m").at(0)   # diagonal: s[i-1] == t[j-1]
#         kernel = TVKernel2D([[z, Num(1.0)], [z, m]])
#         return Recurse2D(kernel, Var2D("s").at(0)).at(0)


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
    # TODO: make p compute nums1[i]*nums2[i]
    with with_types(Type.TropMax, ElementType.I32):
        z = Num(Type.TropMax.zero())
        p = Var2D("p").at(0)
        kernel = TVKernel2D([[z, Num(0.0)], [Num(0.0), p]])
        return Recurse2D(kernel, Var2D("p").at(0)).at(0)


# def _edit_distance() -> SignalExpr:
#     # LeetCode 72 (also 583; 1312 via a reversed axis). Levenshtein, a 2D (min,+) recurrence
#     # D(i,j) = min(D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1) + c(i,j)); c = per-cell substitution cost.
#     with with_types(Type.TropMin, ElementType.I32):
#         posinf = Num(Type.TropMin.zero())
#         gap = Num(1.0)
#         c = Var2D("c").at(0)
#         kernel = TVKernel2D([[posinf, gap], [gap, c]])
#         return Recurse2D(kernel, Impulse(0)).at(0)


def _maximal_square() -> SignalExpr:
    # LeetCode 221 (also 1277): side of the largest all-ones square ending at (i,j):
    #   dp(i,j) = min(dp(i-1,j), dp(i,j-1), dp(i-1,j-1)) + 1  if grid==1 else 0.
    # 2D TropMin with constant +1 taps (incl. the diagonal), like distance_transform. The mask is
    # the ⊕ input g: 0 where grid==0 (forces dp=0), 1 on a grid==1 boundary cell (a single-cell
    # square), and a large "+inf" sentinel on grid==1 interior cells (so the min(neighbors)+1 wins).
    # YZ: this is not faithful to the LeetCode problem, because the problem reset dp(i, j) when grid == 0.
    # Since the grid == 0 check is non-recursive, I think this is an acceptable benchmark to predict the performance.
    with with_types(Type.TropMin, ElementType.I32):
        k = TIKernel2D([[INF, 1.0], [1.0, 1.0]])
        return Recurse2D(k, Var2D("g").at(0)).at(0)


# Regular Expression Matching (LeetCode 10). The classic 2D boolean DP
#   dp[r][c] = "does s[0..r) match p[0..c)"  (dp[0][0] = true; '.' matches any char,
#   '*' matches zero-or-more of the preceding element) is a boolean OR/AND recurrence.
# Grid row r indexes the string (transition char s[r-1]), column c the pattern (char
# p[c-1]); the four DP branches decide which neighbour taps fire per cell, so each tap
# coefficient is a time-varying 0/1 mask (the branch predicate) in the OR(=max)/AND(=min)
# semiring. The masks are emitted by self-describing ComputedSignal2D structs: rather than
# synthesising s and p from the grid position by a hash, each mask reads the string and
# pattern from two external int8 buffers. ``setup_code`` (below) allocates+fills those buffers
# and registers them via regex_set_inputs() before the kernel runs; the masks cache the two
# pointers at construction. The buffers deliberately do NOT flow through ``make_args``: a
# make_args signal becomes a by-value kernel-constructor parameter, and the number/type of
# those parameters depends on the autotuned schedule (rows_per_iter, vector type), which vary
# per machine. Since generated/ headers are regenerated per machine but the C++ baseline
# (benchmarks/leetcode_10_regular_expression_matching.cpp) is committed source, a hard-coded
# constructor call there would not survive a re-tune. Registering the buffers out-of-band
# keeps the kernel constructor at the schedule-independent `(int row_size)`. The baseline
# fills the same buffers the same way, so it evaluates exactly the same recurrence.
# The external input string (s) and pattern (p), tiled to fill the benchmark grid. Letters
# 'a'..'e'; the pattern mixes '*' (zero-or-more of the preceding element) and '.' (any single
# char) so all four DP branches fire. Edit these to benchmark a different case. (These are the
# schedule-search inputs; the C++ baseline .cpp keeps its own matching copy — see there.)
REGEX_S_INPUT = "abcabcddeeabcde"
REGEX_P_INPUT = "a*b.c*d.e*b.a*"

REGEX_MATCH_PRELUDE = r"""
static constexpr char REGEX_SENTINEL = (char)1;  // never equals a real char, '.' or '*'

// The string s and pattern p are supplied as external int8 buffers (filled by the driver's
// setup_code / the C++ baseline) and registered here before the kernel runs; each mask caches
// these two pointers at construction. Grid row r reads s[r-1]; column c reads p[c-1]/p[c-2].
// The buffers are sized to the grid (plus a small halo), so every guarded in-range index is
// valid. Registering out-of-band (instead of via make_args) keeps the kernel constructor
// schedule-independent — see the note at _regex_matching().
static const int8_t *regex_s = nullptr;  // string chars,  length >= #rows
static const int8_t *regex_p = nullptr;  // pattern chars, length >= row_size
static inline void regex_set_inputs(const int8_t *s, const int8_t *p) { regex_s = s; regex_p = p; }

// Does pattern char pc match string char sc ('.' matches any real char)?
static inline bool regex_pat_matches(char pc, char sc) { return pc == '.' || pc == sc; }

// Char-domain vector of `lanes` elements. Element type is `char` (not int8_t) because clang's
// vector compares on ext_vector_type yield char vectors, and there is no implicit conversion
// between char and signed-char vectors. Compares run in this narrow 8-bit domain; the combined
// 0/-1 lane mask is widened to vec_type once with __builtin_convertvector.
template <int lanes>
struct RegexCharVec {
    typedef char type __attribute__((ext_vector_type(lanes)));
    // Unaligned whole-vector load; compiles to one movdqu / ldr q.
    static inline type loadu(const int8_t *base) {
        type v;
        __builtin_memcpy(&v, base, lanes);
        return v;
    }
    // Lane k = p[c0 + k - back], with REGEX_SENTINEL wherever c0 + k - back < 0. Since my_col
    // steps by `lanes`, only the first vector of each row (c0 == 0) touches negative indices;
    // every other call is a single in-bounds unaligned load (in-range reads are valid per the
    // buffer-sizing invariant above). The c0 == 0 case splats the sentinel and overwrites the
    // upper lanes with the first pattern chars — once per row, off the hot path.
    static inline type load_pat(const int8_t *p, int c0, int back) {
        if (__builtin_expect(c0 >= back, 1)) return loadu(p + (c0 - back));
        type v = (char)REGEX_SENTINEL;  // splat; covers lanes with c < back
        if (back < lanes) __builtin_memcpy((char *)&v + back, p, lanes - back);
        return v;
    }
};

// M_diag: dp[r-1][c-1] tap, active iff p[c-1] matches s[r-1] ('.'/literal match).
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct RegexMatchMask {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    using cvec = typename RegexCharVec<lanes>::type;
    static constexpr elt_type ONE = (elt_type)1;   // boolean one (true)
    static constexpr elt_type ZERO = (elt_type)0;  // boolean zero (false)
    const int8_t *__restrict__ s;  // string chars, indexed by (row - 1)
    const int8_t *__restrict__ p;  // pattern chars, indexed by (col - 1)
    int my_col;
    int my_row;
    RegexMatchMask() : s(regex_s), p(regex_p), my_col(0), my_row(block_row) {}
    inline elt_type mask_at(int c) const {
        char sc = (my_row >= 1) ? (char)s[my_row - 1] : REGEX_SENTINEL;
        char pc = (c >= 1) ? (char)p[c - 1] : REGEX_SENTINEL;
        return regex_pat_matches(pc, sc) ? ONE : ZERO;
    }
    // Whole-vector form of mask_at: one unaligned load of the lane pattern chars (sentinel
    // shifted in at the row start), branchless compares (pc=='.' | pc==sc), widen -> 1/0.
    inline vec_type mask_vec(int c0) const {
        const char sc = (my_row >= 1) ? (char)s[my_row - 1] : (char)REGEX_SENTINEL;
        const cvec pcv = RegexCharVec<lanes>::load_pat(p, c0, 1);  // lane k: p[c0+k-1]
        const cvec scv = sc, dotv = '.';
        const cvec m = (pcv == dotv) | (pcv == scv);  // 0 / -1 lane masks
        return __builtin_convertvector(m, vec_type) & (vec_type)ONE;
    }
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        vec_type v;
        if constexpr (lanes == 1) {
            v = mask_at(my_col);
        } else {
            v = mask_vec(my_col);  // single load + whole-vector compares (no per-lane gather)
        }
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) reg[base + i] = v;
        my_col += lanes;
    }
    void reset_and_next_row() { my_col = 0; my_row += 1; }
};

// M_back2: dp[r][c-2] tap (zero occurrences of the starred element), active iff p[c-1]=='*'.
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct RegexStarMask {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    using cvec = typename RegexCharVec<lanes>::type;
    static constexpr elt_type ONE = (elt_type)1;
    static constexpr elt_type ZERO = (elt_type)0;
    const int8_t *__restrict__ p;  // pattern chars, indexed by (col - 1)
    int my_col;
    int my_row;
    RegexStarMask() : p(regex_p), my_col(0), my_row(block_row) {}
    inline elt_type mask_at(int c) const {
        char pc = (c >= 1) ? (char)p[c - 1] : REGEX_SENTINEL;
        return (pc == '*') ? ONE : ZERO;
    }
    // Whole-vector form of mask_at: one unaligned load, compare against '*', widen -> 1/0.
    inline vec_type mask_vec(int c0) const {
        const cvec pcv = RegexCharVec<lanes>::load_pat(p, c0, 1);  // lane k: p[c0+k-1]
        const cvec starv = '*';
        const cvec m = (pcv == starv);
        return __builtin_convertvector(m, vec_type) & (vec_type)ONE;
    }
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        vec_type v;
        if constexpr (lanes == 1) {
            v = mask_at(my_col);
        } else {
            v = mask_vec(my_col);  // single load + whole-vector compare (no per-lane gather)
        }
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) reg[base + i] = v;
        my_col += lanes;
    }
    void reset_and_next_row() { my_col = 0; my_row += 1; }
};

// M_back1 (dp[r][c-1]) and M_up (dp[r-1][c]) taps (one-or-more occurrences): both active
// iff p[c-1]=='*' AND the preceding element p[c-2] matches s[r-1].
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct RegexStarMatchMask {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    using cvec = typename RegexCharVec<lanes>::type;
    static constexpr elt_type ONE = (elt_type)1;
    static constexpr elt_type ZERO = (elt_type)0;
    const int8_t *__restrict__ s;  // string chars, indexed by (row - 1)
    const int8_t *__restrict__ p;  // pattern chars, indexed by (col - 1)
    int my_col;
    int my_row;
    RegexStarMatchMask() : s(regex_s), p(regex_p), my_col(0), my_row(block_row) {}
    inline elt_type mask_at(int c) const {
        char sc  = (my_row >= 1) ? (char)s[my_row - 1] : REGEX_SENTINEL;
        char pc  = (c >= 1) ? (char)p[c - 1] : REGEX_SENTINEL;
        char pc2 = (c >= 2) ? (char)p[c - 2] : REGEX_SENTINEL;
        return (pc == '*' && regex_pat_matches(pc2, sc)) ? ONE : ZERO;
    }
    // Whole-vector form of mask_at: two overlapping unaligned loads give the p[c-1] and
    // p[c-2] lanes (clang typically folds the second into a palignr/ext of the first), then
    // branchless compares: (pc=='*') & (pc2=='.' | pc2==sc), widen -> 1/0.
    inline vec_type mask_vec(int c0) const {
        const char sc = (my_row >= 1) ? (char)s[my_row - 1] : (char)REGEX_SENTINEL;
        const cvec pcv  = RegexCharVec<lanes>::load_pat(p, c0, 1);  // lane k: p[c0+k-1]
        const cvec pc2v = RegexCharVec<lanes>::load_pat(p, c0, 2);  // lane k: p[c0+k-2]
        const cvec starv = '*', scv = sc, dotv = '.';
        const cvec m = (pcv == starv) & ((pc2v == dotv) | (pc2v == scv));
        return __builtin_convertvector(m, vec_type) & (vec_type)ONE;
    }
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        vec_type v;
        if constexpr (lanes == 1) {
            v = mask_at(my_col);
        } else {
            v = mask_vec(my_col);  // two loads + whole-vector compares (no per-lane gather)
        }
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) reg[base + i] = v;
        my_col += lanes;
    }
    void reset_and_next_row() { my_col = 0; my_row += 1; }
};
"""


def _regex_matching() -> SignalExpr:
    # LeetCode 10: dp[r][c] = OR of the four active branch taps, seeded dp[0][0]=true.
    #   dp[r][c] = (M_back1 & dp[r][c-1]) | (M_back2 & dp[r][c-2])       (same row)
    #            | (M_up    & dp[r-1][c]) | (M_diag  & dp[r-1][c-1])     (row back)
    # (& = AND = min, | = OR = max). The base row/column emerge from the masks: the '*'
    # zero-occurrence tap (M_back2, column-only) propagates dp[0][*], and every tap that
    # would cross the empty-string boundary reads an out-of-range (zero) neighbour.
    # The four coefficients are the time-varying 0/1 masks defined in the prelude above.
    # setup_code (emitted once in the schedule-search driver, where `N` and `row_size` are in
    # scope): allocate the string/pattern buffers, tile the fixed inputs across them, and
    # register them via regex_set_inputs() before the kernel is constructed. regex_s_buf is
    # row-indexed (length = #rows = N / row_size), regex_p_buf column-indexed (length =
    # row_size); both get a +64 halo for lane over-read and the trailing reset_and_next_row().
    # Registering out-of-band (rather than passing the pointers through make_args) leaves the
    # kernel constructor at `(int row_size)`, independent of the autotuned schedule — see the
    # note above the prelude for why the committed C++ baseline needs that.
    setup_code = (
        f'static const char REGEX_S_BASE[] = "{REGEX_S_INPUT}";\n'
        f'static const char REGEX_P_BASE[] = "{REGEX_P_INPUT}";\n'
        f'std::vector<int8_t> regex_s_buf(N / row_size + 64), regex_p_buf(row_size + 64);\n'
        f'for (size_t i = 0; i < regex_s_buf.size(); i++) regex_s_buf[i] = (int8_t)REGEX_S_BASE[i % (sizeof(REGEX_S_BASE) - 1)];\n'
        f'for (size_t i = 0; i < regex_p_buf.size(); i++) regex_p_buf[i] = (int8_t)REGEX_P_BASE[i % (sizeof(REGEX_P_BASE) - 1)];\n'
        f'regex_set_inputs(regex_s_buf.data(), regex_p_buf.data());\n'
    )
    with with_types(BoolOrAnd, ElementType.I8):
        def mask(name: str, struct: str) -> SignalExpr:
            return Ith(ComputedSignal2D(name, cpp_struct=struct, prelude=REGEX_MATCH_PRELUDE,
                                        setup_code=setup_code, make_args=""), 0)
        m_diag = mask("rx_diag", "RegexMatchMask")
        m_back2 = mask("rx_star", "RegexStarMask")
        m_starmatch = mask("rx_starmatch", "RegexStarMatchMask")  # shared by M_back1 and M_up
        z = Num(BoolOrAnd.zero())  # boolean false (semiring zero): self / unused taps
        kernel = TVKernel2D(
            [
                [z,           m_starmatch, m_back2],  # di=0: self, dp[r][c-1], dp[r][c-2]
                [m_starmatch, m_diag,      z],        # di=1: dp[r-1][c], dp[r-1][c-1], (unused)
            ]
        )
        return Recurse2D(kernel, Impulse(1.0)).at(0)


# Best Time to Buy and Sell Stock IV (LeetCode 188), as a 2-state mutual recurrence over
# x = number of transactions (rows) and y = day (columns), in TropMax (plus=max, times=+):
#   f_1(x,y) = max(f_1(x,y-1),        f_2(x-1,y-1) - g(y))   # holding a share
#   f_2(x,y) = max(f_1(x,y-1) + g(y), f_2(x,y-1))            # not holding
# g(y) is the day-y price: it depends only on the column, so it is emitted by a column-indexed
# ComputedSignal2D (StockPriceSignal2D) that returns g[my_col], the same value for every row --
# analogous to the on-the-fly score in vec-rec/benchmarks/trop/affine_gap.py. The price buffer is
# registered out-of-band via stock_set_prices() (empty make_args -> the kernel constructor stays
# `(int row_size)`, schedule-independent -- same rationale as _regex_matching).
STOCK_PRICE_PRELUDE = r"""
// Daily prices g(y), column-indexed (identical for every transaction row x), supplied as an
// external int32 buffer registered before the kernel runs. my_col tracks the day, reset per row.
static const int32_t *stock_g = nullptr;   // prices, length >= row_size (+ halo for lane over-read)
static inline void stock_set_prices(const int32_t *g) { stock_g = g; }

template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct StockPriceSignal2D {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    const int32_t *__restrict__ g;
    int my_col;
    StockPriceSignal2D() : g(stock_g), my_col(0) {}
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        vec_type gv;
        if constexpr (lanes == 1) {
            gv = (elt_type)g[my_col];
        } else {
            _Pragma("unroll") for (int k = 0; k < lanes; k++) gv[k] = (elt_type)g[my_col + k];
        }
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) reg[base + i] = gv;
        my_col += lanes;
    }
    void reset_and_next_row() { my_col = 0; }
};
"""

# Deterministic day prices in [1, 100] from a position hash; the C++ baseline fills the price
# buffer with the identical formula so it evaluates the same recurrence.
STOCK_PRICE_SETUP = (
    "std::vector<int32_t> stock_g_buf(row_size + 64);\n"
    "for (size_t j = 0; j < stock_g_buf.size(); j++) {\n"
    "    unsigned h = (unsigned)j * 2654435761u + 1013904223u; h ^= h >> 15;\n"
    "    stock_g_buf[j] = (int32_t)(1 + (int)(h % 100u));\n"
    "}\n"
    "stock_set_prices(stock_g_buf.data());\n"
)


def _best_time_stock_iv() -> SignalExpr:
    # LeetCode 188. RepeaterVec over x (transactions): each row is a 1D column recurrence pair
    # [f_1, f_2]. f_1 (holding) is a prefix-max fed the previous row's f_2 on the diagonal minus
    # the price; f_2 (not holding) is a prefix-max fed the current row's f_1 (delayed one day) plus
    # the price. The origin is seeded with the TropMax one (0) so f_2(0,0)=0 (zero profit, zero
    # transactions); row 0 then stays f_2==0 / f_1==-inf (the "0 transactions" base), and rows x>=1
    # build up the k-transaction profit. Output = f_2 (max profit while not holding).
    with with_types(Type.TropMax, ElementType.I32):
        neginf = Type.TropMax.zero()   # -inf: lag-0 self tap (no instantaneous self-dependence)
        one = Type.TropMax.one()       # 0:    lag-1 self coefficient (prefix-max carries forward)
        z = TIKernel.z()               # unit column delay (z^-1)
        price = ComputedSignal2D("g", cpp_struct="StockPriceSignal2D", prelude=STOCK_PRICE_PRELUDE,
                                 setup_code=STOCK_PRICE_SETUP, make_args="")

        def body(prev):
            g = Ith(price, 0)                                    # g(y)
            neg_g = ChangeSemiring(SNeg(ChangeSemiring(g, Type.Arith)), Type.TropMax)  # -g(y)
            f2_prev_diag = Convolve(z, prev.at(1).at(0))         # f_2(x-1, y-1)
            # f_1(x,y) = max(f_1(x,y-1), f_2(x-1,y-1) - g(y))
            buy = PointwiseMul(f2_prev_diag, neg_g)              # f_2(x-1,y-1) - g(y)
            f1 = Recurse(TIKernel([neginf, one]), buy)
            # f_2(x,y) = max(f_1(x,y-1) + g(y), f_2(x,y-1)), seeded f_2(0,0)=0
            sell = PointwiseMul(Convolve(z, f1), g)              # f_1(x,y-1) + g(y)
            f2 = Recurse(TIKernel([neginf, one]), SAdd(sell, Impulse(Type.TropMax.one())))
            return SignalVector([f1, f2])

        rv = RepeaterVec(2, body)
        return Ith(rv.at(1), 0)  # current-row f_2 (max profit, not holding)


def _best_time_stock_i() -> SignalExpr:
    # Best Time to Buy and Sell Stock (LeetCode 121): one transaction, the classic O(n) scan
    #   f1(i) = min(f1(i-1), g[i])            # cheapest price seen so far
    #   f2(i) = max(f2(i-1), g[i] - f1(i-1))  # best profit selling on day i (buy strictly earlier)
    # f1 is a min-scan and f2 a max-scan -- different (tropical) semirings. To keep everything in
    # one semiring (so the two scans dilate together), we carry pm(i) = -f1(i) = max(pm(i-1), -g[i])
    # instead of f1: then g[i] - f1(i-1) = g[i] + pm(i-1), and both scans are TropMax (max,+) prefix
    # maxima. Output is f2 (the max profit); f1 = -pm is left implicit. g has no additive inverse in
    # TropMax, so -g is formed in Arith and viewed back (as in _dungeon_game / affine_gap).
    with with_types(Type.TropMax, ElementType.I32):
        g = Var("g")
        zero = Type.TropMax.zero()   # -inf: lag-0 self tap (no instantaneous self-dependence)
        one = Type.TropMax.one()     # 0:    lag-1 self coefficient (prefix scan carries forward)
        neg_g = ChangeSemiring(SNeg(ChangeSemiring(g, Type.Arith)), Type.TropMax)  # -g(i)
        pm = Recurse(TIKernel([zero, one]), neg_g)          # pm(i) = max(pm(i-1), -g(i)) = -min g[0..i]
        pm_prev = Convolve(TIKernel.z(), pm)                # pm(i-1)
        buy_low = PointwiseMul(g, pm_prev)                  # g(i) + pm(i-1) = g(i) - min g[0..i-1]
        f2 = Recurse(TIKernel([zero, one]), buy_low)        # f2(i) = max(f2(i-1), g(i) - f1(i-1))
        return f2


def _best_time_stock_with_cooldown() -> SignalExpr:
    # Best Time to Buy and Sell Stock with Cooldown (LeetCode 309). The classic 3-state DP
    #   rest[i] = max(rest[i-1], sold[i-1])          # idle: post-cooldown / never held
    #   hold[i] = max(hold[i-1], rest[i-1] - g[i])   # holding a share (buying reads rest, not sold,
    #                                                #   which is exactly the one-day cooldown)
    #   sold[i] = hold[i-1] + g[i]                   # just sold today
    # sold is not itself recursive -- it is hold delayed one day plus the price -- so we inline
    # sold[i-1] = hold[i-2] + g[i-1] into rest and collapse the system to a 2-state mutual
    # recurrence over (rest, hold), which VecRec dilates like any 2x2 (cf. _max_energy_two_drinks /
    # _double_exp_smoothing) rather than needing a 3-component RecurseV:
    #   rest[i] = max(rest[i-1],  hold[i-2] + g[i-1])
    #   hold[i] = max(hold[i-1],  rest[i-1] - g[i])
    # (max, +): +g(i-1) and -g(i) ride on the cross taps as time-varying coefficients (-g is formed
    # in Arith and viewed back into TropMax, as in _best_time_stock_i). Two seeds carry the base
    # cases: rest gets the TropMax one (0) at day 0 (rest[-1]=0, zero profit), and hold gets -g
    # masked to day 0 (hold[0]=-g[0], so buying on day 0 is allowed). Output = rest, the running max
    # profit while not holding; the LeetCode answer max(rest, sold) on the last day is the next rest,
    # which rest already carries forward.
    with with_types(Type.TropMax, ElementType.I32):
        g = Var("g")
        zero = Num(Type.TropMax.zero())   # -inf: semiring zero (empty tap / no lag-0 self-dep)
        one = Type.TropMax.one()          # 0:    semiring one (prefix-max carries a tap forward)
        neg_g = ChangeSemiring(SNeg(ChangeSemiring(g, Type.Arith)), Type.TropMax)  # -g(i)
        g_prev = Convolve(TIKernel.z(), g)                                          # g(i-1)
        # 2x2 kernel over (rest=0, hold=1); each TVKernel lists taps [lag0, lag1, lag2].
        A = KernelMatrix(
            [
                [TVKernel([zero, Num(one), zero]), TVKernel([zero, zero, g_prev])],   # rest row
                [TVKernel([zero, neg_g, zero]),    TVKernel([zero, Num(one), zero])],  # hold row
            ]
        )
        rest_seed = Impulse(Type.TropMax.one())                       # rest[0] = 0 (base rest[-1]=0)
        hold_seed = PointwiseMul(neg_g, Impulse(Type.TropMax.one()))  # -g masked to day 0
        F = RecurseV(A, SignalVector([rest_seed, hold_seed]))
        return VectorAt(F, 0)  # rest: max profit while not holding


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
        Spec("welford-variance", "welford_variance", "1d", "arith",
             _welford_variance, display_name="Welford Variance",
             description="Welford running population variance var(x)=M2(x)/x, M2 a prefix sum of "
                         "((x-1)/x)*(a(x)-M(x-1))^2 coupled to the running mean M"),
        Spec("continued-fraction", "continued_fraction", "1d", "arith",
             _continued_fraction, display_name="Continued Fraction",
             description="Wallis-Euler continued fraction h(x)=a(x)h(x-1)+b(x)h(x-2), f=h(x)/h(x-1); "
                         "batched 512-step segments in float64 (see registry note)"),

        # ── LeetCode 2D sequence-alignment DPs (TropMax / TropMin) ──
        Spec("leetcode-1143-lcs", "leetcode_1143_lcs", "2d", "tropmax",
             _lcs, display_name="Longest Common Subsequence (LeetCode 1143)",
             description="dp(i,j)=max(dp(i-1,j),dp(i,j-1),dp(i-1,j-1)+match); 2D TropMax alignment"),
        # Spec("leetcode-1035-uncrossed-lines", "leetcode_1035_uncrossed_lines", "2d", "tropmax",
        #      _lcs, display_name="Uncrossed Lines (LeetCode 1035)",
        #      description="Identical recurrence to LCS on two integer arrays; 2D TropMax alignment"),
        # Spec("leetcode-1092-shortest-common-supersequence", "leetcode_1092_shortest_common_supersequence", "2d", "tropmax",
        #      _lcs, display_name="Shortest Common Supersequence (LeetCode 1092)",
        #      description="length = len(a)+len(b)-LCS(a,b); the DP is the LCS recurrence; 2D TropMax alignment"),
        # Spec("leetcode-1216-valid-palindrome-iii", "leetcode_1216_valid_palindrome_iii", "2d", "tropmax",
        #      _lcs, display_name="Valid Palindrome III (LeetCode 1216)",
        #      description="k>=n-LPS(s); LPS = LCS-form over s vs its reverse (diagonal reward 2); 2D TropMax"),
        # Spec("leetcode-516-longest-palindromic-subseq", "leetcode_516_longest_palindromic_subseq", "2d", "tropmax",
        #      _lcs, display_name="Longest Palindromic Subsequence (LeetCode 516)",
        #      description="LPS = LCS-form over one string vs its reverse (diagonal reward 2 on match); 2D TropMax"),
        # Spec("leetcode-3503-longest-palindrome-concat", "leetcode_3503_longest_palindrome_concat", "2d", "tropmax",
        #      _lcs, display_name="Longest Palindrome After Concatenation (LeetCode 3503)",
        #      description="LPS-form (reward 2 on match) over a combined string; 2D TropMax alignment"),
        Spec("leetcode-1458-max-dot-product", "leetcode_1458_max_dot_product", "2d", "tropmax",
             _max_dot_product, display_name="Max Dot Product of Two Subsequences (LeetCode 1458)",
             description="dp(i,j)=max(dp(i-1,j),dp(i,j-1),dp(i-1,j-1)+p,p), p=nums1[i]*nums2[j]; 2D TropMax"),
        # Spec("leetcode-72-edit-distance", "leetcode_72_edit_distance", "2d", "tropmin",
        #      _edit_distance, display_name="Edit Distance (LeetCode 72)",
        #      description="Levenshtein D(i,j)=min(D(i-1,j)+1,D(i,j-1)+1,D(i-1,j-1)+cost); 2D TropMin alignment"),
        # Spec("leetcode-583-delete-two-strings", "leetcode_583_delete_two_strings", "2d", "tropmin",
        #      _edit_distance, display_name="Delete Operation for Two Strings (LeetCode 583)",
        #      description="Edit distance with insert/delete only (substitution forbidden); 2D TropMin alignment"),
        # Spec("leetcode-1312-min-insertion-palindrome", "leetcode_1312_min_insertion_palindrome", "2d", "tropmin",
        #      _edit_distance, display_name="Minimum Insertion Steps to Make Palindrome (LeetCode 1312)",
        #      description="Min insertions = edit-distance form over one string vs its reverse; 2D TropMin"),

        # ── LeetCode 2D arithmetic (summed-area table) ──
        Spec("leetcode-304-range-sum-2d", "leetcode_304_range_sum_2d", "2d", "arith",
             _range_sum_2d, display_name="Range Sum Query 2D (LeetCode 304)",
             description="2D prefix sum / summed-area table: f[i][j]=m[i][j]+f[i-1][j]+f[i][j-1]-f[i-1][j-1]"),
        # Spec("leetcode-1314-matrix-block-sum", "leetcode_1314_matrix_block_sum", "2d", "arith",
        #      _range_sum_2d, display_name="Matrix Block Sum (LeetCode 1314)",
        #      description="same 2D prefix-sum / summed-area-table recurrence as 304; 2D arith"),

        Spec("leetcode-221-maximal-square", "leetcode_221_maximal_square", "2d", "tropmin",
             _maximal_square, display_name="Maximal Square (LeetCode 221)",
             description="dp(i,j)=min(up,left,diag)+1 if grid==1 else 0; 2D TropMin, constant +1 taps"),
        # Spec("leetcode-1277-count-square-submatrices", "leetcode_1277_count_square_submatrices", "2d", "tropmin",
        #      _maximal_square, display_name="Count Square Submatrices with All Ones (LeetCode 1277)",
        #      description="per-cell dp identical to 221 (largest all-ones square side); 2D TropMin"),

        # ── LeetCode 2D custom-semiring DP (boolean OR/AND) ──
        Spec("leetcode-10-regular-expression-matching", "leetcode_10_regular_expression_matching",
             "2d", "boolorand", _regex_matching,
             display_name="Regular Expression Matching (LeetCode 10)",
             description="dp[r][c]= match/'.'/'*' branches as time-varying 0/1 masks; boolean OR/AND, 2D"),
        # Spec("leetcode-392-is-subsequence", "leetcode_392_is_subsequence", "2d", "boolorand",
        #      _is_subsequence, display_name="Is Subsequence (LeetCode 392)",
        #      description="f(i,j)=f(i,j-1) OR (match AND f(i-1,j-1)); boolean OR/AND semiring, 2D"),

        # ── LeetCode 1D tropical time-varying DPs ──
        Spec("leetcode-746-min-cost-climbing-stairs", "leetcode_746_min_cost_climbing_stairs", "1d", "tropmin",
             _min_cost_climbing_stairs, display_name="Min Cost Climbing Stairs (LeetCode 746)",
             description="dp(i)=min(dp(i-1)+cost[i-1],dp(i-2)+cost[i-2]), dp(0)=dp(1)=0; TropMin time-varying"),
        Spec("leetcode-198-house-robber", "leetcode_198_house_robber", "1d", "tropmax",
             _house_robber, display_name="House Robber (LeetCode 198)",
             description="f(i)=max(f(i-1), f(i-2)+nums[i]); TropMax time-varying"),
        # Spec("leetcode-213-house-robber-ii", "leetcode_213_house_robber_ii", "1d", "tropmax",
        #      _house_robber, display_name="House Robber II (LeetCode 213)",
        #      description="Circular House Robber; same max(f(i-1), f(i-2)+nums[i]) recurrence as 198; TropMax TV"),
        # Spec("leetcode-740-delete-and-earn", "leetcode_740_delete_and_earn", "1d", "tropmax",
        #      _delete_and_earn, display_name="Delete and Earn (LeetCode 740)",
        #      description="dp(v)=max(dp(v-1), dp(v-2)+total[v]); House Robber over the value axis; TropMax TV"),

        # ── LeetCode 1D tropical mutual recurrence ──
        Spec("leetcode-3259-max-energy-two-drinks", "leetcode_3259_max_energy_two_drinks", "1d", "tropmax",
             _max_energy_two_drinks, display_name="Maximum Energy Boost From Two Drinks (LeetCode 3259)",
             description="dp_A[i]=A[i]+max(dp_A[i-1],dp_B[i-2]); dp_B symmetric; TropMax 2x2 mutual"),
        Spec("leetcode-309-best-time-stock-cooldown", "leetcode_309_best_time_stock_cooldown", "1d", "tropmax",
             _best_time_stock_with_cooldown,
             display_name="Best Time to Buy and Sell Stock with Cooldown (LeetCode 309)",
             description="rest[i]=max(rest[i-1],hold[i-2]+g[i-1]); hold[i]=max(hold[i-1],rest[i-1]-g[i]); "
                         "TropMax 2-state mutual (sold inlined)"),

        # ── LeetCode 1D tropical prefix scan ──
        Spec("leetcode-121-best-time-stock", "leetcode_121_best_time_stock", "1d", "tropmax",
             _best_time_stock_i, display_name="Best Time to Buy and Sell Stock (LeetCode 121)",
             description="f1(i)=min(f1(i-1),g[i]); f2(i)=max(f2(i-1),g[i]-f1(i-1)); "
                         "TropMax prefix scans (pm=-f1)"),

        # ── LeetCode 2D tropical mutual recurrence ──
        Spec("leetcode-188-best-time-stock-iv", "leetcode_188_best_time_stock_iv", "2d", "tropmax",
             _best_time_stock_iv, display_name="Best Time to Buy and Sell Stock IV (LeetCode 188)",
             description="f1(x,y)=max(f1(x,y-1),f2(x-1,y-1)-g); f2(x,y)=max(f1(x,y-1)+g,f2(x,y-1)); "
                         "TropMax 2-state mutual over transactions x days (RepeaterVec)"),

        # ── LeetCode 1D arithmetic time-varying counting ──
        Spec("leetcode-91-decode-ways", "leetcode_91_decode_ways", "1d", "arith",
             _decode_ways, display_name="Decode Ways (LeetCode 91)",
             description="f(i)=a(i)f(i-1)+b(i)f(i-2), a,b in {0,1}; arith time-varying, batched 64-step segments"),
        # Spec("leetcode-639-decode-ways-ii", "leetcode_639_decode_ways_ii", "1d", "arith",
        #      _decode_ways, display_name="Decode Ways II (LeetCode 639)",
        #      description="same 2nd-order TV recurrence as 91 but with '*' multiplicities (coeffs > 1); batched segments"),

        # # ── LeetCode 1D ring counting (segmented) ──
        # Spec("leetcode-2466-count-good-strings", "leetcode_2466_count_good_strings", "1d", "arith",
        #      _count_good_strings, display_name="Count Ways to Build Good Strings (LeetCode 2466)",
        #      description="f(i)=f(i-1)+f(i-3); arith ring, batched 64-step segments"),
    ]
}
