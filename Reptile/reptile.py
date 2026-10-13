"""REPTILE-style tiling around the VecRec black-box kernel (code generation).

This module is a small stand-in for REPTILE's ``codeGen``/``cCodeGen`` pipeline,
specialized to the two dense-DP recurrences of interest (Needleman-Wunsch and
affine-gap Gotoh). It does *not* re-implement the VecRec / RECUMA compiler; it
treats a compiled VecRec inner-loop kernel as an opaque black box (an "external
function" in REPTILE terms) and constructs tile loops around it, and it also
emits REPTILE-style *generated* tiled loop nests for the same recurrence so the
two can be compared.

Two driver families are produced (both self-time via the shared
``baselines/common/driver_io.hpp`` contract, so their numbers are directly
comparable to everything else in the repo):

1. ``vecrec`` family — the VecRec kernel is the mapped external tile primitive:
   * ``untiled``: one full row-by-row sweep (plain VecRec, the black box as-is).
   * ``row``:     REPTILE row-strip tiling — an outer loop over row-tiles, each
                  computed by the same continuous kernel state.
   * ``col``:     REPTILE column-panel tiling (checkpoint/restore the carries).
   * ``block``:   REPTILE 2D tiling — row-band outer, column-panel inner.

2. ``loops`` family — REPTILE's tiled scalar loops (no external mapping),
   taken from the implementation in REPTILE's artifact and embedded here as
   code templates. Approaches: ``untiled`` / ``row`` / ``col`` / ``block``.
   It is the scalar (no vectorized kernel) comparison point for the VecRec
   kernel.

The chosen approach and tile sizes are read at *run time* from ``input.json``
(fields ``approach`` / ``tile_rows`` / ``tile_cols``), so one compiled binary
covers the whole sweep for a given recurrence + schedule.
"""

from __future__ import annotations

import contextlib
import io as _io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _platform_compiler_flags() -> list[str]:
    from evals_common.harness import platform_compiler_flags

    return platform_compiler_flags()


# --------------------------------------------------------------------------
# Recurrence specs. Each maps to a case's kernel expr + the DP conventions the
# shared traceback helpers expect (gap magnitudes come from input.json).
# --------------------------------------------------------------------------

_M = "static_cast<int>(in.match)"
_MM = "static_cast<int>(in.mismatch)"


def _nw_expr():
    from bio_kernel_evals.cases.needleman_wunsch import _build_expr

    return _build_expr()


def _affine_expr():
    from bio_kernel_evals.cases.affine_gap import _build_expr

    return _build_expr()


def _sw_expr():
    from bio_kernel_evals.cases.smith_waterman import _build_expr

    return _build_expr()


# kind -> config
KINDS = {
    "nw": {
        "expr": _nw_expr,
        "gap_setup": "const int gap = -static_cast<int>(in.gap_extend);",
        # VecRec full-matrix traceback (score = logical corner).
        "vecrec_traceback": (
            f"baselines::nw_traceback_full_inline(output.data(), rows, cols, phys_w, "
            f"in.x.c_str(), in.y.c_str(), gap, {_M}, {_MM}, q.data(), r.data(), c.data());\n"
            "            score = static_cast<long>(output[static_cast<std::size_t>(rows) * phys_w + cols]);"
        ),
        "loops_traceback": (
            f"baselines::nw_traceback_full_inline(mat.data(), rows, cols, phys_w, "
            f"in.x.c_str(), in.y.c_str(), gap, {_M}, {_MM}, q.data(), r.data(), c.data());\n"
            "            score = static_cast<long>(mat[static_cast<std::size_t>(rows) * phys_w + cols]);"
        ),
        # Score-only: global-alignment score = the corner cell H[rows][cols].
        "score_only": "corner",
    },
    "affine": {
        "expr": _affine_expr,
        "gap_setup": (
            "const int go = -static_cast<int>(in.gap_open);\n"
            "        const int ge = -static_cast<int>(in.gap_extend);"
        ),
        "vecrec_traceback": (
            f"baselines::affine_traceback_full_inline(output.data(), rows, cols, phys_w, "
            f"in.x.c_str(), in.y.c_str(), go, ge, {_M}, {_MM}, q.data(), r.data(), c.data());\n"
            "            score = static_cast<long>(output[static_cast<std::size_t>(rows) * phys_w + cols]);"
        ),
        "loops_traceback": (
            f"baselines::affine_traceback_full_inline(H.data(), rows, cols, phys_w, "
            f"in.x.c_str(), in.y.c_str(), go, ge, {_M}, {_MM}, q.data(), r.data(), c.data());\n"
            "            score = static_cast<long>(H[static_cast<std::size_t>(rows) * phys_w + cols]);"
        ),
        "score_only": "corner",
    },
    "sw": {
        "expr": _sw_expr,
        "gap_setup": "const int gap = -static_cast<int>(in.gap_extend);",
        # Local alignment: score is the global max over the whole matrix (argmax),
        # not the corner. sw_traceback_full_inline scans the matrix, writes the
        # local-optimum score into _sw_score, and walks back from the argmax cell.
        # The argmax is a full reduction done driver-side, after the (possibly
        # tiled) matrix is materialized, so it is unaffected by tiling.
        "vecrec_traceback": (
            "int64_t _sw_score = 0;\n"
            f"            baselines::sw_traceback_full_inline(output.data(), rows, cols, phys_w, "
            f"in.x.c_str(), in.y.c_str(), gap, {_M}, {_MM}, q.data(), r.data(), c.data(), &_sw_score);\n"
            "            score = static_cast<long>(_sw_score);"
        ),
        "loops_traceback": (
            "int64_t _sw_score = 0;\n"
            f"            baselines::sw_traceback_full_inline(mat.data(), rows, cols, phys_w, "
            f"in.x.c_str(), in.y.c_str(), gap, {_M}, {_MM}, q.data(), r.data(), c.data(), &_sw_score);\n"
            "            score = static_cast<long>(_sw_score);"
        ),
        # Score-only: local-alignment score = running max over all cells.
        "score_only": "max",
    },
}


# On-the-fly substitution setup for the affine ComputedSignal2D path
# (mirrors evals_common/vecrec_cpp._HARNESS_SUBST_SETUP).
_HARNESS_SUBST_SETUP = """auto _c2i8 = [](char ch) -> int8_t {
            switch (ch) {
            case 'A': case 'a': return 0; case 'C': case 'c': return 1;
            case 'G': case 'g': return 2; case 'T': case 't': return 3;
            default: return 0;
            }
        };
        std::vector<int8_t> s1(static_cast<std::size_t>(rows) + 64, static_cast<int8_t>(-1));
        std::vector<int8_t> s2(static_cast<std::size_t>(phys_w), static_cast<int8_t>(-1));
        for (int i = 0; i < rows; ++i)
            s1[static_cast<std::size_t>(i + 1)] = _c2i8(in.x[static_cast<std::size_t>(i)]);
        for (int j = 0; j < cols; ++j)
            s2[static_cast<std::size_t>(j + 1)] = _c2i8(in.y[static_cast<std::size_t>(j)]);"""


# --------------------------------------------------------------------------
# VecRec black-box header generation (adapted from evals_common/vecrec_cpp.py).
# --------------------------------------------------------------------------

class VecrecKernel:
    """A compiled VecRec kernel header + the metadata a driver needs to call it."""

    def __init__(self, header_path: str, kernel_name: str, epi: int, rpi: int,
                 out_cpp_type: str, s_cpp_type: str, computed_ctor: bool,
                 make_args: str, resolved_type: str) -> None:
        self.header_path = header_path
        self.kernel_name = kernel_name
        self.epi = epi
        self.rpi = rpi
        self.out_cpp_type = out_cpp_type
        self.s_cpp_type = s_cpp_type
        self.computed_ctor = computed_ctor
        self.make_args = make_args
        self.resolved_type = resolved_type


def candidate_schedules(kind: str) -> list:
    """All lane-annotated candidate schedules for a recurrence (the black-box variants)."""
    from vecrec.optimize import get_candidate_schedules
    from vecrec.util import get_native_bits

    expr = KINDS[kind]["expr"]()
    try:
        max_bits = get_native_bits()
    except OSError:
        max_bits = 256
    return list(get_candidate_schedules(expr, max_bits))


def build_vecrec_kernel(scheduled_expr, ele_per_iter_multiple: int) -> Optional[VecrecKernel]:
    """Generate a VecRec kernel header (the tile primitive). Returns metadata or None."""
    try:
        from vecrec.codegen import CodeGen, instantiate_kernels

        kernel_name = "k0"
        codegen = CodeGen(band_width=0, ele_per_iter_multiple=ele_per_iter_multiple)
        with contextlib.redirect_stdout(_io.StringIO()):
            code = codegen.generate(scheduled_expr, kernel_name)
        out_elt = scheduled_expr.element_type

        fd, header_path = tempfile.mkstemp(suffix=".h", prefix="reptile_hdr_", dir="/tmp")
        os.close(fd)
        instantiate_kernels(header_path, [code])

        computed_ctor = False
        make_args = ""
        resolved_type = ""
        # Both NW and affine lower their Var2D("s") input to InputSignal2D; patch it
        # to ProfileSignal2D so the driver can feed a small match/mismatch profile
        # (the current vec-rec KernelProgram has no ComputedSignal2D path).
        import re as _re

        _, vars2d = codegen.collect_input_variables(scheduled_expr)
        svar = next(iter(vars2d.keys()))
        hdr = Path(header_path).read_text()
        hdr = hdr.replace("InputSignal2D<", "ProfileSignal2D<")
        hdr = hdr.replace(f"({svar}, {svar}_stride,", f"({svar}, {svar}_sym, {svar}_stride,")
        hdr = _re.sub(
            rf"\*{svar}, int {svar}_stride",
            f"*{svar}, const int *{svar}_sym, int {svar}_stride",
            hdr,
        )
        Path(header_path).write_text(hdr)

        return VecrecKernel(
            header_path=header_path,
            kernel_name=kernel_name,
            epi=code.element_per_iter,
            rpi=code.rows_per_iter,
            out_cpp_type=out_elt.to_str(),
            s_cpp_type=out_elt.to_str(),
            computed_ctor=computed_ctor,
            make_args=make_args,
            resolved_type=resolved_type,
        )
    except Exception:
        return None


# --------------------------------------------------------------------------
# Common C++ prologue shared by all generated drivers.
# --------------------------------------------------------------------------

_ALIGNED64 = """template <class T> struct Aligned64 {
    using value_type = T;
    Aligned64() = default;
    template <class U> Aligned64(const Aligned64<U>&) {}
    T* allocate(std::size_t n) {
        return static_cast<T*>(::operator new[](n * sizeof(T), std::align_val_t(64)));
    }
    void deallocate(T* p, std::size_t) { ::operator delete[](p, std::align_val_t(64)); }
    template <class U> bool operator==(const Aligned64<U>&) const { return true; }
    template <class U> bool operator!=(const Aligned64<U>&) const { return false; }
};"""


_READ_TILING = """// REPTILE schedule (read at run time so one binary covers the whole sweep).
        const std::string _raw = baselines::detail::read_file(path);
        const std::string approach = baselines::detail::get_string(_raw, "approach");
        const int tile_rows_in = static_cast<int>(baselines::detail::get_int_or(_raw, "tile_rows", 0));
        const int tile_cols_in = static_cast<int>(baselines::detail::get_int_or(_raw, "tile_cols", 0));
        const int verify = static_cast<int>(baselines::detail::get_int_or(_raw, "verify", 0));
        // score_only: never materialize the O(N^2) matrix -> no matrix writeback to
        // DRAM. The kernel still runs every cell (its ring buffer holds the live
        // wavefront), but outputs go to a reused O(cols) scratch row and we keep
        // only the final score (corner for nw/affine, running max for sw).
        const int score_only = static_cast<int>(baselines::detail::get_int_or(_raw, "score_only", 0));"""


def _checksum_block(matrix_var: str) -> str:
    """C++ that FNV-1a-hashes the full logical DP matrix into `_checksum` when verify is set.

    Two configs with equal checksums over the full [0..rows]x[0..cols] matrix have
    bit-identical matrices -> proof that tiling only reorders the traversal.
    """
    return f"""if (verify) {{
                unsigned long long _h = 1469598103934665603ULL;
                for (int _i = 0; _i <= rows; ++_i)
                    for (int _j = 0; _j <= cols; ++_j) {{
                        _h ^= static_cast<unsigned long long>(
                            static_cast<long long>({matrix_var}[static_cast<std::size_t>(_i) * phys_w + _j]));
                        _h *= 1099511628211ULL;
                    }}
                _checksum = _h;
            }}"""


# --------------------------------------------------------------------------
# Driver family 1: VecRec kernel as the mapped external tile primitive.
# --------------------------------------------------------------------------

def vecrec_driver_source(kind: str, k: VecrecKernel, driver_name: str) -> str:
    cfg = KINDS[kind]
    if k.computed_ctor:
        input_inside = _HARNESS_SUBST_SETUP
        kernel_ctor = f"{k.kernel_name}({k.resolved_type}({k.make_args}), phys_w)"
    else:
        input_inside = f"""auto _sym = [](char ch) -> int {{
                switch (ch) {{ case 'A': case 'a': return 0; case 'C': case 'c': return 1;
                               case 'G': case 'g': return 2; case 'T': case 't': return 3;
                               default: return 0; }}
            }};
            std::vector<{k.s_cpp_type}> profile(static_cast<std::size_t>(4) * phys_w, 0);
            for (int cc = 0; cc < 4; ++cc)
                for (int j = 0; j < cols; ++j)
                    profile[static_cast<std::size_t>(cc) * phys_w + (j + 1)] =
                        static_cast<{k.s_cpp_type}>(cc == _sym(in.y[j]) ? in.match : in.mismatch);
            std::vector<int> sym(static_cast<std::size_t>(R), 0);
            for (int i = 0; i < rows; ++i) sym[i + 1] = _sym(in.x[i]);"""
        kernel_ctor = f"{k.kernel_name}(profile.data(), sym.data(), phys_w, phys_w)"

    # score_only capture: fold each just-written EPI-vector into `score` instead of
    # keeping the full matrix. Global (nw/affine) => the corner cell H[rows][cols];
    # local (sw) => a running max over every cell.
    OUT = k.out_cpp_type
    if cfg.get("score_only") == "max":
        capture_body = (
            f"const {OUT}* _s = reinterpret_cast<const {OUT}*>(opr[rr]);\n"
            "                    for (int _e = 0; _e < EPI; ++_e) {\n"
            "                        const long _v = static_cast<long>(_s[_e]);\n"
            "                        if (_v > score) score = _v;\n"
            "                    }"
        )
    else:  # "corner"
        capture_body = (
            "if (lr == rows && cols >= col && cols < col + EPI) {\n"
            f"                        const {OUT}* _s = reinterpret_cast<const {OUT}*>(opr[rr]);\n"
            "                        score = static_cast<long>(_s[cols - col]);\n"
            "                    }"
        )

    return f"""// Auto-generated by Reptile/reptile.py. Do not edit by hand.
// VecRec kernel as the mapped external tile primitive (REPTILE external mapping).
#include "{k.header_path}"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <new>

#include "driver_io.hpp"
#include "matrix_traceback.hpp"

{_ALIGNED64}

using baselines::Input;
using baselines::TimedRun;
using baselines::Timer;

int main(int argc, char** argv) {{
    return baselines::run_driver(argc, argv, "{driver_name}", [](const char* path) -> int {{
        const Input in = baselines::parse_input(path);
        const int rows = static_cast<int>(in.x.size());
        const int cols = static_cast<int>(in.y.size());
        const long cells = static_cast<long>(rows) * static_cast<long>(cols);
        {cfg["gap_setup"]}
        {_READ_TILING}

        constexpr int EPI = {k.epi};
        constexpr int RPI = {k.rpi};
        const int R = rows + 1;
        const int C = cols + 1;
        const int phys_w = ((C + EPI - 1) / EPI) * EPI;
        const long N = static_cast<long>(R) * phys_w;

        // Row-tile height (rows). row/block => tile_rows (rounded up to a whole
        // number of kernel row-blocks RPI); untiled/col => whole matrix.
        int rows_per_tile = R;
        if ((approach == "row" || approach == "block") && tile_rows_in > 0)
            rows_per_tile = tile_rows_in;
        rows_per_tile = ((rows_per_tile + RPI - 1) / RPI) * RPI;
        if (rows_per_tile < RPI) rows_per_tile = RPI;

        // Column-panel width (cols). col/block => tile_cols (rounded up to a whole
        // number of SIMD groups EPI); untiled/row => whole width.
        int cols_per_tile = phys_w;
        if ((approach == "col" || approach == "block") && tile_cols_in > 0)
            cols_per_tile = ((tile_cols_in + EPI - 1) / EPI) * EPI;
        if (cols_per_tile < EPI) cols_per_tile = EPI;
        if (cols_per_tile > phys_w) cols_per_tile = phys_w;
        const bool col_tiled = (approach == "col" || approach == "block");

        long score = 0;
        unsigned long long _checksum = 0;
        baselines::AlignmentTraceback alignment;

        // Alloc once outside the timer so O(N^2) zero-init / traceback do not
        // dilute tiling speedups (same discipline as the loops family).
        {input_inside}
        std::vector<{k.out_cpp_type}, Aligned64<{k.out_cpp_type}>> output;
        std::vector<{k.out_cpp_type}, Aligned64<{k.out_cpp_type}>> scratch;
        if (score_only) {{
            scratch.assign(static_cast<std::size_t>(RPI) * phys_w + static_cast<std::size_t>(phys_w), 0);
        }} else {{
            output.assign(static_cast<std::size_t>(N) + static_cast<std::size_t>(RPI) * phys_w, 0);
        }}
        auto kernel = {kernel_ctor};
        const std::size_t state_stride = kernel.col_state_bytes();
        std::vector<char> colsave(
            static_cast<std::size_t>(rows_per_tile + RPI) * state_stride);
        std::vector<char> q(static_cast<std::size_t>(rows + cols + 1));
        std::vector<char> r(static_cast<std::size_t>(rows + cols + 1));
        std::vector<char> c(static_cast<std::size_t>(rows + cols + 1));

        // score_only: fold each just-written vector into `score` (see capture body).
        auto _capture = [&](long lr0, int col, {k.kernel_name}_vector_type* const* opr) {{
            for (int rr = 0; rr < RPI; ++rr) {{
                const long lr = lr0 + rr; (void)lr;
                {capture_body}
            }}
        }};

        Timer timer;
        const TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {{
            if (score_only) score = 0;
            // Re-construct kernel state each iter (cheap vs O(N^2) matrix fill).
            kernel = {kernel_ctor};

            if (!col_tiled) {{
                // ---- REPTILE row-strip tiling: one continuous VecRec black box -----
                long row_base = 0;
                for (int tr0 = 0; tr0 < R; tr0 += rows_per_tile) {{
                    const long strip_end = std::min<long>(N, static_cast<long>(tr0 + rows_per_tile) * phys_w);
                    while (row_base < strip_end) {{
                        int col = 0;
                        kernel.seek_col(0);
                        while (col < phys_w) {{
                            {k.kernel_name}_vector_type* opr[RPI];
                            for (int rr = 0; rr < RPI; ++rr) {{
                                {k.out_cpp_type}* _dst = score_only
                                    ? (scratch.data() + static_cast<std::size_t>(rr) * phys_w + col)
                                    : (output.data() + static_cast<std::size_t>(row_base)
                                       + static_cast<std::size_t>(rr) * phys_w + col);
                                opr[rr] = reinterpret_cast<{k.kernel_name}_vector_type*>(_dst);
                            }}
                            kernel.run(opr);
                            if (score_only) _capture(row_base / phys_w, col, opr);
                            col += EPI;
                        }}
                        for (int rr = 0; rr < RPI; ++rr) kernel.reset_and_next_row();
                        row_base += static_cast<long>(RPI) * phys_w;
                    }}
                }}
            }} else {{
                // ---- REPTILE column-panel / 2D-block tiling of the VecRec black box -
                for (int bi0 = 0; bi0 < R; bi0 += rows_per_tile) {{
                    const int bi1 = static_cast<int>(std::min<long>(N / phys_w,
                                        static_cast<long>(bi0) + rows_per_tile));
                    for (int cj0 = 0; cj0 < phys_w; cj0 += cols_per_tile) {{
                        const int cj1 = std::min(phys_w, cj0 + cols_per_tile);
                        kernel.seek_row(bi0);
                        for (int i = bi0; i < bi1; i += RPI) {{
                            char *sp = colsave.data()
                                + static_cast<std::size_t>(i - bi0) * state_stride;
                            kernel.seek_col(cj0);
                            if (cj0 > 0) {{ const char *rp = sp; kernel.restore_col_state(rp); }}
                            int col = cj0;
                            while (col < cj1) {{
                                {k.kernel_name}_vector_type* opr[RPI];
                                for (int rr = 0; rr < RPI; ++rr) {{
                                    {k.out_cpp_type}* _dst = score_only
                                        ? (scratch.data() + static_cast<std::size_t>(rr) * phys_w + col)
                                        : (output.data() + static_cast<std::size_t>(i) * phys_w
                                           + static_cast<std::size_t>(rr) * phys_w + col);
                                    opr[rr] = reinterpret_cast<{k.kernel_name}_vector_type*>(_dst);
                                }}
                                kernel.run(opr);
                                if (score_only) _capture(i, col, opr);
                                col += EPI;
                            }}
                            if (cj1 < phys_w) {{ char *wp = sp; kernel.save_col_state(wp); }}
                            for (int rr = 0; rr < RPI; ++rr) kernel.reset_and_next_row();
                        }}
                    }}
                }}
            }}
        }});

        if (!score_only) {{
            {_checksum_block("output")}
            {cfg["vecrec_traceback"]}
            alignment.query = q.data();
            alignment.ref = r.data();
            alignment.comp = c.data();
        }}

        if (verify) std::fprintf(stderr, "REPTILE_CHECKSUM %llu\\n", _checksum);
        baselines::print_result(score, timed, cells, alignment);
        return 0;
    }});
}}
"""


# --------------------------------------------------------------------------
# Driver family 2: REPTILE-generated tiled loops (no external mapping).
# --------------------------------------------------------------------------

# Split into: alloc (once, outside timer) / reset+DP (timed); allocation and
# traceback are not timed.
#
# The NW/SW timed path is O(N) score-only (2-row / col-panel / block-panel).
# Full-matrix nests (SEQ nw.cpp / sw.c style) are used for --verify so
# bit-identity checks still work.

_NW_LOOPS_ALLOC = """const char* xs = in.x.c_str();
        const char* ys = in.y.c_str();
        // Allocate ONLY the buffers the chosen approach needs. Pre-allocating
        // every O(N) array (~6 full rows) pollutes L1/L2 and can erase the
        // column-tiling win (standalone microbench keeps just strip+left_edge).
        std::vector<int32_t, Aligned64<int32_t>> prev, curr, strip, left_edge, top_row, panel, mat;
        const int stride = phys_w;
        if (verify) {
            mat.assign(static_cast<std::size_t>(N), 0);
        } else if (approach == "untiled" || approach == "row") {
            prev.assign(static_cast<std::size_t>(C), 0);
            curr.assign(static_cast<std::size_t>(C), 0);
        } else if (approach == "col") {
            strip.assign(static_cast<std::size_t>(TC) + 2, 0);
            left_edge.assign(static_cast<std::size_t>(R), 0);
        } else { // block
            top_row.assign(static_cast<std::size_t>(C), 0);
            panel.assign(static_cast<std::size_t>(TC) + 2, 0);
            left_edge.assign(static_cast<std::size_t>(R), 0);
        }"""

_NW_LOOPS_TIMED = """if (!verify) {
                // ---- O(N) score-only: tiling shrinks the live working set ----
                // Branchless match/mismatch so we measure locality, not branch-pred
                // artifacts from periodic DNA patterns (those can fake ~1.3x).
                const int32_t match = in.match;
                const int32_t mism = in.mismatch;
                if (approach == "untiled" || approach == "row") {
                    // Row tiling is a no-op on a 2-row form (same nest).
                    for (int j = 0; j <= cols; ++j) prev[j] = j * gap;
                    for (int i = 1; i <= rows; ++i) {
                        curr[0] = i * gap;
                        int32_t left = curr[0];
                        for (int j = 1; j <= cols; ++j) {
                            const int32_t eq = (xs[i - 1] == ys[j - 1]);
                                const int32_t sub = match * eq + mism * (1 - eq);
                            int32_t d = prev[j - 1] + sub;
                            int32_t u = prev[j] + gap;
                            int32_t l = left + gap;
                            int32_t m = d > u ? d : u;
                            m = m > l ? m : l;
                            left = m;
                            curr[j] = m;
                        }
                        prev.swap(curr);
                    }
                    score = prev[cols];
                } else if (approach == "col") {
                    // Column panels: strip of width TC stays L1-hot.
                    // left_edge[i] = H[i][bj-1] entering a panel; overwritten with
                    // H[i][j1] after each row. prev_left carries H[i-1][bj-1] so we
                    // don't need a second O(N) right_edge buffer.
                    for (int i = 0; i <= rows; ++i) left_edge[i] = i * gap;
                    for (int bj = 1; bj <= cols; bj += TC) {
                        const int j1 = std::min(cols, bj + TC - 1);
                        const int tw = j1 - bj + 1;
                        int32_t prev_left = (bj - 1) * gap;  // H[0][bj-1]
                        for (int jj = 0; jj < tw; ++jj) strip[jj] = (bj + jj) * gap;
                        for (int i = 1; i <= rows; ++i) {
                            int32_t diag = prev_left;          // H[i-1][bj-1]
                            int32_t left = left_edge[i];       // H[i][bj-1]
                            prev_left = left;                  // for row i+1
                            for (int jj = 0; jj < tw; ++jj) {
                                const int j = bj + jj;
                                const int32_t up = strip[jj];
                                const int32_t eq = (xs[i - 1] == ys[j - 1]);
                                const int32_t sub = match * eq + mism * (1 - eq);
                                int32_t d = diag + sub;
                                int32_t u = up + gap;
                                int32_t l = left + gap;
                                int32_t m = d > u ? d : u;
                                m = m > l ? m : l;
                                diag = up;
                                left = m;
                                strip[jj] = m;
                            }
                            left_edge[i] = left;  // H[i][j1] for next panel
                        }
                    }
                    score = left_edge[rows];
                } else { // block: row-bands x col-panels (SEQ genNWtile shape, O(N) storage)
                    for (int j = 0; j <= cols; ++j) top_row[j] = j * gap;
                    for (int bi = 1; bi <= rows; bi += TR) {
                        const int i1 = std::min(rows, bi + TR - 1);
                        // Left boundary of this row-band starts at column 0.
                        for (int i = bi - 1; i <= i1; ++i) left_edge[i] = i * gap;
                        for (int bj = 1; bj <= cols; bj += TC) {
                            const int j1 = std::min(cols, bj + TC - 1);
                            for (int j = bj; j <= j1; ++j) panel[j - bj] = top_row[j];
                            int32_t prev_left = left_edge[bi - 1];  // H[bi-1][bj-1]
                            for (int i = bi; i <= i1; ++i) {
                                int32_t diag = prev_left;
                                int32_t left = left_edge[i];
                                prev_left = left;
                                for (int j = bj; j <= j1; ++j) {
                                    const int jj = j - bj;
                                    const int32_t up = panel[jj];
                                    const int32_t eq = (xs[i - 1] == ys[j - 1]);
                                const int32_t sub = match * eq + mism * (1 - eq);
                                    int32_t d = diag + sub;
                                    int32_t u = up + gap;
                                    int32_t l = left + gap;
                                    int32_t m = d > u ? d : u;
                                    m = m > l ? m : l;
                                    diag = up;
                                    left = m;
                                    panel[jj] = m;
                                }
                                left_edge[i] = left;  // H[i][j1]
                            }
                            const int32_t top_at_j1 = top_row[j1];
                            for (int j = bj; j <= j1; ++j) top_row[j] = panel[j - bj];
                            left_edge[bi - 1] = top_at_j1;  // for next panel
                        }
                    }
                    score = top_row[cols];
                }
            } else {
                // ---- Full-matrix path (verify / non-score_only): SEQ-style nest ----
                for (int j = 0; j <= cols; ++j) mat[j] = j * gap;
                for (int i = 0; i <= rows; ++i) mat[static_cast<std::size_t>(i) * stride] = i * gap;
                auto cell = [&](int i, int j) __attribute__((always_inline)) {
                    const std::size_t p = static_cast<std::size_t>(i) * stride + j;
                    const std::size_t pin1 = static_cast<std::size_t>(i - 1) * stride;
                    const int32_t Ain1jn1 = mat[pin1 + (j - 1)];
                    const int32_t Cij = (xs[i - 1] == ys[j - 1]) ? in.match : in.mismatch;
                    const int32_t A3ij = Ain1jn1 + Cij;
                    mat[p] = A3ij;
                    const int32_t Ain1j = mat[pin1 + j];
                    const int32_t A1ij = Ain1j + gap;
                    mat[p] = A1ij;
                    const int32_t Aijn1 = mat[p - 1];
                    const int32_t A0ij = Aijn1 + gap;
                    const int32_t A2ij = A1ij > A0ij ? A1ij : A0ij;
                    mat[p] = A3ij > A2ij ? A3ij : A2ij;
                };
                if (approach == "untiled") {
                    for (int i = 1; i <= rows; ++i)
                        for (int j = 1; j <= cols; ++j) cell(i, j);
                } else if (approach == "row") {
                    for (int bi = 1; bi <= rows; bi += TR)
                        for (int i = bi; i < bi + TR && i <= rows; ++i)
                            for (int j = 1; j <= cols; ++j) cell(i, j);
                } else if (approach == "col") {
                    for (int bj = 1; bj <= cols; bj += TC)
                        for (int i = 1; i <= rows; ++i)
                            for (int j = bj; j < bj + TC && j <= cols; ++j) cell(i, j);
                } else {
                    for (int bi = 1; bi <= rows; bi += TR)
                        for (int bj = 1; bj <= cols; bj += TC)
                            for (int i = bi; i < bi + TR && i <= rows; ++i)
                                for (int j = bj; j < bj + TC && j <= cols; ++j) cell(i, j);
                }
            }"""


_AFFINE_LOOPS_ALLOC = """const int stride = phys_w;
        const int NEG = -1000000000;  // finite -inf sentinel (scores stay well above)
        // One-time NEG fill: DP visits every interior cell; only boundaries are
        // reset each timed iteration (avoids O(N^2) fill inside the timer).
        std::vector<int32_t, Aligned64<int32_t>> Mm(static_cast<std::size_t>(N), NEG);
        std::vector<int32_t, Aligned64<int32_t>> Dx(static_cast<std::size_t>(N), NEG);
        std::vector<int32_t, Aligned64<int32_t>> Dy(static_cast<std::size_t>(N), NEG);
        std::vector<int32_t, Aligned64<int32_t>> H(static_cast<std::size_t>(N), NEG);
        const int GO_GE = go + ge;
        const char* xs = in.x.c_str();
        const char* ys = in.y.c_str();
        auto smax3 = [](int32_t a, int32_t b, int32_t c) __attribute__((always_inline)) {
            int32_t m = a > b ? a : b; return m > c ? m : c;
        };"""

_AFFINE_LOOPS_TIMED = """Mm[0] = 0;
            for (int i = 1; i <= rows; ++i) Dx[static_cast<std::size_t>(i) * stride] = go + i * ge;
            for (int j = 1; j <= cols; ++j) Dy[j] = go + j * ge;
            auto cell = [&](int i, int j) __attribute__((always_inline)) {
                const std::size_t p = static_cast<std::size_t>(i) * stride + j;
                const std::size_t up = static_cast<std::size_t>(i - 1) * stride + j;
                const std::size_t lf = static_cast<std::size_t>(i) * stride + (j - 1);
                const std::size_t dg = static_cast<std::size_t>(i - 1) * stride + (j - 1);
                const int32_t sub = (xs[i - 1] == ys[j - 1]) ? in.match : in.mismatch;
                const int32_t best_diag = smax3(Mm[dg], Dx[dg], Dy[dg]);
                Mm[p] = best_diag + sub;
                Dx[p] = (Mm[up] + GO_GE) > (Dx[up] + ge) ? (Mm[up] + GO_GE) : (Dx[up] + ge);
                Dy[p] = (Mm[lf] + GO_GE) > (Dy[lf] + ge) ? (Mm[lf] + GO_GE) : (Dy[lf] + ge);
                H[p] = smax3(Mm[p], Dx[p], Dy[p]);
            };
            if (approach == "untiled") {
                for (int i = 1; i <= rows; ++i)
                    for (int j = 1; j <= cols; ++j) cell(i, j);
            } else if (approach == "row") {
                for (int bi = 1; bi <= rows; bi += TR)
                    for (int i = bi; i < bi + TR && i <= rows; ++i)
                        for (int j = 1; j <= cols; ++j) cell(i, j);
            } else if (approach == "col") {
                for (int bj = 1; bj <= cols; bj += TC)
                    for (int i = 1; i <= rows; ++i)
                        for (int j = bj; j < bj + TC && j <= cols; ++j) cell(i, j);
            } else { // block (2D)
                for (int bi = 1; bi <= rows; bi += TR)
                    for (int bj = 1; bj <= cols; bj += TC)
                        for (int i = bi; i < bi + TR && i <= rows; ++i)
                            for (int j = bj; j < bj + TC && j <= cols; ++j) cell(i, j);
            }"""


_SW_LOOPS_ALLOC = """const char* xs = in.x.c_str();
        const char* ys = in.y.c_str();
        // Approach-local buffers only (see NW comment) — unused full-width
        // vectors erase the L1 win from column tiling.
        std::vector<int32_t, Aligned64<int32_t>> prev, curr, strip, left_edge, top_row, panel, mat;
        const int stride = phys_w;
        if (verify) {
            mat.assign(static_cast<std::size_t>(N), 0);
        } else if (approach == "untiled" || approach == "row") {
            prev.assign(static_cast<std::size_t>(C), 0);
            curr.assign(static_cast<std::size_t>(C), 0);
        } else if (approach == "col") {
            strip.assign(static_cast<std::size_t>(TC) + 2, 0);
            left_edge.assign(static_cast<std::size_t>(R), 0);
        } else { // block
            top_row.assign(static_cast<std::size_t>(C), 0);
            panel.assign(static_cast<std::size_t>(TC) + 2, 0);
            left_edge.assign(static_cast<std::size_t>(R), 0);
        }"""

_LOOPS_ALLOC = """const char* xs = in.x.c_str();
        const char* ys = in.y.c_str();
        // Allocate ONLY the buffers the chosen approach needs. Pre-allocating
        // every O(N) array (~6 full rows) pollutes L1/L2 and can erase the
        // column-tiling win (standalone microbench keeps just strip+left_edge).
        std::vector<int32_t, Aligned64<int32_t>> prev, curr, strip, left_edge, top_row, panel, mat;
        const int stride = phys_w;
        if (verify) {
            mat.assign(static_cast<std::size_t>(N), 0);
        } else if (approach == "untiled" || approach == "row") {
            prev.assign(static_cast<std::size_t>(C), 0);
            curr.assign(static_cast<std::size_t>(C), 0);
        } else if (approach == "col") {
            strip.assign(static_cast<std::size_t>(TC) + 2, 0);
            left_edge.assign(static_cast<std::size_t>(R), 0);
        } else { // block
            top_row.assign(static_cast<std::size_t>(C), 0);
            panel.assign(static_cast<std::size_t>(TC) + 2, 0);
            left_edge.assign(static_cast<std::size_t>(R), 0);
        }"""

_NW_LOOPS_TIMED = """if (!verify) {
                // ---- O(N) score-only: tiling shrinks the live working set ----
                // Branchless match/mismatch so we measure locality, not branch-pred
                // artifacts from periodic DNA patterns (those can fake ~1.3x).
                const int32_t match = in.match;
                const int32_t mism = in.mismatch;
                if (approach == "untiled" || approach == "row") {
                    // Row tiling is a no-op on a 2-row form (same nest).
                    for (int j = 0; j <= cols; ++j) prev[j] = j * gap;
                    for (int i = 1; i <= rows; ++i) {
                        curr[0] = i * gap;
                        int32_t left = curr[0];
                        for (int j = 1; j <= cols; ++j) {
                            const int32_t eq = (xs[i - 1] == ys[j - 1]);
                                const int32_t sub = match * eq + mism * (1 - eq);
                            int32_t d = prev[j - 1] + sub;
                            int32_t u = prev[j] + gap;
                            int32_t l = left + gap;
                            int32_t m = d > u ? d : u;
                            m = m > l ? m : l;
                            left = m;
                            curr[j] = m;
                        }
                        prev.swap(curr);
                    }
                    score = prev[cols];
                } else if (approach == "col") {
                    // Column panels: strip of width TC stays L1-hot.
                    // left_edge[i] = H[i][bj-1] entering a panel; overwritten with
                    // H[i][j1] after each row. prev_left carries H[i-1][bj-1] so we
                    // don't need a second O(N) right_edge buffer.
                    for (int i = 0; i <= rows; ++i) left_edge[i] = i * gap;
                    for (int bj = 1; bj <= cols; bj += TC) {
                        const int j1 = std::min(cols, bj + TC - 1);
                        const int tw = j1 - bj + 1;
                        int32_t prev_left = (bj - 1) * gap;  // H[0][bj-1]
                        for (int jj = 0; jj < tw; ++jj) strip[jj] = (bj + jj) * gap;
                        for (int i = 1; i <= rows; ++i) {
                            int32_t diag = prev_left;          // H[i-1][bj-1]
                            int32_t left = left_edge[i];       // H[i][bj-1]
                            prev_left = left;                  // for row i+1
                            for (int jj = 0; jj < tw; ++jj) {
                                const int j = bj + jj;
                                const int32_t up = strip[jj];
                                const int32_t eq = (xs[i - 1] == ys[j - 1]);
                                const int32_t sub = match * eq + mism * (1 - eq);
                                int32_t d = diag + sub;
                                int32_t u = up + gap;
                                int32_t l = left + gap;
                                int32_t m = d > u ? d : u;
                                m = m > l ? m : l;
                                diag = up;
                                left = m;
                                strip[jj] = m;
                            }
                            left_edge[i] = left;  // H[i][j1] for next panel
                        }
                    }
                    score = left_edge[rows];
                } else { // block: row-bands x col-panels (SEQ genNWtile shape, O(N) storage)
                    for (int j = 0; j <= cols; ++j) top_row[j] = j * gap;
                    for (int bi = 1; bi <= rows; bi += TR) {
                        const int i1 = std::min(rows, bi + TR - 1);
                        // Left boundary of this row-band starts at column 0.
                        for (int i = bi - 1; i <= i1; ++i) left_edge[i] = i * gap;
                        for (int bj = 1; bj <= cols; bj += TC) {
                            const int j1 = std::min(cols, bj + TC - 1);
                            for (int j = bj; j <= j1; ++j) panel[j - bj] = top_row[j];
                            int32_t prev_left = left_edge[bi - 1];  // H[bi-1][bj-1]
                            for (int i = bi; i <= i1; ++i) {
                                int32_t diag = prev_left;
                                int32_t left = left_edge[i];
                                prev_left = left;
                                for (int j = bj; j <= j1; ++j) {
                                    const int jj = j - bj;
                                    const int32_t up = panel[jj];
                                    const int32_t eq = (xs[i - 1] == ys[j - 1]);
                                const int32_t sub = match * eq + mism * (1 - eq);
                                    int32_t d = diag + sub;
                                    int32_t u = up + gap;
                                    int32_t l = left + gap;
                                    int32_t m = d > u ? d : u;
                                    m = m > l ? m : l;
                                    diag = up;
                                    left = m;
                                    panel[jj] = m;
                                }
                                left_edge[i] = left;  // H[i][j1]
                            }
                            const int32_t top_at_j1 = top_row[j1];
                            for (int j = bj; j <= j1; ++j) top_row[j] = panel[j - bj];
                            left_edge[bi - 1] = top_at_j1;  // for next panel
                        }
                    }
                    score = top_row[cols];
                }
            } else {
                // ---- Full-matrix path (verify / non-score_only): SEQ-style nest ----
                for (int j = 0; j <= cols; ++j) mat[j] = j * gap;
                for (int i = 0; i <= rows; ++i) mat[static_cast<std::size_t>(i) * stride] = i * gap;
                auto cell = [&](int i, int j) __attribute__((always_inline)) {
                    const std::size_t p = static_cast<std::size_t>(i) * stride + j;
                    const std::size_t pin1 = static_cast<std::size_t>(i - 1) * stride;
                    const int32_t Ain1jn1 = mat[pin1 + (j - 1)];
                    const int32_t Cij = (xs[i - 1] == ys[j - 1]) ? in.match : in.mismatch;
                    const int32_t A3ij = Ain1jn1 + Cij;
                    mat[p] = A3ij;
                    const int32_t Ain1j = mat[pin1 + j];
                    const int32_t A1ij = Ain1j + gap;
                    mat[p] = A1ij;
                    const int32_t Aijn1 = mat[p - 1];
                    const int32_t A0ij = Aijn1 + gap;
                    const int32_t A2ij = A1ij > A0ij ? A1ij : A0ij;
                    mat[p] = A3ij > A2ij ? A3ij : A2ij;
                };
                if (approach == "untiled") {
                    for (int i = 1; i <= rows; ++i)
                        for (int j = 1; j <= cols; ++j) cell(i, j);
                } else if (approach == "row") {
                    for (int bi = 1; bi <= rows; bi += TR)
                        for (int i = bi; i < bi + TR && i <= rows; ++i)
                            for (int j = 1; j <= cols; ++j) cell(i, j);
                } else if (approach == "col") {
                    for (int bj = 1; bj <= cols; bj += TC)
                        for (int i = 1; i <= rows; ++i)
                            for (int j = bj; j < bj + TC && j <= cols; ++j) cell(i, j);
                } else {
                    for (int bi = 1; bi <= rows; bi += TR)
                        for (int bj = 1; bj <= cols; bj += TC)
                            for (int i = bi; i < bi + TR && i <= rows; ++i)
                                for (int j = bj; j < bj + TC && j <= cols; ++j) cell(i, j);
                }
            }"""


_AFFINE_LOOPS_ALLOC = """const int stride = phys_w;
        const int NEG = -1000000000;  // finite -inf sentinel (scores stay well above)
        // One-time NEG fill: DP visits every interior cell; only boundaries are
        // reset each timed iteration (avoids O(N^2) fill inside the timer).
        std::vector<int32_t, Aligned64<int32_t>> Mm(static_cast<std::size_t>(N), NEG);
        std::vector<int32_t, Aligned64<int32_t>> Dx(static_cast<std::size_t>(N), NEG);
        std::vector<int32_t, Aligned64<int32_t>> Dy(static_cast<std::size_t>(N), NEG);
        std::vector<int32_t, Aligned64<int32_t>> H(static_cast<std::size_t>(N), NEG);
        const int GO_GE = go + ge;
        const char* xs = in.x.c_str();
        const char* ys = in.y.c_str();
        auto smax3 = [](int32_t a, int32_t b, int32_t c) __attribute__((always_inline)) {
            int32_t m = a > b ? a : b; return m > c ? m : c;
        };"""

_AFFINE_LOOPS_TIMED = """Mm[0] = 0;
            for (int i = 1; i <= rows; ++i) Dx[static_cast<std::size_t>(i) * stride] = go + i * ge;
            for (int j = 1; j <= cols; ++j) Dy[j] = go + j * ge;
            auto cell = [&](int i, int j) __attribute__((always_inline)) {
                const std::size_t p = static_cast<std::size_t>(i) * stride + j;
                const std::size_t up = static_cast<std::size_t>(i - 1) * stride + j;
                const std::size_t lf = static_cast<std::size_t>(i) * stride + (j - 1);
                const std::size_t dg = static_cast<std::size_t>(i - 1) * stride + (j - 1);
                const int32_t sub = (xs[i - 1] == ys[j - 1]) ? in.match : in.mismatch;
                const int32_t best_diag = smax3(Mm[dg], Dx[dg], Dy[dg]);
                Mm[p] = best_diag + sub;
                Dx[p] = (Mm[up] + GO_GE) > (Dx[up] + ge) ? (Mm[up] + GO_GE) : (Dx[up] + ge);
                Dy[p] = (Mm[lf] + GO_GE) > (Dy[lf] + ge) ? (Mm[lf] + GO_GE) : (Dy[lf] + ge);
                H[p] = smax3(Mm[p], Dx[p], Dy[p]);
            };
            if (approach == "untiled") {
                for (int i = 1; i <= rows; ++i)
                    for (int j = 1; j <= cols; ++j) cell(i, j);
            } else if (approach == "row") {
                for (int bi = 1; bi <= rows; bi += TR)
                    for (int i = bi; i < bi + TR && i <= rows; ++i)
                        for (int j = 1; j <= cols; ++j) cell(i, j);
            } else if (approach == "col") {
                for (int bj = 1; bj <= cols; bj += TC)
                    for (int i = 1; i <= rows; ++i)
                        for (int j = bj; j < bj + TC && j <= cols; ++j) cell(i, j);
            } else { // block (2D)
                for (int bi = 1; bi <= rows; bi += TR)
                    for (int bj = 1; bj <= cols; bj += TC)
                        for (int i = bi; i < bi + TR && i <= rows; ++i)
                            for (int j = bj; j < bj + TC && j <= cols; ++j) cell(i, j);
            }"""


_SW_LOOPS_ALLOC = """const char* xs = in.x.c_str();
        const char* ys = in.y.c_str();
        // Approach-local buffers only (see NW comment) — unused full-width
        // vectors erase the L1 win from column tiling.
        std::vector<int32_t, Aligned64<int32_t>> prev, curr, strip, left_edge, top_row, panel, mat;
        const int stride = phys_w;
        if (verify) {
            mat.assign(static_cast<std::size_t>(N), 0);
        } else if (approach == "untiled" || approach == "row") {
            prev.assign(static_cast<std::size_t>(C), 0);
            curr.assign(static_cast<std::size_t>(C), 0);
        } else if (approach == "col") {
            strip.assign(static_cast<std::size_t>(TC) + 2, 0);
            left_edge.assign(static_cast<std::size_t>(R), 0);
        } else { // block
            top_row.assign(static_cast<std::size_t>(C), 0);
            panel.assign(static_cast<std::size_t>(TC) + 2, 0);
            left_edge.assign(static_cast<std::size_t>(R), 0);
        }"""

_SW_LOOPS_TIMED = """if (!verify) {
                // ---- O(N) local SW: running max; col/block keep the strip L1-hot ----
                const int32_t match = in.match;
                const int32_t mism = in.mismatch;
                long best = 0;
                if (approach == "untiled" || approach == "row") {
                    std::fill(prev.begin(), prev.begin() + cols + 1, 0);
                    for (int i = 1; i <= rows; ++i) {
                        curr[0] = 0;
                        int32_t left = 0;
                        for (int j = 1; j <= cols; ++j) {
                            const int32_t sub = (xs[i - 1] == ys[j - 1]) ? in.match : in.mismatch;
                            int32_t d = prev[j - 1] + sub;
                            int32_t u = prev[j] + gap;
                            int32_t l = left + gap;
                            int32_t m = d > u ? d : u;
                            m = m > l ? m : l;
                            m = m > 0 ? m : 0;
                            left = m;
                            curr[j] = m;
                            if (m > best) best = m;
                        }
                        prev.swap(curr);
                    }
                } else if (approach == "col") {
                    // SW boundaries are 0; prev_left carries H[i-1][bj-1].
                    std::fill(left_edge.begin(), left_edge.begin() + rows + 1, 0);
                    for (int bj = 1; bj <= cols; bj += TC) {
                        const int j1 = std::min(cols, bj + TC - 1);
                        const int tw = j1 - bj + 1;
                        std::fill(strip.begin(), strip.begin() + tw, 0);
                        int32_t prev_left = 0;  // H[0][bj-1]
                        for (int i = 1; i <= rows; ++i) {
                            int32_t diag = prev_left;
                            int32_t left = left_edge[i];
                            prev_left = left;
                            for (int jj = 0; jj < tw; ++jj) {
                                const int j = bj + jj;
                                const int32_t up = strip[jj];
                                const int32_t sub = (xs[i - 1] == ys[j - 1]) ? in.match : in.mismatch;
                                int32_t d = diag + sub;
                                int32_t u = up + gap;
                                int32_t l = left + gap;
                                int32_t m = d > u ? d : u;
                                m = m > l ? m : l;
                                m = m > 0 ? m : 0;
                                diag = up;
                                left = m;
                                strip[jj] = m;
                                if (m > best) best = m;
                            }
                            left_edge[i] = left;
                        }
                    }
                } else { // block
                    std::fill(top_row.begin(), top_row.begin() + cols + 1, 0);
                    for (int bi = 1; bi <= rows; bi += TR) {
                        const int i1 = std::min(rows, bi + TR - 1);
                        std::fill(left_edge.begin() + (bi - 1), left_edge.begin() + i1 + 1, 0);
                        for (int bj = 1; bj <= cols; bj += TC) {
                            const int j1 = std::min(cols, bj + TC - 1);
                            for (int j = bj; j <= j1; ++j) panel[j - bj] = top_row[j];
                            int32_t prev_left = left_edge[bi - 1];
                            for (int i = bi; i <= i1; ++i) {
                                int32_t diag = prev_left;
                                int32_t left = left_edge[i];
                                prev_left = left;
                                for (int j = bj; j <= j1; ++j) {
                                    const int jj = j - bj;
                                    const int32_t up = panel[jj];
                                    const int32_t sub = (xs[i - 1] == ys[j - 1]) ? in.match : in.mismatch;
                                    int32_t d = diag + sub;
                                    int32_t u = up + gap;
                                    int32_t l = left + gap;
                                    int32_t m = d > u ? d : u;
                                    m = m > l ? m : l;
                                    m = m > 0 ? m : 0;
                                    diag = up;
                                    left = m;
                                    panel[jj] = m;
                                    if (m > best) best = m;
                                }
                                left_edge[i] = left;
                            }
                            const int32_t top_at_j1 = top_row[j1];
                            for (int j = bj; j <= j1; ++j) top_row[j] = panel[j - bj];
                            left_edge[bi - 1] = top_at_j1;
                        }
                    }
                }
                score = best;
            } else {
                // ---- Full-matrix path (verify / non-score_only) ----
                std::fill(mat.begin(), mat.begin() + static_cast<std::size_t>(N), 0);
                auto cell = [&](int i, int j) __attribute__((always_inline)) {
                    const std::size_t p = static_cast<std::size_t>(i) * stride + j;
                    const std::size_t pin1 = static_cast<std::size_t>(i - 1) * stride;
                    const int32_t Ain1jn1 = mat[pin1 + (j - 1)];
                    const int32_t Cij = (xs[i - 1] == ys[j - 1]) ? in.match : in.mismatch;
                    const int32_t A3ij = Ain1jn1 + Cij;
                    mat[p] = A3ij;
                    const int32_t Ain1j = mat[pin1 + j];
                    const int32_t A1ij = Ain1j + gap;
                    mat[p] = A1ij;
                    const int32_t Aijn1 = mat[p - 1];
                    const int32_t A0ij = Aijn1 + gap;
                    const int32_t A2ij = A1ij > A0ij ? A1ij : A0ij;
                    const int32_t A4ij = A3ij > A2ij ? A3ij : A2ij;
                    mat[p] = A4ij > 0 ? A4ij : 0;
                };
                if (approach == "untiled") {
                    for (int i = 1; i <= rows; ++i)
                        for (int j = 1; j <= cols; ++j) cell(i, j);
                } else if (approach == "row") {
                    for (int bi = 1; bi <= rows; bi += TR)
                        for (int i = bi; i < bi + TR && i <= rows; ++i)
                            for (int j = 1; j <= cols; ++j) cell(i, j);
                } else if (approach == "col") {
                    for (int bj = 1; bj <= cols; bj += TC)
                        for (int i = 1; i <= rows; ++i)
                            for (int j = bj; j < bj + TC && j <= cols; ++j) cell(i, j);
                } else {
                    for (int bi = 1; bi <= rows; bi += TR)
                        for (int bj = 1; bj <= cols; bj += TC)
                            for (int i = bi; i < bi + TR && i <= rows; ++i)
                                for (int j = bj; j < bj + TC && j <= cols; ++j) cell(i, j);
                }
            }"""


_LOOPS_ALLOC = {"nw": _NW_LOOPS_ALLOC, "affine": _AFFINE_LOOPS_ALLOC, "sw": _SW_LOOPS_ALLOC}
_LOOPS_TIMED = {"nw": _NW_LOOPS_TIMED, "affine": _AFFINE_LOOPS_TIMED, "sw": _SW_LOOPS_TIMED}
# Which matrix variable each loops body fills (for the verify checksum).
_LOOPS_MATRIX_VAR = {"nw": "mat", "affine": "H", "sw": "mat"}


def loops_driver_source(kind: str, driver_name: str) -> str:
    """REPTILE-generated tiled DP loops (no external mapping) — the "vanilla"
    scalar comparison point.

    For NW/SW the timed path defaults to O(N) score-only (``--score-only``):
    2-row untiled, column panels, and 2D block panels. That is the setup where
    tiling actually shrinks the working set relative to an untiled prev-row
    that spills L1 — matching the locality argument behind SEQ's ``genNWtile``
    / ``genSWtile``, without the full-matrix DRAM writeback that hides the win
    on modern L3. Full-matrix nests remain for ``--verify`` / non-score_only.
    Affine stays full-matrix (Gotoh needs three planes).
    """
    cfg = KINDS[kind]
    alloc = _LOOPS_ALLOC[kind]
    timed_body = _LOOPS_TIMED[kind]
    # NW/SW: O(N) timed path sets `score`; full-matrix traceback only for --verify.
    # Affine: always materialize + traceback (no O(N) Gotoh path yet).
    post_timed = ""
    if kind in ("nw", "sw"):
        post_timed = f"""if (verify) {{
            {_checksum_block(_LOOPS_MATRIX_VAR[kind])}
            {cfg["loops_traceback"]}
        }}
        alignment.query = q.data();
        alignment.ref = r.data();
        alignment.comp = c.data();"""
    else:
        post_timed = f"""{_checksum_block(_LOOPS_MATRIX_VAR[kind])}
        {cfg["loops_traceback"]}
        alignment.query = q.data();
        alignment.ref = r.data();
        alignment.comp = c.data();"""

    return f"""// Auto-generated by Reptile/reptile.py. Do not edit by hand.
// REPTILE-generated tiled DP loops (no external mapping) for {kind}.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <new>

#include "driver_io.hpp"
#include "matrix_traceback.hpp"

{_ALIGNED64}

using baselines::Input;
using baselines::TimedRun;
using baselines::Timer;

int main(int argc, char** argv) {{
    return baselines::run_driver(argc, argv, "{driver_name}", [](const char* path) -> int {{
        const Input in = baselines::parse_input(path);
        const int rows = static_cast<int>(in.x.size());
        const int cols = static_cast<int>(in.y.size());
        const long cells = static_cast<long>(rows) * static_cast<long>(cols);
        {cfg["gap_setup"]}
        {_READ_TILING}

        const int R = rows + 1;
        const int C = cols + 1;
        const int phys_w = C;
        const long N = static_cast<long>(R) * phys_w;

        const int TR = (tile_rows_in > 0) ? tile_rows_in : 128;
        const int TC = (tile_cols_in > 0) ? tile_cols_in : 128;

        long score = 0;
        unsigned long long _checksum = 0;
        baselines::AlignmentTraceback alignment;

        // Alloc once (outside timer).
        {alloc}
        std::vector<char> q(static_cast<std::size_t>(rows + cols + 1));
        std::vector<char> r(static_cast<std::size_t>(rows + cols + 1));
        std::vector<char> c(static_cast<std::size_t>(rows + cols + 1));

        Timer timer;
        const TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {{
            {timed_body}
        }});

        {post_timed}

        if (verify) std::fprintf(stderr, "REPTILE_CHECKSUM %llu\\n", _checksum);
        baselines::print_result(score, timed, cells, alignment);
        return 0;
    }});
}}
"""


# --------------------------------------------------------------------------
# Compilation.
# --------------------------------------------------------------------------

def _bin_dir() -> Path:
    d = repo_root() / "Reptile" / "build"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compile(src: str, driver_name: str) -> Optional[Path]:
    fd, src_path = tempfile.mkstemp(suffix=".cpp", prefix=f"{driver_name}_", dir="/tmp")
    os.close(fd)
    Path(src_path).write_text(src)
    out_path = _bin_dir() / driver_name
    root = repo_root()
    cmd = [
        "clang++", "-std=c++20", "-O3", "-march=native",
        "-I", str(root / "baselines" / "common"),
        "-I", str(root / "baselines" / "traceback"),
        *_platform_compiler_flags(),
        src_path, "-o", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"compile failed for {driver_name}:\n{result.stderr}\n--- source head ---\n"
            + "\n".join(src.splitlines()[:40])
        )
    return out_path


def build_vecrec_driver(kind: str, scheduled_expr, mult: int, driver_name: str) -> Optional[Path]:
    k = build_vecrec_kernel(scheduled_expr, ele_per_iter_multiple=mult)
    if k is None:
        return None
    src = vecrec_driver_source(kind, k, driver_name)
    return _compile(src, driver_name)


def build_loops_driver(kind: str, driver_name: str) -> Optional[Path]:
    src = loops_driver_source(kind, driver_name)
    return _compile(src, driver_name)
