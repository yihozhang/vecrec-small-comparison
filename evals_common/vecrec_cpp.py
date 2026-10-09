"""Compile a vecrec kernel into a self-timed native C++ traceback driver.

The bio alignment cases used to time vecrec *in Python* (the ``kernel_expr`` path
in the harness). That is not comparable to the C++ library baselines, which
self-time the whole alignment — *including traceback* — under
``baselines::Timer``. This module closes the gap: it embeds vecrec's generated
kernel into a C++ driver that follows the same ``driver_io.hpp`` contract as the
library drivers and self-times **score-matrix build + memory allocation + kernel
+ traceback** as one region.

``vecrec_traceback_baselines(expr, kind)`` enumerates every candidate schedule
for ``expr`` and returns one :class:`VecrecTracebackBaseline` per schedule
(``vecrec_<kind>_tb:k0``, ``:k1``, ...), so all schedules are compared against
the libraries and the fastest is visible in the throughput table.
"""

from __future__ import annotations

import contextlib
import io as _io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from evals_common.cpp import CppBaseline

# Traceback kind -> (traceback+score C++ block, gap-setup C++).
#
# The kernel produces the full (rows+1) x (cols+1) textbook DP matrix (physical row
# stride `phys_w`). Substitution comes from a per-symbol profile, so `s` is never
# materialized and the traceback recomputes subst inline (match/mismatch). The block
# sets `score`: the logical corner (rows, cols) for global alignment. For local SW
# the global max+argmax is produced by the fused in-sweep reduce (see _driver_source),
# which overrides this block's traceback (only its gap-setup is used). Gap magnitudes
# come from the input.json scoring fields:
#   nw/sw:  gap = -gap_extend       affine: go=-gap_open, ge=-gap_extend
_M = "static_cast<int>(in.match)"
_MM = "static_cast<int>(in.mismatch)"
_CORNER_SCORE = (
    "score = static_cast<long>(output[static_cast<std::size_t>(rows) * phys_w + cols]);"
)
_KIND_TRACEBACK = {
    "nw": (
        f"baselines::nw_traceback_full_inline(output.data(), rows, cols, phys_w, "
        f"in.x.c_str(), in.y.c_str(), gap, {_M}, {_MM}, q.data(), r.data(), c.data());\n"
        f"            {_CORNER_SCORE}",
        "const int gap = -static_cast<int>(in.gap_extend);",
    ),
    "affine": (
        f"baselines::affine_traceback_full_inline(output.data(), rows, cols, phys_w, "
        f"in.x.c_str(), in.y.c_str(), go, ge, {_M}, {_MM}, q.data(), r.data(), c.data());\n"
        f"            {_CORNER_SCORE}",
        "const int go = -static_cast<int>(in.gap_open);\n"
        "        const int ge = -static_cast<int>(in.gap_extend);",
    ),
    "levenshtein": (  # already recomputes unit cost inline; no s
        "baselines::levenshtein_traceback_full(output.data(), rows, cols, phys_w, "
        "in.x.c_str(), in.y.c_str(), q.data(), r.data(), c.data());\n"
        f"            {_CORNER_SCORE}",
        "",
    ),
    "sw": (
        f"int64_t sw_score = 0;\n"
        f"            baselines::sw_traceback_full_inline(output.data(), rows, cols, phys_w, "
        f"in.x.c_str(), in.y.c_str(), gap, {_M}, {_MM}, q.data(), r.data(), c.data(), &sw_score);\n"
        f"            score = static_cast<long>(sw_score);",
        "const int gap = -static_cast<int>(in.gap_extend);",
    ),
    # Affine-gap LOCAL alignment (matches ksw_ll_i16). Like "sw" it is a
    # local (restart-at-0, max-cell) alignment, so the score comes from the fused
    # in-sweep max reduce (reduce_argmax). The full-matrix traceback block below is
    # only used if a non-score-only affine-local variant is ever wired; the
    # score-only path (used by the ll_i16 case) skips it entirely. Gap-setup is
    # affine (go/ge) to mirror the affine recurrence.
    "affine_sw": (
        f"int64_t sw_score = 0;\n"
        f"            baselines::sw_traceback_full_inline(output.data(), rows, cols, phys_w, "
        f"in.x.c_str(), in.y.c_str(), ge, {_M}, {_MM}, q.data(), r.data(), c.data(), &sw_score);\n"
        f"            score = static_cast<long>(sw_score);",
        "const int go = -static_cast<int>(in.gap_open);\n"
        "        const int ge = -static_cast<int>(in.gap_extend);",
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bin_dir() -> Path:
    override = os.environ.get("VECREC_EVALS_BIN")
    return Path(override) if override else _repo_root() / "bin"


def _platform_compiler_flags() -> list[str]:
    # Mirror harness.platform_compiler_flags without importing the harness here
    # (avoids an import cycle: harness imports nothing from this module).
    from evals_common.harness import platform_compiler_flags

    return platform_compiler_flags()


def _band_col_bounds(band_width: int, rows_per_iter: int, ele_per_iter: int) -> str:
    """Return C++ statements that set `col_start`, `col`, and `col_end` for the inner column loop.

    When band_width == 0 the full row [0, phys_w) is used (no banding); col_start=0.
    When band_width > 0 the column range is the symmetric diagonal band:
      col_start = max(0, first_row - bw) aligned DOWN to epi
      col_end   = min(phys_w, (first_row + rpi + bw) aligned UP to epi)
    col_start aligns to epi (not just vec_lanes) to match InputSignal2D._block_col_start().
    kernel.seek_col(col_start) repositions all state directly (O(1)) so that the first
    in-band column receives correct predecessor values.
    """
    epi = ele_per_iter
    rpi = rows_per_iter
    if band_width == 0:
        return (
            "int col_start = 0;\n"
            "                int col = 0;\n"
            "                int col_end = phys_w;"
        )
    bw = band_width
    return (
        f"int _log_row = row_base / phys_w;\n"
        f"                int col_start = (std::max(0, _log_row - {bw}) / {epi}) * {epi};\n"
        f"                int col = col_start;\n"
        f"                int col_end = std::min(phys_w, ((_log_row + {rpi} + {bw} + {epi} - 1) / {epi}) * {epi});"
    )


# SubstSignal2D buffer setup for ComputedSignal2D kernels (see affine_gap.py).
# Emitted inside the timed region; phys_w/rows/cols/in are in driver scope.
_HARNESS_SUBST_SETUP = """auto _c2i8 = [](char ch) -> int8_t {
            switch (ch) {
            case 'A': case 'a': return 0; case 'C': case 'c': return 1;
            case 'G': case 'g': return 2; case 'T': case 't': return 3;
            default: return 0;
            }
        };
        // Halo: SubstSignal2D may read s1 one row past the last on final reset_and_next_row().
        std::vector<int8_t> s1(static_cast<std::size_t>(rows) + 64, static_cast<int8_t>(-1));
        std::vector<int8_t> s2(static_cast<std::size_t>(phys_w), static_cast<int8_t>(-1));
        for (int i = 0; i < rows; ++i)
            s1[static_cast<std::size_t>(i + 1)] = _c2i8(in.x[static_cast<std::size_t>(i)]);
        for (int j = 0; j < cols; ++j)
            s2[static_cast<std::size_t>(j + 1)] = _c2i8(in.y[static_cast<std::size_t>(j)]);"""


def _driver_source(
    *,
    header_path: str,
    kernel_name: str,
    out_cpp_type: str,
    s_cpp_type: str,
    ele_per_iter: int,
    rows_per_iter: int,
    driver_name: str,
    traceback_block: str,
    gap_setup: str,
    reduce_argmax: bool = False,
    band_width: int = 0,
    computed_ctor: str = "",
    score_only: bool = False,
) -> str:
    """Render the C++ driver that wraps one compiled vecrec kernel.

    Substitution is either a per-symbol profile (``ProfileSignal2D`` patch path) or
    an on-the-fly ``ComputedSignal2D`` (``SubstSignal2D``). Local alignment
    (``reduce_argmax``) folds the global max+argmax flat over the output as it is
    produced (no separate scan).
    """
    out_per_row_init = "\n".join(
        f"                    out_buf[{r}]," for r in range(rows_per_iter)
    )
    # Direct-write (global, full matrix, 64B-aligned output): point out_per_row
    # straight into the output matrix so the kernel's run() writes results in
    # place, eliminating the staging out_buf and a whole O(N^2) memcpy pass.
    # Banded / SW (reduce_argmax) / score-only keep the staging buffer.
    _direct_write = (band_width == 0) and (not reduce_argmax) and (not score_only)
    if _direct_write:
        out_per_row_decl = (
            f"{kernel_name}_vector_type* out_per_row[{rows_per_iter}];\n"
            f"                    for (int _rr = 0; _rr < {rows_per_iter}; ++_rr)\n"
            f"                        out_per_row[_rr] = reinterpret_cast<{kernel_name}_vector_type*>("
            f"output.data() + static_cast<std::size_t>(row_base) + static_cast<std::size_t>(_rr) * phys_w + col);"
        )
    else:
        out_per_row_decl = (
            f"{kernel_name}_vector_type out_buf[{rows_per_iter}][{ele_per_iter}/vec_lanes];\n"
            f"                    {kernel_name}_vector_type* out_per_row[{rows_per_iter}] = {{\n"
            f"{out_per_row_init}\n"
            f"                    }};"
        )
    # The kernel reads a per-symbol profile (O(alphabet*cols)) indexed by a per-row
    # `sym` table (O(rows)); the O(N^2) substitution matrix is never materialized
    # (the generated header is patched to use ProfileSignal2D).
    profile_build = f"""auto _sym = [](char ch) -> int {{
            switch (ch) {{ case 'A': case 'a': return 0; case 'C': case 'c': return 1;
                           case 'G': case 'g': return 2; case 'T': case 't': return 3;
                           default: return 0; }}
        }};
        std::vector<{s_cpp_type}> profile(static_cast<std::size_t>(4) * phys_w, 0);
        for (int c = 0; c < 4; ++c)
            for (int j = 0; j < cols; ++j)
                profile[static_cast<std::size_t>(c) * phys_w + (j + 1)] =
                    static_cast<{s_cpp_type}>(c == _sym(in.y[j]) ? in.match : in.mismatch);
        std::vector<int> sym(static_cast<std::size_t>(R), 0);
        for (int i = 0; i < rows; ++i) sym[i + 1] = _sym(in.x[i]);"""

    # Sentinel for band_buf: TropMax zero (INT16_MIN/8 = -4096 for I16 kernels).
    # Computed here so it is available in both the reduce_argmax and alloc blocks.
    sentinel_val = f"static_cast<{out_cpp_type}>(-4096)"

    # VECREC_REDUCE_ARGMAX (SW only): a FLAT parallel max+argmax fold over each output
    # vector AS it is produced in registers (a `Reduce` primitive prototyped in the
    # sweep loop) -- __builtin_reduce_max per vector + a rare lane-find -- so the global
    # max + start cell come out of the sweep with no separate O(N^2) scan pass.
    if reduce_argmax:
        reduce_decl = "elt_type gmax = 0; long gargmax = 0;"
        # Vector schedules only (vec_lanes>1); scalar schedules (k0/k1) can't use
        # __builtin_reduce_max and simply SKIP -- they're the slowest anyway.
        reduce_fold = (
            "for (int _s = 0; _s < %d/vec_lanes; ++_s) {\n"
            "                            %s_vector_type _v = out_buf[rr][_s];\n"
            "                            elt_type _hm = __builtin_reduce_max(_v);\n"
            "                            if (_hm > gmax) { gmax = _hm; long _b = (long)dst + (long)_s * vec_lanes;\n"
            "                                for (int _l = 0; _l < vec_lanes; ++_l) if (_v[_l] == _hm) { gargmax = _b + _l; break; } }\n"
            "                        }"
        ) % (ele_per_iter, kernel_name)
        if band_width > 0:
            # Banded SW: gargmax is a flat index (row_base + rr*phys_w + col + lane).
            # Convert to logical (bi, bj) then walk the band strip.
            traceback_block = (
                "score = static_cast<long>(gmax);\n"
                "            { int _bi = (int)(gargmax / phys_w), _bj = (int)(gargmax % phys_w);\n"
                "              baselines::sw_walk_banded_inline(band_buf.data(), col_starts.data(), n_col,\n"
                "                rows, cols, in.x.c_str(), in.y.c_str(),\n"
                "                gap, static_cast<int>(in.match), static_cast<int>(in.mismatch),\n"
                f"                {sentinel_val}, _bi, _bj, q.data(), r.data(), c.data()); }}"
            )
        else:
            # SW walk-only traceback using the fused (gmax, gargmax) -- no scan.
            traceback_block = (
                "score = static_cast<long>(gmax);\n"
                "            { int _bi = (int)(gargmax / phys_w), _bj = (int)(gargmax % phys_w);\n"
                "              baselines::sw_walk_inline(output.data(), rows, cols, phys_w, in.x.c_str(), in.y.c_str(), "
                "gap, static_cast<int>(in.match), static_cast<int>(in.mismatch), _bi, _bj, q.data(), r.data(), c.data()); }"
            )
    else:
        reduce_decl = ""
        reduce_fold = ""

    input_outside = ""
    if computed_ctor:
        input_inside = _HARNESS_SUBST_SETUP
        kernel_ctor = computed_ctor
        sbuild_note = "SubstSignal2D on-the-fly (O(N) s1/s2; no profile, no O(N^2) s)"
    else:
        input_inside = profile_build
        kernel_ctor = f"{kernel_name}(profile.data(), sym.data(), phys_w, phys_w)"
        sbuild_note = "parasail-style profile in-kernel (O(N) profile+sym; no O(N^2) s)"

    # --- banded vs full-matrix allocation and write path ---
    bw = band_width
    epi = ele_per_iter
    rpi = rows_per_iter
    if bw > 0:
        # Band-strip allocation: one row slice of width n_col per logical row.
        # n_col = ceil((2*bw + epi) / epi) * epi — same rounding as col_end in
        # _band_col_bounds so every in-band column written fits within the strip.
        # col_starts[i] mirrors col_start from the sweep loop exactly.
        alloc_comment = f"band-strip O(R × n_col) with bw={bw}, n_col≈{2*bw+epi}"
        alloc_block = (
            f"const int n_col = std::min(phys_w, (((2 * {bw} + {epi} - 1) / {epi}) + 1) * {epi});\n"
            f"            std::vector<int> col_starts(static_cast<std::size_t>(R), 0);\n"
            f"            for (int _i = 0; _i < R; ++_i)\n"
            f"                col_starts[_i] = (std::max(0, _i - {bw}) / {epi}) * {epi};\n"
            f"            std::vector<{out_cpp_type}> band_buf(\n"
            f"                static_cast<std::size_t>(R) * n_col,\n"
            f"                {sentinel_val});"
        )
        # Write each rr-row's epi-wide chunk into its band-relative slot.
        # `dst` is still computed in the full phys_w-stride address space so that
        # the SW reduce_fold can use it to compute gargmax as a flat (row*phys_w+col)
        # index (which is later decoded to (bi,bj) via gargmax/phys_w and %phys_w).
        write_block = (
            f"int _log_row = row_base / phys_w + rr;\n"
            f"                        if (_log_row >= R) break;\n"
            f"                        int dst = row_base + rr * phys_w + col;  // flat index for SW reduce_fold\n"
            f"                        int _dst_b = _log_row * n_col + (col - col_starts[_log_row]);\n"
            f"                        int _copy_b = std::min({epi}, n_col - (col - col_starts[_log_row]));\n"
            f"                        if (_copy_b > 0)\n"
            f"                            std::memcpy(band_buf.data() + _dst_b, out_buf[rr],\n"
            f"                                        sizeof({out_cpp_type}) * _copy_b);"
        )
        tb_comment = "banded strip (O(n×bw) memory, traceback via band_get)"
        # Override the traceback_block for non-SW banded kinds.
        # (SW banded traceback is already set above in reduce_argmax branch.)
        if not reduce_argmax:
            if "nw_traceback_full_inline" in traceback_block:
                traceback_block = traceback_block.replace(
                    f"baselines::nw_traceback_full_inline(output.data(), rows, cols, phys_w, "
                    f"in.x.c_str(), in.y.c_str(), gap, {_M}, {_MM}, q.data(), r.data(), c.data());",
                    f"baselines::nw_traceback_banded_inline(band_buf.data(), col_starts.data(), n_col, "
                    f"rows, cols, in.x.c_str(), in.y.c_str(), gap, {_M}, {_MM}, "
                    f"{sentinel_val}, q.data(), r.data(), c.data());",
                )
            if "affine_traceback_full_inline" in traceback_block:
                traceback_block = traceback_block.replace(
                    f"baselines::affine_traceback_full_inline(output.data(), rows, cols, phys_w, "
                    f"in.x.c_str(), in.y.c_str(), go, ge, {_M}, {_MM}, q.data(), r.data(), c.data());",
                    f"baselines::affine_traceback_banded_inline(band_buf.data(), col_starts.data(), n_col, "
                    f"rows, cols, in.x.c_str(), in.y.c_str(), go, ge, {_M}, {_MM}, "
                    f"{sentinel_val}, q.data(), r.data(), c.data());",
                )
            # Score from band strip instead of full output[] corner.
            traceback_block = traceback_block.replace(
                "score = static_cast<long>(output[static_cast<std::size_t>(rows) * phys_w + cols]);",
                "score = static_cast<long>(band_buf[static_cast<std::size_t>(rows) * n_col + (cols - col_starts[rows])]);",
            )
    elif reduce_argmax:
        # SW non-banded: reduce_fold reads out_buf, keep staging.
        alloc_comment = "produce the full (R x phys_w) DP matrix"
        alloc_block = f"std::vector<{out_cpp_type}> output(static_cast<std::size_t>(N));"
        write_block = (
            f"int dst = row_base + rr * phys_w + col;\n"
            f"                        if (dst >= N) break;\n"
            f"                        int copy_count = ((N - dst) < {epi}) ? (N - dst) : {epi};\n"
            f"                        std::memcpy(output.data() + dst, out_buf[rr],\n"
            f"                                    sizeof({out_cpp_type}) * copy_count);"
        )
        tb_comment = "full matrix (logical R x C, physical stride phys_w)"
    else:
        # Direct-write: 64B-aligned, padded so the final block-row never overruns.
        alloc_comment = "produce the full (R x phys_w) DP matrix (direct write, 64B-aligned)"
        alloc_block = (
            f"std::vector<{out_cpp_type}, Aligned64<{out_cpp_type}>> output("
            f"static_cast<std::size_t>(N) + static_cast<std::size_t>({rpi}) * phys_w);"
        )
        write_block = ""
        tb_comment = "full matrix (logical R x C, physical stride phys_w), direct write"

    # --- score-only override: O(row) rolling buffer, no full matrix, no traceback ---
    # Overrides whatever alloc/write path was chosen above. The score comes from the
    # fused in-sweep max reduce (reduce_argmax kinds: affine-local) or the captured
    # corner cell (global/linear kinds). A 2-row ring (curr+prev) keeps memory O(row),
    # matching the score-only ksw2 kernels (ksw_ll_i16, ksw_extf2_sse).
    if score_only:
        alloc_comment = "score-only: O(row) rolling 2-row buffer (curr+prev), no full matrix"
        alloc_block = (
            f"std::vector<{out_cpp_type}> rowbuf(static_cast<std::size_t>(2) * phys_w, {sentinel_val});"
        )
        write_block = (
            f"int _lrow = row_base / phys_w + rr;\n"
            f"                        if (_lrow >= R) break;\n"
            f"                        int dst = row_base + rr * phys_w + col;  // flat idx for reduce_fold\n"
            f"                        int _ring = (_lrow & 1) * phys_w;\n"
            f"                        int _cc = ((phys_w - col) < {epi}) ? (phys_w - col) : {epi};\n"
            f"                        if (_cc > 0)\n"
            f"                            std::memcpy(rowbuf.data() + _ring + col, out_buf[rr],\n"
            f"                                        sizeof({out_cpp_type}) * _cc);"
        )
        tb_comment = "score-only (no traceback)"

    # --- final score / traceback section (conditional on score_only) ---
    if score_only:
        if reduce_argmax:
            score_stmt = (
                "// local (restart-at-0) score = fused in-sweep max reduce\n"
                "            score = static_cast<long>(gmax);"
            )
        else:
            score_stmt = (
                "// global/linear score = corner cell H(rows, cols) from the row ring\n"
                "            score = static_cast<long>("
                "rowbuf[static_cast<std::size_t>(rows & 1) * phys_w + cols]);"
            )
        traceback_section = (
            f"// --- score-only finalize ({tb_comment}); O(row) memory, no traceback ---\n"
            f"            {score_stmt}"
        )
    else:
        traceback_section = (
            f"// --- traceback ({tb_comment}) ---\n"
            f"            // The block also sets `score` (logical corner for global alignment, or\n"
            f"            // the scanned global max for local Smith-Waterman).\n"
            f"            std::vector<char> q(static_cast<std::size_t>(rows + cols + 1));\n"
            f"            std::vector<char> r(static_cast<std::size_t>(rows + cols + 1));\n"
            f"            std::vector<char> c(static_cast<std::size_t>(rows + cols + 1));\n"
            f"            {traceback_block}\n"
            f"            alignment.query = q.data();\n"
            f"            alignment.ref = r.data();\n"
            f"            alignment.comp = c.data();"
        )

    return f"""// Auto-generated by evals_common/vecrec_cpp.py. Do not edit by hand.
//
// Self-timed vecrec kernel + custom matrix traceback, reported via the shared
// baselines driver contract so it sits in the same table as the C++ libraries.
#include "{header_path}"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "driver_io.hpp"
#include "matrix_traceback.hpp"
#include <new>

// 64-byte aligned allocator so the output matrix can be written in place by the
// kernel's vector stores (direct-write path, no staging out_buf).
template <class T> struct Aligned64 {{
    using value_type = T;
    Aligned64() = default;
    template <class U> Aligned64(const Aligned64<U>&) {{}}
    T* allocate(std::size_t n) {{
        return static_cast<T*>(::operator new[](n * sizeof(T), std::align_val_t(64)));
    }}
    void deallocate(T* p, std::size_t) {{ ::operator delete[](p, std::align_val_t(64)); }}
    template <class U> bool operator==(const Aligned64<U>&) const {{ return true; }}
    template <class U> bool operator!=(const Aligned64<U>&) const {{ return false; }}
}};

using baselines::Input;
using baselines::TimedRun;
using baselines::Timer;

int main(int argc, char** argv) {{
    return baselines::run_driver(argc, argv, "{driver_name}", [](const char* path) -> int {{
        const Input in = baselines::parse_input(path);
        const int rows = static_cast<int>(in.x.size());
        const int cols = static_cast<int>(in.y.size());
        const long cells = static_cast<long>(rows) * static_cast<long>(cols);
        {gap_setup}

        // The vecrec kernel computes the textbook DP over a grid whose (0,0) cell
        // is the boundary origin (no character consumed) and whose diagonal move
        // into (i,j) scores (x[i-1], y[j-1]). Feeding it an (rows+1) x (cols+1)
        // matrix yields the full textbook matrix, with the optimum at (rows,cols).
        // The physical row width is rounded up to a multiple of ele_per_iter so the
        // kernel's block loop never reads past the score buffer.
        constexpr int EPI = {ele_per_iter};
        const int R = rows + 1;                       // logical rows (incl. boundary)
        const int C = cols + 1;                       // logical cols (incl. boundary)
        const int phys_w = ((C + EPI - 1) / EPI) * EPI;  // physical row width
        const int N = R * phys_w;

        long score = 0;
        baselines::AlignmentTraceback alignment;

        // {sbuild_note}
        {input_outside}

        Timer timer;
        const TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {{
            // --- substitution input (full s, or parasail-style profile+sym) ---
            {input_inside}

            // --- vecrec kernel: {alloc_comment} ---
            {alloc_block}
            auto kernel = {kernel_ctor};
            using elt_type = typename ElementType<{kernel_name}_vector_type>::type;
            constexpr int vec_lanes = sizeof({kernel_name}_vector_type) / sizeof(elt_type);
            {reduce_decl}
            int row_base = 0;
            while (row_base < N) {{
                {_band_col_bounds(band_width, rows_per_iter, ele_per_iter)}
                // O(1) seek: reposition all internal kernel state (ProfileSignal2D row
                // pointers, Signal2D column pointer, SConvolve diagonal carry) directly to
                // col_start without running the warm-up loop.
                kernel.seek_col(col_start);
                while (col < col_end) {{
                    {out_per_row_decl}
                    kernel.run(out_per_row);
                    for (int rr = 0; rr < {rows_per_iter}; rr++) {{
                        {write_block}
                        {reduce_fold}
                    }}
                    col += {ele_per_iter};
                }}
                for (int rr = 0; rr < {rows_per_iter}; rr++) kernel.reset_and_next_row();
                row_base += {rows_per_iter} * phys_w;
            }}

            {traceback_section}
        }});

        baselines::print_result(score, timed, cells, alignment);
        return 0;
    }});
}}
"""


# In-process cache: driver_name -> compiled binary path (or None if build failed).
_BUILD_CACHE: dict[str, Optional[Path]] = {}


def build_vecrec_traceback_driver(scheduled_expr, kind: str, driver_name: str, band_width: int = 0, ele_per_iter_multiple: int = 2, score_only: bool = False) -> Optional[Path]:
    """Generate + compile a self-timed vecrec traceback driver. Returns its path.

    ``scheduled_expr`` must already be lane-annotated (a candidate schedule, not
    the raw expr). Returns ``None`` if vecrec codegen or the C++ compile fails.
    Results are cached per ``driver_name`` for the process lifetime.
    ``band_width > 0`` restricts the column loop per row to a symmetric diagonal
    band of that half-width.  A warm-up pass runs kernel.run() from col=0 to
    col_start (without writing output) to correctly prime all kernel state before
    the in-band loop, matching ksw2's ``w`` parameter semantics.
    """
    if driver_name in _BUILD_CACHE:
        return _BUILD_CACHE[driver_name]

    if kind not in _KIND_TRACEBACK:
        raise ValueError(f"unknown traceback kind: {kind!r}")
    # Fused flat-reduce argmax is local-alignment only (SW / affine-local); global
    # cases read the corner.
    reduce_argmax = kind in ("sw", "affine_sw")
    traceback_block, gap_setup = _KIND_TRACEBACK[kind]

    try:
        from vecrec.codegen import CodeGen, instantiate_kernels

        kernel_name = "k0"
        codegen = CodeGen(band_width=band_width, ele_per_iter_multiple=ele_per_iter_multiple)
        # Suppress vecrec stdout chatter (temp paths etc.).
        with contextlib.redirect_stdout(_io.StringIO()):
            code = codegen.generate(scheduled_expr, kernel_name)
        out_elt = scheduled_expr.element_type

        fd, header_path = tempfile.mkstemp(suffix=".h", prefix=f"{driver_name}_hdr_", dir="/tmp")
        os.close(fd)
        instantiate_kernels(header_path, [code])

        computed_ctor = ""
        # Older/sibling vec-rec codegen may not expose `computed_signals`; treat a
        # missing attribute as "no computed signals" (legacy ProfileSignal2D path).
        if getattr(code, "computed_signals", None):
            # Preludes are embedded by instantiate_kernels; driver constructs the signal.
            decl = code.computed_signals[0]
            computed_ctor = f"{kernel_name}({decl.resolved_type}({decl.make_args}), phys_w)"
        else:
            # Patch Var2D inputs to ProfileSignal2D (legacy bio eval path).
            _, vars2d = codegen.collect_input_variables(scheduled_expr)
            svar = next(iter(vars2d.keys()))
            import re as _re
            hdr = Path(header_path).read_text()
            hdr = hdr.replace("InputSignal2D<", "ProfileSignal2D<")
            hdr = hdr.replace(f"({svar}, {svar}_stride,", f"({svar}, {svar}_sym, {svar}_stride,")
            hdr = _re.sub(
                rf"\*{svar}, int {svar}_stride",
                f"*{svar}, const int *{svar}_sym, int {svar}_stride",
                hdr,
            )
            Path(header_path).write_text(hdr)

        src = _driver_source(
            header_path=header_path,
            kernel_name=kernel_name,
            out_cpp_type=out_elt.to_str(),
            s_cpp_type=out_elt.to_str(),
            ele_per_iter=code.element_per_iter,
            rows_per_iter=code.rows_per_iter,
            driver_name=driver_name,
            traceback_block=traceback_block,
            gap_setup=gap_setup,
            reduce_argmax=reduce_argmax,
            band_width=band_width,
            computed_ctor=computed_ctor,
            score_only=score_only,
        )
        fd, src_path = tempfile.mkstemp(suffix=".cpp", prefix=f"{driver_name}_", dir="/tmp")
        os.close(fd)
        Path(src_path).write_text(src)

        bin_dir = _bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)
        out_path = bin_dir / driver_name

        root = _repo_root()
        cmd = [
            "clang++", "-std=c++20", "-O3", "-march=native",
            "-I", str(root / "baselines" / "common"),
            "-I", str(root / "baselines" / "traceback"),
            *_platform_compiler_flags(),
            src_path, "-o", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Surface the compiler error via the harness's "not available" note path.
            _BUILD_CACHE[driver_name] = None
            return None
    except Exception:
        _BUILD_CACHE[driver_name] = None
        return None

    _BUILD_CACHE[driver_name] = out_path
    return out_path


class VecrecTracebackBaseline(CppBaseline):
    """A vecrec kernel + C++ traceback driver, compiled on first use.

    Behaves like any :class:`CppBaseline` once built: same ``input.json`` payload
    and JSON result shape, so the harness times/validates it identically to the
    library drivers (work mode ``traceback``).
    """

    def __init__(self, name: str, scheduled_expr, kind: str, band_width: int = 0, ele_per_iter_multiple: int = 2, score_only: bool = False) -> None:
        # Display name may contain ':' (e.g. vecrec_nw_tb:k0); keep the on-disk
        # binary name filesystem-safe.
        binary = name.replace(":", "_")
        super().__init__(name=name, binary=binary, algo=kind)
        self.scheduled_expr = scheduled_expr
        self.kind = kind
        self.band_width = band_width
        self.ele_per_iter_multiple = ele_per_iter_multiple
        self.score_only = score_only
        self._built: Optional[Path] = None

    def _ensure_built(self) -> Optional[Path]:
        if self._built is None:
            self._built = build_vecrec_traceback_driver(
                self.scheduled_expr, self.kind, self.binary, band_width=self.band_width,
                ele_per_iter_multiple=self.ele_per_iter_multiple, score_only=self.score_only,
            )
        return self._built

    def path(self) -> Path:
        built = self._ensure_built()
        return built if built is not None else _bin_dir() / self.binary

    def available(self) -> bool:
        built = self._ensure_built()
        return built is not None and built.exists() and os.access(built, os.X_OK)


def vecrec_traceback_baselines_multi_width(
    expr_fn,
    kind: str,
    band_width: int = 0,
    score_only: bool = False,
    element_types: tuple = None,
    variant: str = "",
) -> list[VecrecTracebackBaseline]:
    """Like :func:`vecrec_traceback_baselines`, but sweeps element width too.

    ``expr_fn`` is a callable ``(ElementType) -> SignalExpr`` (typically a
    case's ``self.kernel_expr`` bound method, which takes ``element_type``
    with a default) -- NOT an already-built expr. For each width in
    ``element_types`` (default I8/I16/I32) this builds the expr at that width,
    enumerates every candidate SIMD schedule for it, and labels the results
    ``vecrec_<kind>_i8_tb:k0``, ``_i16_tb:k0``, ``_i32_tb:k0``, ... so the
    throughput table shows the fastest schedule *and* the fastest width
    side by side -- autoscheduling isn't just picking a schedule anymore, it's
    picking (width, schedule).

    Narrower widths (I8) fit more lanes per physical SIMD register, so they can
    win on throughput, but they saturate (clamp, not wrap -- VecRec uses
    saturating arithmetic) once a score exceeds the type's range. That shows up
    as a correctness FAIL for i8 at larger N, which is expected and does not
    gate the suite (vecrec entries never gate correctness; see
    ``evals_common/harness.py::run_case``). Comparing PASS/FAIL across widths
    at each sweep size *is* the interesting signal: it tells you the largest N
    where int8 is still safe to use for this scoring scheme.

    ``variant`` is an extra disambiguator prefixed to the per-width variant tag
    (``vecrec_<kind>_<variant>_i8_tb:k*`` instead of ``vecrec_<kind>_i8_tb:k*``)
    -- needed when two cases share ``kind`` but compile different expressions
    (e.g. a 1-piece vs 2-piece affine both using ``kind="affine"``), same as
    the plain ``vecrec_traceback_baselines``'s ``variant`` param.
    """
    from vecrec.util import ElementType

    widths = element_types or (ElementType.I8, ElementType.I16, ElementType.I32)
    out: list[VecrecTracebackBaseline] = []
    for et in widths:
        expr = expr_fn(et)
        width_variant = f"{variant}_i{et.bit_width()}" if variant else f"i{et.bit_width()}"
        out.extend(
            vecrec_traceback_baselines(
                expr,
                kind=kind,
                band_width=band_width,
                score_only=score_only,
                variant=width_variant,
            )
        )
    return out


def vecrec_traceback_baselines(expr, kind: str, band_width: int = 0, score_only: bool = False, variant: str = "") -> list[VecrecTracebackBaseline]:
    """One :class:`VecrecTracebackBaseline` per candidate schedule of ``expr``.

    ``band_width > 0`` restricts each row's column sweep to a diagonal band of
    that half-width, matching ksw2's ``w`` parameter semantics.

    ``score_only=True`` generates a score-only driver: it keeps only an O(row)
    rolling buffer (no full O(N^2) matrix) and does NO traceback -- the score is
    the fused in-sweep max reduce (local kinds) or the captured corner cell
    (global/linear kinds). Use it to pair with score-only ksw2 kernels
    (ksw_ll_i16, ksw_extf2_sse) so memory footprint and timed span match.

    ``variant`` disambiguates the on-disk binary name when two cases share the
    same ``kind`` and ``band_width`` but compile DIFFERENT expressions (e.g. a
    1-piece vs a 2-piece affine expr both use ``kind="affine"``). Without it the
    per-process build cache would hand back the wrong binary.
    """
    from vecrec.optimize import get_candidate_schedules
    from vecrec.util import get_native_bits

    try:
        max_bits = get_native_bits()
    except OSError:
        # get_native_bits shells out to `lscpu`, absent on macOS; vecrec's default.
        max_bits = 256
    # Schedules use NATIVE-width SIMD lanes (one physical register per vector).
    # We expose more parallelism not by widening the logical lane count (which
    # makes each shift span multiple physical registers — expensive cross-lane
    # shuffles) but by raising ele_per_iter_multiple: that emits 2x/4x
    # INDEPENDENT native-width vectors per run(), so the dilation log-tower runs
    # as several independent intra-register towers stitched by a cheap O(1)
    # carry — exactly the structure that minimises shuffle + latency cost. The
    # harness benchmarks all and the fastest per size wins.
    schedules = list(get_candidate_schedules(expr, max_bits))
    band_suffix = f"_bw{band_width}" if band_width > 0 else ""
    # score-only drivers get a distinct binary name so their build cache never
    # collides with a full-matrix driver of the same kind/band (e.g. the nw kind
    # is used full-matrix by needleman_wunsch and score-only by the extf2 case).
    so_suffix = "_so" if score_only else ""
    variant_suffix = f"_{variant}" if variant else ""
    out: list[VecrecTracebackBaseline] = []
    k = 0
    for mult in (2, 4):
        for sched in schedules:
            out.append(
                VecrecTracebackBaseline(
                    f"vecrec_{kind}{variant_suffix}_tb{band_suffix}{so_suffix}:k{k}", sched, kind,
                    band_width=band_width, ele_per_iter_multiple=mult,
                    score_only=score_only,
                )
            )
            k += 1
    return out
