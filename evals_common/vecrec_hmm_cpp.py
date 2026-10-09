"""Self-timed vecrec profile-HMM Viterbi drivers (one per candidate schedule).

Parallel to :mod:`evals_common.vecrec_cpp` (the alignment drivers) but for the
profile-HMM Viterbi kernel (:mod:`bio_kernel_evals.cases._hmm_kernel`), which has
ten 2D inputs instead of one ``s``. Each driver:

  1. reads the compact model JSON (``baselines/common/hmm_io.hpp``: M/L/K, obs, and
     the configured tsc/emitM/emitI score arrays HMMER dumped),
  2. inside one ``Timer`` region: materializes the ten per-cell ``L x phys_w`` kernel
     inputs from the compact model (vecrec's structural setup cost, like the
     alignment s-build), runs the kernel to fill the H = max(M,I,D) matrix, and does
     an O(L+M) traceback over H to recover an M/I/D state path,
  3. prints a float score (the corner ``H[L-1][M]``) via ``print_hmm_result``.

Compiled on the fly (clang++ -O3) into ``bin/`` and cached per driver name, exactly
like the alignment drivers; sits in the same ``work_mode=traceback`` results table.
"""

from __future__ import annotations

import contextlib
import io as _io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from evals_common.cpp import CppBaseline, CppResult

# Transition sub-indices into tsc[k*8 + s] (p7P_TSC layout).
_MM, _IM, _DM, _BM, _MD, _DD, _MI, _II = range(8)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bin_dir() -> Path:
    override = os.environ.get("VECREC_EVALS_BIN")
    return Path(override) if override else _repo_root() / "bin"


def _platform_compiler_flags() -> list[str]:
    from evals_common.harness import platform_compiler_flags

    return platform_compiler_flags()


# The ten 2D inputs, in vecrec codegen constructor order (alphabetical by name),
# each mapped to (C++ profile vector, C++ sym table). Instead of materializing ten
# full L x phys_w arrays, the header is patched so every input is a ProfileSignal2D
# reading `profile + sym[row]*phys_w` -- so we build only compact profiles:
#   emissions: K profile rows, indexed by sym=obs (the per-residue query profile);
#   the 7 transitions are row-independent: a single profile row, sym=0 everywhere;
#   begin is TBM(k-1) at row 0 only: a 2-row profile (row0=TBM, row1=-inf), sym picks
#   row 0 at i==0 and row 1 thereafter.
_VAR_PROFILE: dict[str, tuple[str, str]] = {
    "a_dd": ("tprof_dd", "sym_zero"),
    "a_dm": ("tprof_dm", "sym_zero"),
    "a_ii": ("tprof_ii", "sym_zero"),
    "a_im": ("tprof_im", "sym_zero"),
    "a_md": ("tprof_md", "sym_zero"),
    "a_mi": ("tprof_mi", "sym_zero"),
    "a_mm": ("tprof_mm", "sym_zero"),
    "begin": ("begin_prof", "sym_begin"),
    "emitI": ("emitI_prof", "sym_obs"),
    "emitM": ("emitM_prof", "sym_obs"),
}
_INPUTS = tuple(sorted(_VAR_PROFILE))  # alphabetical == codegen ctor order


def _build_profiles_cpp() -> str:
    """C++ that builds the compact per-symbol profiles + sym tables for the
    ProfileSignal2D inputs (O(K*M + L), vs the old O(10*L*M) full materialization).
    Each profile row is phys_w wide (pad columns are -inf). Transitions for the M/D
    updates are pre-shifted (value at col k = T*(k-1)); I-update transitions are read
    directly at k. sym tables are sized L+1 so the kernel's final reset_and_next_row
    (which advances current_row to L) never reads out of bounds."""
    return f"""
        const int Mm = m.M, Lm = m.L, Kk = m.K;
        const float NEG = -std::numeric_limits<float>::infinity();
        auto T = [&](int s, int k) {{ return m.tsc[(std::size_t)k * 8 + s]; }};

        // emission profiles: Kk rows x phys_w, row x = emission vector for residue x.
        std::vector<float> emitM_prof((std::size_t)Kk * phys_w, NEG);
        std::vector<float> emitI_prof((std::size_t)Kk * phys_w, NEG);
        for (int x = 0; x < Kk; ++x)
            for (int k = 0; k <= Mm; ++k) {{
                emitM_prof[(std::size_t)x * phys_w + k] = m.emitM[(std::size_t)x * (Mm + 1) + k];
                emitI_prof[(std::size_t)x * phys_w + k] = m.emitI[(std::size_t)x * (Mm + 1) + k];
            }}

        // transition profiles: one row x phys_w each (row-independent).
        std::vector<float> tprof_mm(phys_w, NEG), tprof_im(phys_w, NEG), tprof_dm(phys_w, NEG),
            tprof_md(phys_w, NEG), tprof_dd(phys_w, NEG), tprof_mi(phys_w, NEG), tprof_ii(phys_w, NEG);
        for (int k = 0; k <= Mm; ++k) {{
            if (k >= 1) {{
                tprof_mm[k] = T({_MM}, k - 1); tprof_im[k] = T({_IM}, k - 1);
                tprof_dm[k] = T({_DM}, k - 1); tprof_md[k] = T({_MD}, k - 1);
                tprof_dd[k] = T({_DD}, k - 1);
            }}
            tprof_mi[k] = T({_MI}, k); tprof_ii[k] = T({_II}, k);
        }}

        // begin profile: 2 rows x phys_w. row 0 = TBM(k-1) (the begin entry), row 1 = -inf.
        std::vector<float> begin_prof((std::size_t)2 * phys_w, NEG);
        for (int k = 1; k <= Mm; ++k) begin_prof[k] = T({_BM}, k - 1);

        // sym tables (length L+1): emissions index by residue; transitions always row 0;
        // begin uses row 0 at i==0 (the begin source) and row 1 (-inf) thereafter.
        std::vector<int> sym_obs(Lm + 1, 0), sym_zero(Lm + 1, 0), sym_begin(Lm + 1, 1);
        sym_begin[0] = 0;  // only row 0 uses the begin (TBM) profile row
        for (int i = 0; i < Lm; ++i) sym_obs[i] = m.obs[i];"""


def _traceback_cpp() -> str:
    """O(L+M) greedy backtrack over the H matrix to produce an M/I/D state path
    string. The Viterbi *score* is the exact corner H[L-1][M]; this path is a
    reconstruction for display (it follows the highest-H predecessor among the
    diagonal/up/left moves), not necessarily the unique optimal path on ties."""
    return """
        // score = corner H[L-1][M] (max(M,I,D) at last residue, last node).
        score = (double)output[(std::size_t)(rows - 1) * phys_w + cols_m];
        std::string path;
        {
            int i = rows - 1, k = cols_m;
            auto Hv = [&](int ii, int kk) -> float {
                if (ii < 0 || kk < 0) return -std::numeric_limits<float>::infinity();
                return output[(std::size_t)ii * phys_w + kk];
            };
            while (i > 0 || k > 0) {
                float d = Hv(i - 1, k - 1), up = Hv(i - 1, k), lf = Hv(i, k - 1);
                if (k > 0 && lf >= d && lf >= up) { path.push_back('D'); --k; }
                else if (i > 0 && up >= d) { path.push_back('I'); --i; }
                else { path.push_back('M'); --i; --k; }
            }
            std::reverse(path.begin(), path.end());
        }"""


def _driver_source(*, header_path: str, kernel_name: str, out_cpp_type: str,
                   ele_per_iter: int, rows_per_iter: int, driver_name: str) -> str:
    out_per_row_init = "\n".join(
        f"                        out_buf[{r}]," for r in range(rows_per_iter)
    )
    # Each ProfileSignal2D input takes (profile, sym, stride); row_size last.
    ctor = ", ".join(
        f"{_VAR_PROFILE[nm][0]}.data(), {_VAR_PROFILE[nm][1]}.data(), phys_w" for nm in _INPUTS
    ) + ", phys_w"
    return f"""// Auto-generated by evals_common/vecrec_hmm_cpp.py. Do not edit by hand.
//
// Self-timed vecrec profile-HMM Viterbi kernel + traceback, reported via the shared
// baselines driver contract (float score) so it sits beside the HMMER baseline.
#include "{header_path}"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "driver_io.hpp"
#include "hmm_io.hpp"

using baselines::HmmModel;
using baselines::TimedRun;
using baselines::Timer;

int main(int argc, char** argv) {{
    return baselines::run_driver(argc, argv, "{driver_name}", [](const char* path) -> int {{
        const HmmModel m = baselines::parse_hmm(path);
        const int warmup = 3, iterations = 20;

        // Logical grid: rows = L (sequence), cols = M+1 (model nodes 0..M, node 0 is
        // the boundary). Physical row width rounded up to a multiple of ele_per_iter
        // so the kernel's block loop never reads past the buffers.
        constexpr int EPI = {ele_per_iter};
        const int rows = m.L;
        const int cols_m = m.M;                          // logical corner column
        const int phys_w = ((m.M + 1 + EPI - 1) / EPI) * EPI;
        const int N = rows * phys_w;

        double score = 0.0;
        baselines::AlignmentTraceback alignment;

        Timer timer;
        const TimedRun timed = timer.timed_run(warmup, iterations, [&] {{
            // --- build compact per-symbol profiles + sym tables (vecrec's setup cost) ---
            {{
{_build_profiles_cpp()}

            // --- vecrec kernel: fill the H = max(M,I,D) matrix (rows x phys_w) ---
            std::vector<{out_cpp_type}> output(static_cast<std::size_t>(N));
            auto kernel = {kernel_name}({ctor});
            int row_base = 0;
            while (row_base < N) {{
                int col = 0;
                while (col < phys_w) {{
                    {kernel_name}_vector_type out_buf[{rows_per_iter}][{ele_per_iter}/(sizeof({kernel_name}_vector_type)/sizeof({out_cpp_type}))];
                    {kernel_name}_vector_type* out_per_row[{rows_per_iter}] = {{
{out_per_row_init}
                    }};
                    kernel.run(out_per_row);
                    for (int rr = 0; rr < {rows_per_iter}; rr++) {{
                        int dst = row_base + rr * phys_w + col;
                        if (dst >= N) break;
                        int copy_count = ((N - dst) < {ele_per_iter}) ? (N - dst) : {ele_per_iter};
                        std::memcpy(output.data() + dst, out_buf[rr],
                                    sizeof({out_cpp_type}) * copy_count);
                    }}
                    col += {ele_per_iter};
                }}
                for (int rr = 0; rr < {rows_per_iter}; rr++) kernel.reset_and_next_row();
                row_base += {rows_per_iter} * phys_w;
            }}

            // --- traceback over H (sets `score` and the M/I/D path) ---
{_traceback_cpp()}
            alignment.query = path;
            }}
        }});

        baselines::print_hmm_result(score, timed, (long)rows * (long)cols_m, alignment.query);
        return 0;
    }});
}}
"""


_BUILD_CACHE: dict[str, Optional[Path]] = {}


def build_vecrec_hmm_driver(scheduled_expr, driver_name: str) -> Optional[Path]:
    """Generate + compile a self-timed vecrec HMM Viterbi driver. Returns its path
    (cached per ``driver_name``), or None if codegen/compile fails."""
    if driver_name in _BUILD_CACHE:
        return _BUILD_CACHE[driver_name]
    try:
        from vecrec.codegen import CodeGen, instantiate_kernels

        kernel_name = "k0"
        codegen = CodeGen()
        with contextlib.redirect_stdout(_io.StringIO()):
            code = codegen.generate(scheduled_expr, kernel_name)
        out_elt = scheduled_expr.element_type

        fd, header_path = tempfile.mkstemp(suffix=".h", prefix=f"{driver_name}_hdr_", dir="/tmp")
        os.close(fd)
        instantiate_kernels(header_path, [code])

        # Patch the generated header so every 2D input is read from a per-symbol
        # profile (ProfileSignal2D, in common.h) indexed by a per-row `sym` table,
        # instead of a full InputSignal2D over an O(L*M) array. vecrec codegen/AST are
        # untouched; this is a localized textual rewrite, like the alignment kernels.
        import re as _re

        hdr = Path(header_path).read_text()
        hdr = hdr.replace("InputSignal2D<", "ProfileSignal2D<")
        for var in _INPUTS:
            # member init:  v##(var, var_stride, ...) -> v##(var, var_sym, var_stride, ...)
            hdr = hdr.replace(f"({var}, {var}_stride,", f"({var}, {var}_sym, {var}_stride,")
            # ctor signature:  *var, int var_stride -> add const int *var_sym
            hdr = _re.sub(rf"\*{var}, int {var}_stride",
                          f"*{var}, const int *{var}_sym, int {var}_stride", hdr)
        Path(header_path).write_text(hdr)

        src = _driver_source(
            header_path=header_path,
            kernel_name=kernel_name,
            out_cpp_type=out_elt.to_str(),
            ele_per_iter=code.element_per_iter,
            rows_per_iter=code.rows_per_iter,
            driver_name=driver_name,
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
            *_platform_compiler_flags(),
            src_path, "-o", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            _BUILD_CACHE[driver_name] = None
            return None
    except Exception:
        _BUILD_CACHE[driver_name] = None
        return None

    _BUILD_CACHE[driver_name] = out_path
    return out_path


class VecrecHmmBaseline(CppBaseline):
    """A vecrec profile-HMM Viterbi driver (one schedule), compiled on first use.

    Unlike the alignment ``CppBaseline``, its input file is the compact model JSON
    (``inputs["prepared_json"]`` from the case's ``make_inputs``), and its score is
    a float (Viterbi log-odds in nats)."""

    def __init__(self, name: str, scheduled_expr) -> None:
        super().__init__(name=name, binary=name.replace(":", "_"), algo="hmm")
        self.scheduled_expr = scheduled_expr
        self._built: Optional[Path] = None

    def _ensure_built(self) -> Optional[Path]:
        if self._built is None:
            self._built = build_vecrec_hmm_driver(self.scheduled_expr, self.binary)
        return self._built

    def path(self) -> Path:
        built = self._ensure_built()
        return built if built is not None else _bin_dir() / self.binary

    def available(self) -> bool:
        built = self._ensure_built()
        return built is not None and built.exists() and os.access(built, os.X_OK)

    def run(self, inputs: dict) -> CppResult:
        import json

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write(inputs["prepared_json"])
            tmp_path = fh.name
        cmd = [str(self.path()), tmp_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if proc.returncode != 0:
            raise RuntimeError(f"{self.binary} exited {proc.returncode}\nstderr:\n{proc.stderr}")
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        data = json.loads(lines[-1])
        from evals_common.cpp import AlignmentStrings

        align = data.get("alignment")
        alignment = (
            AlignmentStrings(str(align.get("query", "")), str(align.get("ref", "")),
                             str(align.get("comp", "")))
            if isinstance(align, dict) else None
        )
        return CppResult(
            score=float(data["score"]),
            seconds=float(data["seconds"]),
            gcups=float(data["gcups"]),
            cells=int(data["cells"]),
            samples=[float(x) for x in data.get("samples", [])],
            alignment=alignment,
        )


def vecrec_hmm_baselines(expr) -> list[VecrecHmmBaseline]:
    """One :class:`VecrecHmmBaseline` per candidate schedule of ``expr``."""
    from vecrec.optimize import get_candidate_schedules
    from vecrec.util import get_native_bits

    try:
        max_bits = get_native_bits()
    except OSError:
        max_bits = 256
    schedules = get_candidate_schedules(expr, max_bits)
    return [VecrecHmmBaseline(f"vecrec_hmm_tb:k{i}", s) for i, s in enumerate(schedules)]
