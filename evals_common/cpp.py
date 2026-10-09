"""Bridge from the Python harness to compiled native baseline drivers.

A :class:`CppBaseline` names a binary under ``bin/`` plus an algorithm. At run
time it serializes the case inputs to a temporary ``input.json``, invokes the
binary as a subprocess, and parses the single JSON result line the driver prints
(see ``baselines/common/driver_io.hpp`` for the matching C++ side).

The driver self-times the alignment, so the harness uses the driver-reported
``seconds`` / ``gcups`` rather than timing the subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _bin_dir() -> Path:
    """Directory the build script installs driver binaries into (``<repo>/bin``)."""
    override = os.environ.get("VECREC_EVALS_BIN")
    if override:
        return Path(override)
    # evals_common/cpp.py -> repo root is two levels up (editable install).
    return Path(__file__).resolve().parents[1] / "bin"


@dataclass
class AlignmentStrings:
    query: str
    ref: str
    comp: str


@dataclass
class CppResult:
    score: float  # int for alignment cases; float (log-odds nats) for the HMM case
    seconds: float
    gcups: float
    cells: int
    samples: list[float] = field(default_factory=list)
    alignment: Optional[AlignmentStrings] = None


def _parse_driver_json(proc, binary: str, cmd: list, *, score_is_float: bool) -> "CppResult":
    """Parse the single JSON result line a driver prints (shared by CppBaseline and
    the HMM baselines). ``score_is_float`` keeps the alignment int contract intact."""
    if proc.returncode != 0:
        raise RuntimeError(
            f"{binary} exited {proc.returncode}\ncommand: {' '.join(cmd)}\n"
            f"stderr:\n{(proc.stderr or '').rstrip() or '(empty)'}"
        )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"{binary} produced no output\nstderr:\n{(proc.stderr or '').rstrip()}")
    data = json.loads(lines[-1])
    raw_alignment = data.get("alignment")
    alignment = (
        AlignmentStrings(
            query=str(raw_alignment.get("query", "")),
            ref=str(raw_alignment.get("ref", "")),
            comp=str(raw_alignment.get("comp", "")),
        )
        if isinstance(raw_alignment, dict) else None
    )
    return CppResult(
        score=float(data["score"]) if score_is_float else int(data["score"]),
        seconds=float(data["seconds"]),
        gcups=float(data["gcups"]),
        cells=int(data["cells"]),
        samples=[float(x) for x in data.get("samples", [])],
        alignment=alignment,
    )


@dataclass
class HmmCppBaseline:
    """The HMMER profile-HMM Viterbi library baseline (``baselines/hmmer/``).

    Rebuilds the same seeded model the case prepared and self-times p7_GViterbi +
    p7_GTrace; reports a float Viterbi score (log-odds nats). Its payload is the
    seed/M/L (not the alignment x/y/scoring), so it doesn't reuse ``CppBaseline.run``.
    """

    name: str
    binary: str = "hmmer_viterbi"
    algo: str = "hmm_run"
    warmup: int = 3
    iterations: int = 20

    def path(self) -> Path:
        return _bin_dir() / self.binary

    def available(self) -> bool:
        p = self.path()
        return p.exists() and os.access(p, os.X_OK)

    def run(self, inputs: dict) -> CppResult:
        payload = {
            "algo": self.algo,
            "seed": int(inputs["seed"]),
            "M": int(inputs["M"]),
            "L": int(inputs["L"]),
            "warmup": int(self.warmup),
            "iterations": int(self.iterations),
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(payload, fh)
            tmp_path = fh.name
        cmd = [str(self.path()), tmp_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return _parse_driver_json(proc, self.binary, cmd, score_is_float=True)


@dataclass
class CppBaseline:
    """A native C++ driver baseline.

    Attributes:
        name: Label shown in the results table (typically the binary name).
        binary: Executable name under ``bin/``.
        algo: Algorithm tag passed to the driver (``"sw"``, ``"nw"``, ...).
        returns: ``"score"`` -- compared to ``EvalCase.optimal_score``.
        warmup / iterations: Self-timing budget forwarded to the driver.
        check_correctness: When ``False``, the reported score is *not* compared
            to the reference (``correct`` stays ``None``, shown as ``-``). Use
            for timing-only baselines whose scoring semantics legitimately
            differ from the case reference (e.g. ksw2's extension-semantics
            SIMD kernel), so they still time but never gate the suite.
    """

    name: str
    binary: str
    algo: str
    returns: str = "score"
    warmup: int = 3
    iterations: int = 20
    check_correctness: bool = True
    extra_payload: dict = field(default_factory=dict)

    def path(self) -> Path:
        return _bin_dir() / self.binary

    def available(self) -> bool:
        p = self.path()
        return p.exists() and os.access(p, os.X_OK)

    def run(self, inputs: dict) -> CppResult:
        """Run the driver on the case inputs and return its parsed result.

        ``inputs`` must contain the sequence/scoring keys ``x``, ``y``,
        ``match``, ``mismatch``, ``gap_open``, ``gap_extend`` (the sequence
        bridge produced by the case's ``make_inputs``).
        """
        payload = {
            "algo": self.algo,
            "x": str(inputs["x"]),
            "y": str(inputs["y"]),
            "match": int(inputs["match"]),
            "mismatch": int(inputs["mismatch"]),
            "gap_open": int(inputs["gap_open"]),
            "gap_extend": int(inputs["gap_extend"]),
            "warmup": int(self.warmup),
            "iterations": int(self.iterations),
            **self.extra_payload,
        }

        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as fh:
            json.dump(payload, fh)
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
            raise RuntimeError(
                f"{self.binary} exited {proc.returncode}\n"
                f"command: {' '.join(cmd)}\n"
                f"stderr:\n{(proc.stderr or '').rstrip() or '(empty)'}\n"
                f"stdout:\n{(proc.stdout or '').rstrip() or '(empty)'}"
            )

        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError(
                f"{self.binary} produced no output\n"
                f"command: {' '.join(cmd)}\n"
                f"stderr:\n{(proc.stderr or '').rstrip() or '(empty)'}"
            )
        try:
            data = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{self.binary} returned invalid JSON: {exc}\n"
                f"last line: {lines[-1]!r}\n"
                f"stderr:\n{(proc.stderr or '').rstrip() or '(empty)'}"
            ) from exc
        samples = [float(x) for x in data.get("samples", [])]
        alignment = None
        raw_alignment = data.get("alignment")
        if isinstance(raw_alignment, dict):
            alignment = AlignmentStrings(
                query=str(raw_alignment.get("query", "")),
                ref=str(raw_alignment.get("ref", "")),
                comp=str(raw_alignment.get("comp", "")),
            )
        return CppResult(
            score=int(data["score"]),
            seconds=float(data["seconds"]),
            gcups=float(data["gcups"]),
            cells=int(data["cells"]),
            samples=samples,
            alignment=alignment,
        )
