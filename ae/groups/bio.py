"""Table 3: VecRec alignment kernels vs vectorized libraries (group ``bio``).

Reuses the ``bio_kernel_evals`` harness: for each of the four recurrences the
harness compiles every candidate VecRec schedule at int8/int16/int32, runs
every C++ library driver under ``bin/`` (parasail, ksw2, seqan3, edlib, WFA2),
checks optimal-score agreement against a textbook DP, and self-times the whole
alignment including traceback. The two ``*Sim65`` / ``*Sim85`` cases are the
WFA2-only runs on 65% / 85% identity pairs that the paper reports separately.

Results: ``results/bio/<host>.csv`` (one row per (case, implementation);
schema in ``evals_common/results_csv.py``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ae.layout import BIO_SIZE
from ae.paths import BIN, ROOT, THIRD_PARTY, result_file

# Library drivers Table 3 needs (built by scripts/build_baselines.sh).
REQUIRED_BINARIES = ["seqan3_nw", "seqan3_sw", "parasail_align", "ksw2_align",
                     "edlib_editdist", "wfa2_align"]
# third_party/ clones those drivers compile against (scripts/setup_baselines.sh).
REQUIRED_LIBS = ["seqan3", "parasail", "ksw2", "edlib", "wfa2", "minimap2"]


def missing_binaries() -> list[str]:
    return [b for b in REQUIRED_BINARIES if not (BIN / b).exists()]


def setup() -> None:
    """Clone (pinned refs) and build the C++ baseline libraries."""
    libs = [lib for lib in REQUIRED_LIBS if not (THIRD_PARTY / lib).is_dir()]
    if libs:
        subprocess.run(["bash", str(ROOT / "scripts" / "setup_baselines.sh"), *libs],
                       cwd=ROOT, check=True)
    env = dict(os.environ)
    env.setdefault("AE_SKIP_HMMER", "1")  # HMMER is not part of any paper table
    subprocess.run(["bash", str(ROOT / "scripts" / "build_baselines.sh")],
                   cwd=ROOT, check=True, env=env)
    missing = missing_binaries()
    if missing:
        raise SystemExit(f"baseline build finished but drivers are missing: {missing}")


def cases(size: int) -> list:
    from bio_kernel_evals.cases.affine_gap import AffineGap, AffineGapSim65, AffineGapSim85
    from bio_kernel_evals.cases.edit_distance import EditDistance
    from bio_kernel_evals.cases.needleman_wunsch import (
        NeedlemanWunsch,
        NeedlemanWunschSim65,
        NeedlemanWunschSim85,
    )
    from bio_kernel_evals.cases.smith_waterman import SmithWaterman

    return [
        EditDistance(rows=size, cols=size),
        NeedlemanWunsch(rows=size, cols=size),
        SmithWaterman(rows=size, cols=size),
        AffineGap(rows=size, cols=size),
        NeedlemanWunschSim65(rows=size, cols=size),
        NeedlemanWunschSim85(rows=size, cols=size),
        AffineGapSim65(rows=size, cols=size),
        AffineGapSim85(rows=size, cols=size),
    ]


def exec_group(host: str, *, size: int = BIO_SIZE, only: list[str] | None = None) -> Path:
    from evals_common.harness import run_cases
    from evals_common.results_csv import ResultsCsvWriter

    if missing_binaries():
        print(f"[bio] missing drivers {missing_binaries()}; running setup first", flush=True)
        setup()

    selected = [c for c in cases(size) if not only or c.name in only]
    out = result_file("bio", host)
    tmp = out.with_suffix(".partial.csv")
    with ResultsCsvWriter(str(tmp)) as writer:
        code = run_cases(selected, on_results=writer.write_case)
    # Merge into the host file so `--only CASE` refreshes one case only.
    import csv

    from evals_common.results_csv import CSV_FIELDS

    from ae.paths import merge_csv

    with open(tmp, newline="") as f:
        rows = list(csv.DictReader(f))
    merge_csv(out, rows, CSV_FIELDS, key="display_name")
    tmp.unlink()
    print(f"wrote {out} (harness exit code {code})")
    return out
