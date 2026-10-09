"""Regenerate the curated synthetic datasets for the minimap2 comparison harness.

Thin orchestrator over the per-dataset generators so the whole curated set can be
(re)produced with one command. Real datasets (parity-smoke's MT-human.fa, the
cross-species genomes) are committed / fetched separately and are not regenerated
here.

Usage:
    uv run python end_to_end_baselines/minimap2/make_datasets.py            # all synthetic
    uv run python end_to_end_baselines/minimap2/make_datasets.py sv-fair    # one dataset
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "end_to_end_baselines" / "scripts"

# dataset name -> generator script (run from repo root via uv)
RECIPES: dict[str, pathlib.Path] = {
    "parity-smoke": SCRIPTS / "generate_test_reads.py",   # mt_reads.fq from data/MT-human.fa
    "sv-fair": SCRIPTS / "generate_sv_reads.py",           # sv_reads.fq over yeast/chrXII.fa
    "tandem": SCRIPTS / "generate_tandem_reads.py",        # tandem_ref.fa + tandem_reads.fq
}


def run(name: str, script: pathlib.Path) -> None:
    print(f"\n=== {name}: {script.name} ===")
    subprocess.run(["uv", "run", "python", str(script)], cwd=REPO, check=True)


def main(argv: list[str]) -> int:
    names = argv or list(RECIPES)
    unknown = [n for n in names if n not in RECIPES]
    if unknown:
        print(f"unknown dataset(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(RECIPES)}", file=sys.stderr)
        return 2
    for name in names:
        run(name, RECIPES[name])
    print("\nDone. Curated synthetic datasets regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
