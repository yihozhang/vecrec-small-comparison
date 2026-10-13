"""Real-world large-extension benchmark: divergent cross-species genome alignment.

This is the real-data analog of generate_sv_reads.py (Step 1). Aligning two
divergent bacterial genomes with `asm20` produces large gap-fill DPs: the
conserved core anchors, but the ~15-20% divergent segments between anchors have
no surviving seeds, so minimap2 fills them with large gap-fill DPs.

Default pair (downloaded from NCBI RefSeq, ~1.4 MB gzipped each):
  reference: Salmonella enterica LT2 (GCF_000006945.2)
  query:     Escherichia coli K-12   (GCF_000005845.2)
  divergence ~15-20% -> forces large divergent gap-fills.

Usage:
    uv run python end_to_end_baselines/scripts/run_crossspecies_bench.py [--reps 3]

Requires bin/minimap2_original and bin/minimap2_vecrec (build_minimap2.sh).

Fair comparison: stock minimap2 is run with MM_NO_APPROX=1 so its gap-fills use
the EXACT max path (matching the VecRec drop-in's exact full matrix); both use
-z/-r large to force the full, unbanded, no-z-drop regime the drop-in assumes.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import subprocess
import time

_REPO = pathlib.Path(__file__).resolve().parents[2]
_DATA = _REPO / "end_to_end_baselines" / "data" / "realbench"

GENOMES = {
    "ecoli.fna": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
                 "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz",
    "salmonella.fna": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/945/"
                      "GCF_000006945.2_ASM694v2/GCF_000006945.2_ASM694v2_genomic.fna.gz",
}
ASM_FLAGS = ["-cx", "asm20", "-t", "1", "-z", "1000000", "-r", "100000"]


def ensure_genomes() -> None:
    _DATA.mkdir(parents=True, exist_ok=True)
    for name, url in GENOMES.items():
        fna = _DATA / name
        if fna.exists():
            continue
        gz = fna.with_suffix(fna.suffix + ".gz")
        print(f"Downloading {name} from NCBI ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "120", "-o", str(gz), url], check=True)
        subprocess.run(["gunzip", "-f", str(gz)], check=True)


def med_ms(binary: pathlib.Path, ref: pathlib.Path, qry: pathlib.Path,
           env_extra: dict, reps: int) -> float:
    env = dict(os.environ); env.update(env_extra)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = subprocess.run([str(binary)] + ASM_FLAGS + [str(ref), str(qry)],
                           capture_output=True, text=True, env=env)
        ts.append(time.perf_counter() - t0)
        if r.returncode != 0:
            raise RuntimeError(f"{binary.name} exited {r.returncode}\n{r.stderr[:400]}")
    return statistics.median(ts) * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    ensure_genomes()
    ref = _DATA / "salmonella.fna"
    qry = _DATA / "ecoli.fna"
    orig = _REPO / "bin" / "minimap2_original"
    vrec = _REPO / "bin" / "minimap2_vecrec"
    for b in (orig, vrec):
        if not b.exists():
            raise SystemExit(f"missing {b}; run build_minimap2.sh first")

    print(f"asm20 cross-species  ref={ref.name}  query={qry.name}  reps={args.reps}")
    print(f"flags: {' '.join(ASM_FLAGS)}\n")

    v = med_ms(vrec, ref, qry, {}, args.reps)
    o_fair = med_ms(orig, ref, qry, {"MM_NO_APPROX": "1"}, args.reps)
    o_def = med_ms(orig, ref, qry, {}, args.reps)

    print(f"  minimap2-vecrec                 {v:9.1f} ms")
    print(f"  minimap2-original (fair,exact)  {o_fair:9.1f} ms   speedup {o_fair / v:.2f}x")
    print(f"  minimap2-original (approx-max)  {o_def:9.1f} ms   speedup {o_def / v:.2f}x")


if __name__ == "__main__":
    main()
