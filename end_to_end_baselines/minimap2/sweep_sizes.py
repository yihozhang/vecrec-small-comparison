"""Sweep a minimap2 scenario across DP matrix sizes (core length).

Generates one SV-style dataset per --core value (generate_sv_reads.py; two
exact-match flanks around a divergent core of the given length, so the
gap-fill extension DP is ~core x core), then times original vs vecrec on
each, reusing run.py's build/timing machinery. bw stays at the scenario's
configured width (WIDE_BW for approx-max/exact-*) so no size in the sweep
gets clipped by banding -- only core length changes.

Usage:
    uv run python end_to_end_baselines/minimap2/sweep_sizes.py [scenario] \\
        [--cores 128,256,512,1024,2048,4096] [--reps 5] [--n-reads 30] \\
        [--p-sub 0.20] [--p-ins 0.08] [--p-del 0.08] \\
        [--parity] [--rebuild]

``--p-sub``/``--p-ins``/``--p-del`` control the divergent core's edit rate
(forwarded to generate_sv_reads.py). Lower divergence means the core's
*actual* achieved score sits closer to the naive "all-matches" upper bound
generate_kernel.py's I8 dispatch uses (min(qlen,tlen)*max_match), so it's
useful for probing how conservative/accurate that bound is against a scoring
scheme's real saturation point -- see the ``i8-forced`` scenario.

Datasets are cached under end_to_end_baselines/data/sv_sweep/ (gitignored),
keyed by core length + flank + divergence + n-reads + seed, and reused
across runs.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import statistics
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import datasets  # noqa: E402
import scenarios  # noqa: E402
import run as mm_run  # noqa: E402

REPO = _HERE.parents[1]
GEN_SV = REPO / "end_to_end_baselines" / "scripts" / "generate_sv_reads.py"
COMPARE_PY = REPO / "end_to_end_baselines" / "scripts" / "compare_e2e_sam.py"
SWEEP_DIR = REPO / "end_to_end_baselines" / "data" / "sv_sweep"
REF = REPO / "end_to_end_baselines" / "data" / "yeast" / "chrXII.fa"

_AS_PCT_RE = re.compile(r"same AS score\s*:\s*\d+/\d+\s*\(([\d.]+)%\)")


def gen_dataset(core: int, flank: int, n_reads: int, seed: int,
                p_sub: float, p_ins: float, p_del: float) -> datasets.Dataset:
    div_tag = f"_d{p_sub:g}-{p_ins:g}-{p_del:g}"
    out = SWEEP_DIR / f"sv_core{core}_flank{flank}{div_tag}_n{n_reads}_s{seed}.fq"
    if not out.exists():
        SWEEP_DIR.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["uv", "run", "python", str(GEN_SV),
             "--core", str(core), "--flank", str(flank),
             "--n-reads", str(n_reads), "--seed", str(seed),
             "--p-sub", str(p_sub), "--p-ins", str(p_ins), "--p-del", str(p_del),
             "--ref", str(REF), "--out", str(out)],
            cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"generate_sv_reads.py failed for core={core}:\n{r.stdout}\n{r.stderr}")
    edit = p_sub + p_ins + p_del
    return datasets.Dataset(
        name=f"sv-core{core}", ref=REF, reads=out, kind="synthetic",
        fair=True, desc=f"SV-style, core={core} (~{core}x{core} gap-fill DP), {edit:.0%} edit")


def run_parity_pct(sc: scenarios.Scenario, ds: datasets.Dataset) -> float | None:
    """Like run.py's run_parity, but captures the 'same AS score' percentage
    instead of only printing it, so sweeps can tabulate correctness alongside
    speed. Returns None if the check couldn't run (e.g. non-map-ont preset)."""
    if sc.preset != "map-ont":
        return None
    r = subprocess.run(
        ["uv", "run", "python", str(COMPARE_PY),
         "--ref", str(ds.ref), "--reads", str(ds.reads), "--zdrop", str(sc.zdrop)],
        cwd=REPO, capture_output=True, text=True)
    m = _AS_PCT_RE.search(r.stdout)
    return float(m.group(1)) if m else None


def time_one(sc: scenarios.Scenario, ds: datasets.Dataset, reps: int):
    mm_flags = mm_run._mm2_flags(sc)
    meds, alns = {}, {}
    for label, binary in (("orig", mm_run.ORIGINAL), ("vecrec", mm_run.VECREC)):
        cmd = [str(binary)] + mm_flags + [str(ds.ref), str(ds.reads)]
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True)
            times.append((time.perf_counter() - t0) * 1e3)
            if r.returncode != 0:
                raise SystemExit(f"{label} exited {r.returncode}\n{r.stderr[-1500:]}")
            alns[label] = mm_run._count_alns(r.stdout)
        meds[label] = statistics.median(times)
    speed = meds["orig"] / meds["vecrec"] if meds["vecrec"] else float("nan")
    return meds["orig"], meds["vecrec"], speed, alns["orig"], alns["vecrec"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", default="approx-max", help="scenario name (see run.py --list)")
    ap.add_argument("--cores", default="128,256,512,1024,2048,4096",
                    help="comma-separated core lengths (DP matrix ~= core x core)")
    ap.add_argument("--flank", type=int, default=600, help="exact-match flank length each side")
    ap.add_argument("--n-reads", type=int, default=30)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--p-sub", type=float, default=0.20, help="core substitution rate")
    ap.add_argument("--p-ins", type=float, default=0.08, help="core insertion rate")
    ap.add_argument("--p-del", type=float, default=0.08, help="core deletion rate")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--parity", action="store_true", help="also run a parity check at each size")
    args = ap.parse_args(argv)

    sc = scenarios.get(args.scenario)
    cores = [int(c) for c in args.cores.split(",")]
    edit = args.p_sub + args.p_ins + args.p_del

    print(f"=== size sweep: scenario={sc.name}  defines={' '.join(sc.defines) or 'default'}  "
          f"bw={sc.bw}  zdrop={sc.zdrop}  edit={edit:.0%} (sub={args.p_sub} ins={args.p_ins} del={args.p_del}) ===")
    mm_run.ensure_build(sc.defines, args.rebuild)

    rows = []
    for core in cores:
        ds = gen_dataset(core, args.flank, args.n_reads, args.seed, args.p_sub, args.p_ins, args.p_del)
        print(f"\n--- core={core}  (~{core}x{core} = {core*core:,} cells) ---")
        as_pct = run_parity_pct(sc, ds) if args.parity else None
        if as_pct is not None:
            print(f"  same AS score: {as_pct:.1f}%")
        o_med, v_med, speed, o_aln, v_aln = time_one(sc, ds, args.reps)
        ok = o_aln == v_aln
        print(f"  orig={o_med:.1f}ms  vecrec={v_med:.1f}ms  speedup={speed:.2f}x  "
              f"alns={o_aln}/{v_aln} [{'OK' if ok else 'MISMATCH'}]")
        rows.append((core, o_med, v_med, speed, ok, as_pct))

    print("\n=== summary ===")
    header = f"  {'core':>6s} {'cells':>14s} {'orig_ms':>9s} {'vecrec_ms':>10s} {'speedup':>9s} {'alns':>6s}"
    if args.parity:
        header += f" {'AS%':>7s}"
    print(header)
    for core, o_med, v_med, speed, ok, as_pct in rows:
        line = f"  {core:6d} {core*core:14,d} {o_med:9.1f} {v_med:10.1f} {speed:8.2f}x {'OK' if ok else 'DIFF':>6s}"
        if args.parity:
            line += f" {as_pct:6.1f}%" if as_pct is not None else f" {'-':>7s}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
