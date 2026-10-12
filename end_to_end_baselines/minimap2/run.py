"""Scenario-driven runner for the minimap2 VecRec comparison.

Usage:
    uv run python end_to_end_baselines/minimap2/run.py --list
    uv run python end_to_end_baselines/minimap2/run.py exact-sv [--reps N] [--rebuild]

For a scenario it (1) builds the stock + vecrec binaries with the scenario's
compile defines (skipped if the last build already used the same define-set),
then (2) collects the scenario's metrics:

  parity  — per-read SAM agreement (POS/MAPQ/AS/CIGAR) via compare_e2e_sam.py
  timing  — median wall-clock over --reps, both binaries, + speedup

Scenarios live in scenarios.py, datasets in datasets.py, compile flags in
flags.py. Add a comparison by adding a Scenario entry — no change here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import datasets  # noqa: E402
import flags  # noqa: E402
import scenarios  # noqa: E402

REPO = _HERE.parents[1]
BIN = REPO / "bin"
BUILD_SH = REPO / "end_to_end_baselines" / "scripts" / "build_minimap2.sh"
COMPARE_PY = REPO / "end_to_end_baselines" / "scripts" / "compare_e2e_sam.py"
ORIGINAL = BIN / "minimap2_original"
VECREC = BIN / "minimap2_vecrec"
_MARKER = BIN / ".mm2_build_defines"
SCHEDULE_META = _HERE / "generated" / "vecrec_schedule_search.json"


def _build_env(defines: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(flags.build_env(defines))
    zlib = REPO / "third_party" / "zlib_local"
    if zlib.exists() and "ZLIB_DIR" not in env:
        env["ZLIB_DIR"] = str(zlib)
    return env


def ensure_build(defines: list[str], rebuild: bool) -> None:
    # The schedule search is a build step. A change to its inputs or code must
    # invalidate the binary cache, otherwise an old fixed-schedule binary could
    # silently survive a new e2e run.
    build_env_inputs = {name: os.environ.get(name) for name in (
        "VECREC_MAX_BITS", "VECREC_TUNE_SIZE", "VECREC_TUNE_ITERS",
        "VECREC_SCHED_1P", "VECREC_SCHED_1P_I8", "VECREC_SCHED_2P",
        "VECREC_SCHED_2P_I8", "VECREC_MULT_1P", "VECREC_MULT_2P",
        "VECREC_RPI_1P", "VECREC_RPI_2P", "VECREC_PARALLEL_COMPILE",
        "VECREC_EXTRA_DEFS", "VECREC_PHASE_TIMING", "VECREC_UNSAFE",
        "MM_DEFS", "DEBUG", "ZLIB_DIR",
    )}
    source_paths = [BUILD_SH, BUILD_SH.with_name("generate_kernel.py"),
                    REPO / "vec-rec" / "src" / "vecrec" / "optimize.py",
                    REPO / "vec-rec" / "src" / "vecrec" / "codegen" / "generate_kernel.py"]
    source_hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_paths}
    key = json.dumps({"defines": sorted(defines), "build_env": build_env_inputs,
                      "sources": source_hashes}, sort_keys=True)
    have = ORIGINAL.exists() and VECREC.exists()
    cached = _MARKER.read_text().strip() if _MARKER.exists() else None
    if have and SCHEDULE_META.exists() and not rebuild and cached == key:
        print(f"[build] reusing binaries (defines: {','.join(sorted(defines)) or 'default'})")
        _print_schedules()
        return
    print(f"[build] searching VecRec schedules and building minimap2 "
          f"(defines: {','.join(sorted(defines)) or 'default'}) ...", flush=True)
    r = subprocess.run(["bash", str(BUILD_SH)], cwd=REPO, env=_build_env(defines),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:])
        raise SystemExit(f"[build] FAILED (exit {r.returncode})")
    _MARKER.write_text(key)
    print("[build] ok")
    _print_schedules()


def _print_schedules() -> None:
    kernels = json.loads(SCHEDULE_META.read_text())["kernels"]
    selected = ", ".join(
        f"{name}=k{kernels[name]['schedule_index']}"
        for name in ("2p_i16", "2p_i8", "1p_i16", "1p_i8")
    )
    print(f"[build] VecRec schedules: {selected}")


def _mm2_flags(sc: scenarios.Scenario) -> list[str]:
    f = ["-a", "-x", sc.preset, "-t", "1", "--secondary=no"]
    if sc.zdrop is not None:
        f += ["-z", f"{sc.zdrop},{sc.zdrop}"]
    # else: omit -z entirely -> minimap2 preset's real default z-drop applies
    # (can actually fire; used by the "dynamic approx" ablation tier).
    if sc.bw is not None:
        f += ["-r", f"{sc.bw},{sc.bw}"]
    return f


def _count_alns(sam: str) -> int:
    return sum(1 for ln in sam.splitlines() if ln and not ln.startswith("@"))


def run_timing(sc: scenarios.Scenario, ds: datasets.Dataset, reps: int) -> None:
    mm_flags = _mm2_flags(sc)
    print(f"\n[timing] reps={reps}  cmd: minimap2 {' '.join(mm_flags)} <ref> <reads>")
    rows = []
    for label, binary in (("minimap2_original", ORIGINAL), ("minimap2_vecrec", VECREC)):
        times, alns = [], None
        cmd = [str(binary)] + mm_flags + [str(ds.ref), str(ds.reads)]
        for _ in range(reps):
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True)
            dt = (time.perf_counter() - t0) * 1e3
            if r.returncode != 0:
                sys.stderr.write(r.stderr[-1500:])
                raise SystemExit(f"[timing] {label} exited {r.returncode}")
            times.append(dt)
            alns = _count_alns(r.stdout)
        rows.append((label, statistics.median(times), min(times), alns))
    print(f"  {'binary':20s} {'median_ms':>10s} {'min_ms':>9s} {'alns':>8s}")
    for label, med, mn, alns in rows:
        print(f"  {label:20s} {med:10.1f} {mn:9.1f} {alns:8d}")
    (_, o_med, _, o_aln), (_, v_med, _, v_aln) = rows
    speed = o_med / v_med if v_med else float("nan")
    match = "==" if o_aln == v_aln else "!="
    ok = "OK" if o_aln == v_aln else "MISMATCH"
    print(f"  speedup (orig/vecrec): {speed:.2f}x   alignments: {o_aln} {match} {v_aln}  [{ok}]")


def run_parity(sc: scenarios.Scenario, ds: datasets.Dataset) -> None:
    print("\n[parity] per-read SAM agreement (POS/MAPQ/AS/CIGAR):")
    cmd = ["uv", "run", "python", str(COMPARE_PY),
           "--ref", str(ds.ref), "--reads", str(ds.reads), "--preset", sc.preset]
    if sc.zdrop is not None:
        cmd += ["--zdrop", str(sc.zdrop)]
    if sc.bw is not None:
        cmd += ["--bw", str(sc.bw)]
    subprocess.run(cmd, cwd=REPO, check=False)


def cmd_list() -> None:
    print("Scenarios (end_to_end_baselines/minimap2/scenarios.py):\n")
    for sc in scenarios.SCENARIOS.values():
        ds = datasets.DATASETS.get(sc.dataset)
        flagstr = " ".join(sc.defines) or "default"
        tags = "/".join(sc.metrics)
        stub = "  [STUB]" if sc.stub else ""
        avail = "" if (ds and ds.present()) else "  (dataset MISSING)"
        print(f"  {sc.name:20s} dataset={sc.dataset:18s} metrics={tags:14s} defines={flagstr}{stub}{avail}")
        print(f"      {sc.desc}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", help="scenario name (see --list)")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--reps", type=int, default=5, help="timing repetitions (default 5)")
    ap.add_argument("--rebuild", action="store_true", help="force rebuild even if cached")
    args = ap.parse_args(argv)

    if args.list or not args.scenario:
        cmd_list()
        return 0

    sc = scenarios.get(args.scenario)
    if sc.stub:
        print(f"Scenario {sc.name!r} is a STUB (not runnable yet):\n  {sc.desc}")
        return 2
    ds = datasets.get(sc.dataset)
    if not ds.present():
        print(f"Dataset {ds.name!r} missing files:\n  ref={ds.ref}\n  reads={ds.reads}")
        print(f"Produce it with:\n  {ds.make}")
        return 2

    print(f"=== scenario: {sc.name} ===")
    print(f"dataset={sc.dataset}  preset={sc.preset}  bw={sc.bw}  zdrop={sc.zdrop}  "
          f"defines={' '.join(sc.defines) or 'default'}")
    ensure_build(sc.defines, args.rebuild)
    if "parity" in sc.metrics:
        run_parity(sc, ds)
    if "timing" in sc.metrics:
        run_timing(sc, ds, args.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
