"""Sweep harness for the REPTILE tiling study.

Two comparisons live here (``--family`` selects which run):

1. ``vecrec`` (default) — REPTILE tiling (row/col/block) wrapped around the
   compiled VecRec kernel vs. that same kernel run untiled (plain VecRec).
2. ``loops`` — REPTILE's tiled scalar DP loops (no VecRec; taken from
   REPTILE's artifact) vs VecRec untiled.

``--family both`` runs both and additionally reports an explicit
scalar-loops-vs-VecRec speedup (see ``vecrec_vs_scalar`` in the saved JSON).

For each recurrence (Needleman-Wunsch, affine-gap) it:
  1. Builds the VecRec external-mapping driver for every candidate schedule and
     picks the fastest schedule (the black box) via a quick untiled probe.
  2. Builds the REPTILE-generated tiled-loops driver (when family includes it).
  3. Sweeps { matrix size } x { tiling approach } x { tile size }, validating
     every configuration's alignment score against the textbook reference, and
     reports median runtime + speedup vs the untiled baseline of each family.

Results are printed as tables and saved under ``Reptile/results/``.

Usage (from repo root):
    uv run python Reptile/bench.py                 # full large sweep (vecrec family)
    uv run python Reptile/bench.py --sizes 512,1024 --reps 5
    uv run python Reptile/bench.py --kinds nw
    uv run python Reptile/bench.py --family both   # + scalar loops vs VecRec
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Optional

from Reptile import reptile

# Default sweep axes (the "large" study: sizes where locality begins to matter).
# Include paper/SEQ-scale tiles (4..32) plus larger ones; tiny 2D blocks are
# what the REPTILE NW/SW figures use.
DEFAULT_SIZES = (512, 1024, 2048, 4096)
DEFAULT_TILES = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
SELECT_SIZE = 1024  # size used to pick the fastest VecRec schedule


@dataclass
class RunResult:
    score: int
    seconds: float
    samples: list = field(default_factory=list)
    checksum: Optional[int] = None


def _write_input(inputs: dict, approach: str, tile_rows: int, tile_cols: int,
                 warmup: int, iterations: int, verify: int, score_only: int = 0) -> str:
    payload = {
        "algo": "nw",
        "x": str(inputs["x"]),
        "y": str(inputs["y"]),
        "match": int(inputs["match"]),
        "mismatch": int(inputs["mismatch"]),
        "gap_open": int(inputs["gap_open"]),
        "gap_extend": int(inputs["gap_extend"]),
        "approach": approach,
        "tile_rows": int(tile_rows),
        "tile_cols": int(tile_cols),
        "warmup": int(warmup),
        "iterations": int(iterations),
        "verify": int(verify),
        "score_only": int(score_only),
    }
    fd, path = tempfile.mkstemp(suffix=".json", prefix="reptile_in_", dir="/tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh)
    return path


def run_driver(binary: Path, inputs: dict, approach: str, *, tile_rows: int = 0,
               tile_cols: int = 0, warmup: int = 3, iterations: int = 15,
               verify: bool = False, score_only: bool = False) -> RunResult:
    path = _write_input(inputs, approach, tile_rows, tile_cols, warmup, iterations,
                        int(verify), int(score_only))
    try:
        proc = subprocess.run([str(binary), path], capture_output=True, text=True)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary.name} exited {proc.returncode}\nstderr:\n{proc.stderr}")
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    data = json.loads(lines[-1])
    checksum = None
    for ln in (proc.stderr or "").splitlines():
        if ln.startswith("REPTILE_CHECKSUM "):
            checksum = int(ln.split()[1])
    return RunResult(
        score=int(data["score"]),
        seconds=float(data["seconds"]),
        samples=[float(x) for x in data.get("samples", [])],
        checksum=checksum,
    )


# --------------------------------------------------------------------------

def _case(kind: str, n: int):
    if kind == "nw":
        from bio_kernel_evals.cases.needleman_wunsch import NeedlemanWunsch

        return NeedlemanWunsch(rows=n, cols=n, seed=0)
    if kind == "sw":
        from bio_kernel_evals.cases.smith_waterman import SmithWaterman

        return SmithWaterman(rows=n, cols=n, seed=0)
    from bio_kernel_evals.cases.affine_gap import AffineGap

    return AffineGap(rows=n, cols=n, seed=0)


def _reference_score(kind: str, n: int) -> tuple[dict, int]:
    case = _case(kind, n)
    inputs = case.make_inputs()
    ref = case.reference(**inputs)
    return inputs, int(case.optimal_score(ref))


def build_fastest_vecrec(kind: str, select_size: int, warmup: int, iterations: int) -> Optional[tuple[Path, str]]:
    """Compile every candidate VecRec schedule and return (binary, label) of the fastest untiled one."""
    scheds = reptile.candidate_schedules(kind)
    inputs, expected = _reference_score(kind, select_size)
    best: Optional[tuple[float, Path, str]] = None
    k = 0
    print(f"  [{kind}] selecting fastest VecRec schedule over {len(scheds) * 2} candidates "
          f"at size {select_size} ...")
    for mult in (2, 4):
        for sched in scheds:
            label = f"k{k}(mult{mult})"
            name = f"reptile_vecrec_{kind}_k{k}"
            k += 1
            try:
                binary = reptile.build_vecrec_driver(kind, sched, mult, name)
            except Exception as exc:
                print(f"      {label}: build failed ({str(exc).splitlines()[0]})")
                continue
            if binary is None:
                print(f"      {label}: codegen skipped")
                continue
            try:
                res = run_driver(binary, inputs, "untiled", warmup=warmup, iterations=iterations)
            except Exception as exc:
                print(f"      {label}: run failed ({str(exc).splitlines()[0]})")
                continue
            ok = res.score == expected
            tag = "OK" if ok else f"SCORE {res.score}!={expected}"
            print(f"      {label}: {res.seconds*1e3:.3f} ms  [{tag}]")
            if ok and (best is None or res.seconds < best[0]):
                best = (res.seconds, binary, label)
    if best is None:
        return None
    print(f"  [{kind}] fastest VecRec schedule: {best[2]} ({best[0]*1e3:.3f} ms)")
    return best[1], best[2]


def build_all_vecrec(kind: str, select_size: int, warmup: int, iterations: int) -> list[tuple[Path, str]]:
    """Compile *every* candidate VecRec schedule and return (binary, label) for
    each one that builds and produces the correct untiled score. Unlike
    build_fastest_vecrec (which keeps only the untiled winner), this keeps all of
    them so the caller can tile each schedule and ask whether the best *tiled*
    schedule is the same as the best *untiled* one.
    """
    scheds = reptile.candidate_schedules(kind)
    inputs, expected = _reference_score(kind, select_size)
    built: list[tuple[Path, str]] = []
    k = 0
    print(f"  [{kind}] building all VecRec schedules ({len(scheds) * 2} candidates) "
          f"at size {select_size} ...")
    for mult in (2, 4):
        for sched in scheds:
            label = f"k{k}(mult{mult})"
            name = f"reptile_vecrec_{kind}_k{k}"
            k += 1
            try:
                binary = reptile.build_vecrec_driver(kind, sched, mult, name)
            except Exception as exc:
                print(f"      {label}: build failed ({str(exc).splitlines()[0]})")
                continue
            if binary is None:
                print(f"      {label}: codegen skipped")
                continue
            try:
                res = run_driver(binary, inputs, "untiled", warmup=warmup, iterations=iterations)
            except Exception as exc:
                print(f"      {label}: run failed ({str(exc).splitlines()[0]})")
                continue
            if res.score != expected:
                print(f"      {label}: SCORE {res.score}!={expected} (dropped)")
                continue
            print(f"      {label}: probe {res.seconds*1e3:.3f} ms  [OK]")
            built.append((binary, label))
    print(f"  [{kind}] {len(built)} schedules usable")
    return built


# --------------------------------------------------------------------------

def verify(kind: str, sizes, tiles, warmup: int, iterations: int, family: str = "both") -> bool:
    """Prove tiling correctness: within each family, every tiled variant must
    produce a bit-identical full DP matrix to that family's untiled one (equal
    checksum), and every variant's score must equal the textbook reference.
    """
    print(f"\n=== REPTILE tiling verification: {kind} ===")
    vr = build_fastest_vecrec(kind, SELECT_SIZE, warmup=1, iterations=3) if family in ("both", "vecrec") else None
    loops_bin = (reptile.build_loops_driver(kind, f"reptile_loops_{kind}")
                 if family in ("both", "loops") else None)
    all_ok = True
    for n in sizes:
        inputs, expected = _reference_score(kind, n)
        print(f"\n  --- {kind} {n}x{n} (reference score {expected}) ---")

        # VecRec family: untiled vs every row/col/block tile size. Every tiled
        # variant must produce a bit-identical full matrix to VecRec untiled.
        if vr is not None:
            vr_bin, _ = vr
            base = run_driver(vr_bin, inputs, "untiled", warmup=warmup,
                              iterations=iterations, verify=True)
            base_ok = base.score == expected
            print(f"    VecRec untiled: score={base.score} (ref {expected}) "
                  f"checksum={base.checksum} [{'OK' if base_ok else 'SCORE FAIL'}]")
            all_ok &= base_ok
            for approach in ("row", "col", "block"):
                for t in tiles:
                    tr = t if approach in ("row", "block") else 0
                    tc = t if approach in ("col", "block") else 0
                    r = run_driver(vr_bin, inputs, approach, tile_rows=tr, tile_cols=tc,
                                   warmup=warmup, iterations=iterations, verify=True)
                    same = r.checksum == base.checksum
                    sok = r.score == expected
                    all_ok &= same and sok
                    print(f"      {approach+' t='+str(t):<14} score={r.score} "
                          f"matrix={'IDENTICAL' if same else 'DIFF!'} "
                          f"[{'OK' if same and sok else 'FAIL'}]")

        # Loops family (vanilla scalar, no VecRec): untiled vs row/col/block.
        if loops_bin is not None:
            lbase = run_driver(loops_bin, inputs, "untiled", warmup=warmup,
                               iterations=iterations, verify=True)
            lbase_ok = lbase.score == expected
            print(f"    loops untiled: score={lbase.score} (ref {expected}) "
                  f"checksum={lbase.checksum} [{'OK' if lbase_ok else 'SCORE FAIL'}]")
            all_ok &= lbase_ok
            for approach in ("row", "col", "block"):
                for t in tiles:
                    tr = t if approach in ("row", "block") else 0
                    tc = t if approach in ("col", "block") else 0
                    r = run_driver(loops_bin, inputs, approach, tile_rows=tr, tile_cols=tc,
                                   warmup=warmup, iterations=iterations, verify=True)
                    same = r.checksum == lbase.checksum
                    sok = r.score == expected
                    all_ok &= same and sok
                    print(f"      {approach+' t='+str(t):<14} score={r.score} "
                          f"matrix={'IDENTICAL' if same else 'DIFF!'} "
                          f"[{'OK' if same and sok else 'FAIL'}]")
    print(f"\n  [{kind}] verification: {'ALL PASS' if all_ok else 'FAILURES DETECTED'}")
    return all_ok


def _median_over_reps(binary: Path, inputs: dict, approach: str, *, tile_rows: int = 0,
                      tile_cols: int = 0, reps: int = 3, warmup: int = 3,
                      iterations: int = 15, score_only: bool = False) -> RunResult:
    best: Optional[RunResult] = None
    all_samples: list = []
    score = 0
    secs: list = []
    for _ in range(reps):
        r = run_driver(binary, inputs, approach, tile_rows=tile_rows, tile_cols=tile_cols,
                       warmup=warmup, iterations=iterations, score_only=score_only)
        secs.append(r.seconds)
        all_samples.extend(r.samples)
        score = r.score
    return RunResult(score=score, seconds=median(secs), samples=all_samples)


def sweep(kind: str, sizes, tiles, reps: int, warmup: int, iterations: int,
          family: str = "vecrec", score_only: bool = False) -> dict:
    print(f"\n=== REPTILE tiling sweep: {kind} ===")
    vr = build_fastest_vecrec(kind, SELECT_SIZE, warmup=2, iterations=8) if family in ("both", "vecrec") else None
    loops_bin = (reptile.build_loops_driver(kind, f"reptile_loops_{kind}")
                 if family in ("both", "loops") else None)

    records = []
    for n in sizes:
        inputs, expected = _reference_score(kind, n)
        row: dict = {"size": n, "expected": expected, "configs": {}}

        # VecRec family: untiled baseline (plain VecRec) + row/col/block tiling
        # of the same kernel.
        if vr is not None:
            vr_bin, vr_label = vr
            base = _median_over_reps(vr_bin, inputs, "untiled", reps=reps,
                                     warmup=warmup, iterations=iterations,
                                     score_only=score_only)
            row["configs"]["vecrec_untiled"] = {
                "seconds": base.seconds, "score": base.score,
                "ok": base.score == expected, "speedup_vs_family": 1.0,
                "schedule": vr_label,
            }
            for approach in ("row", "col", "block"):
                for t in tiles:
                    tr = t if approach in ("row", "block") else 0
                    tc = t if approach in ("col", "block") else 0
                    r = _median_over_reps(vr_bin, inputs, approach, tile_rows=tr,
                                          tile_cols=tc, reps=reps, warmup=warmup,
                                          iterations=iterations, score_only=score_only)
                    row["configs"][f"vecrec_{approach}_t{t}"] = {
                        "seconds": r.seconds, "score": r.score, "ok": r.score == expected,
                        "speedup_vs_family": base.seconds / r.seconds if r.seconds > 0 else 0.0,
                    }

        # Loops family: vanilla scalar REPTILE-generated tiled loops (no VecRec).
        # Untiled baseline + row/col/block tiling of the same scalar loop nest.
        if loops_bin is not None:
            lbase = _median_over_reps(loops_bin, inputs, "untiled", reps=reps,
                                      warmup=warmup, iterations=iterations)
            row["configs"]["loops_untiled"] = {
                "seconds": lbase.seconds, "score": lbase.score,
                "ok": lbase.score == expected, "speedup_vs_family": 1.0,
            }
            for approach in ("row", "col", "block"):
                for t in tiles:
                    tr = t if approach in ("row", "block") else 0
                    tc = t if approach in ("col", "block") else 0
                    r = _median_over_reps(loops_bin, inputs, approach, tile_rows=tr,
                                          tile_cols=tc, reps=reps, warmup=warmup,
                                          iterations=iterations)
                    row["configs"][f"loops_{approach}_t{t}"] = {
                        "seconds": r.seconds, "score": r.score, "ok": r.score == expected,
                        "speedup_vs_family": lbase.seconds / r.seconds if r.seconds > 0 else 0.0,
                    }

        # Cross-family verdict: scalar loops (best tiled or untiled config,
        # whichever is fastest) vs plain VecRec untiled.
        vecrec_entry = row["configs"].get("vecrec_untiled")
        loops_entries = {k: v for k, v in row["configs"].items()
                         if k.startswith("loops_") and v["ok"]}
        if vecrec_entry is not None and loops_entries:
            vecrec_s = vecrec_entry["seconds"]
            best_loops_key = min(loops_entries, key=lambda k: loops_entries[k]["seconds"])
            best_loops_s = loops_entries[best_loops_key]["seconds"]
            row["vecrec_vs_scalar"] = {
                "best_loops_config": best_loops_key,
                "best_loops_seconds": best_loops_s,
                "vecrec_untiled_seconds": vecrec_s,
                "vecrec_speedup": best_loops_s / vecrec_s if vecrec_s > 0 else 0.0,
            }

        records.append(row)
        _print_size_table(kind, row, tiles)
    return {"kind": kind, "sizes": list(sizes), "tiles": list(tiles), "records": records}


def sweep_all_schedules(kind: str, sizes, tiles, reps: int, warmup: int,
                        iterations: int, score_only: bool = False) -> dict:
    """Tile *every* VecRec schedule (not just the untiled-fastest one) and report,
    per size, whether the best UNTILED schedule is also the best TILED schedule.
    """
    print(f"\n=== REPTILE per-schedule tiling sweep: {kind} ===")
    built = build_all_vecrec(kind, SELECT_SIZE, warmup=2, iterations=8)
    if not built:
        print(f"  [{kind}] no schedules built; skipping")
        return {"kind": kind, "sizes": list(sizes), "tiles": list(tiles), "records": []}

    records = []
    for n in sizes:
        inputs, expected = _reference_score(kind, n)
        per_sched: dict = {}
        for binary, label in built:
            u = _median_over_reps(binary, inputs, "untiled", reps=reps, warmup=warmup,
                                  iterations=iterations, score_only=score_only)
            entry = {"untiled": u.seconds, "ok": u.score == expected, "tiled": {}}
            for approach in ("row", "col", "block"):
                for t in tiles:
                    tr = t if approach in ("row", "block") else 0
                    tc = t if approach in ("col", "block") else 0
                    r = _median_over_reps(binary, inputs, approach, tile_rows=tr,
                                          tile_cols=tc, reps=reps, warmup=warmup,
                                          iterations=iterations, score_only=score_only)
                    entry["tiled"][f"{approach}_t{t}"] = {
                        "seconds": r.seconds, "ok": r.score == expected,
                    }
            per_sched[label] = entry
        records.append({"size": n, "expected": expected, "schedules": per_sched})
        _print_all_sched_table(kind, n, per_sched, tiles)
    return {"kind": kind, "sizes": list(sizes), "tiles": list(tiles),
            "mode": "all_schedules", "records": records}


def _best_tiled_of(entry: dict) -> tuple[Optional[str], float]:
    """(config_name, seconds) of the fastest valid tiled config for a schedule."""
    valid = {k: v["seconds"] for k, v in entry["tiled"].items() if v["ok"]}
    if not valid:
        return None, float("inf")
    bk = min(valid, key=valid.get)
    return bk, valid[bk]


def _print_all_sched_table(kind: str, n: int, per_sched: dict, tiles) -> None:
    print(f"\n  --- {kind} {n}x{n} : per-schedule untiled vs tiled ---")
    print(f"    {'schedule':<14} {'untiled(ms)':>11} {'bestTiled(ms)':>13} "
          f"{'best config':<14} {'tiled/untiled':>13}")
    # Rank schedules by untiled speed for readability.
    order = sorted(per_sched, key=lambda l: per_sched[l]["untiled"])
    best_untiled_label = order[0]
    best_tiled_label, best_tiled_cfg, best_tiled_sec = None, None, float("inf")
    for label in order:
        e = per_sched[label]
        cfg, sec = _best_tiled_of(e)
        ratio = e["untiled"] / sec if sec > 0 and sec != float("inf") else 0.0
        mark = "  <- fastest untiled" if label == best_untiled_label else ""
        print(f"    {label:<14} {e['untiled']*1e3:>11.3f} "
              f"{(sec*1e3 if sec != float('inf') else float('nan')):>13.3f} "
              f"{(cfg or '-'):<14} {ratio:>12.2f}x{mark}")
        if sec < best_tiled_sec:
            best_tiled_label, best_tiled_cfg, best_tiled_sec = label, cfg, sec
    same = best_tiled_label == best_untiled_label
    print(f"    => best UNTILED schedule : {best_untiled_label} "
          f"({per_sched[best_untiled_label]['untiled']*1e3:.3f} ms)")
    print(f"    => best TILED   overall  : {best_tiled_label} {best_tiled_cfg} "
          f"({best_tiled_sec*1e3:.3f} ms)")
    print(f"    => best tiled schedule == best untiled schedule?  "
          f"{'YES' if same else 'NO'}")
    if not same:
        bu = per_sched[best_untiled_label]
        print(f"       (best-untiled {best_untiled_label} tiled best = "
              f"{_best_tiled_of(bu)[1]*1e3:.3f} ms; overall-tiled winner is "
              f"{(bu['untiled']/best_tiled_sec):.3f}x faster than it)")


def _save_all_schedules(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out_dir / f"schedules_{stamp}.json").write_text(json.dumps(results, indent=2))
    lines = ["# REPTILE per-schedule tiling (best untiled vs best tiled)", "",
             f"Generated {stamp}", ""]
    for res in results:
        lines += [f"## {res['kind']}", "",
                  "| size | best untiled sched | untiled (ms) | best tiled sched | "
                  "best tiled config | tiled (ms) | same? |",
                  "|---|---|---|---|---|---|---|"]
        for row in res["records"]:
            ps = row["schedules"]
            bu = min(ps, key=lambda l: ps[l]["untiled"])
            bt_label, bt_cfg, bt_sec = None, None, float("inf")
            for label, e in ps.items():
                cfg, sec = _best_tiled_of(e)
                if sec < bt_sec:
                    bt_label, bt_cfg, bt_sec = label, cfg, sec
            lines.append(
                f"| {row['size']} | {bu} | {ps[bu]['untiled']*1e3:.3f} | {bt_label} | "
                f"{bt_cfg} | {bt_sec*1e3:.3f} | {'YES' if bt_label == bu else 'NO'} |")
        lines.append("")
    (out_dir / f"schedules_{stamp}.md").write_text("\n".join(lines))
    print(f"\nSaved per-schedule results to {out_dir}/ (stamp {stamp})")


def _fmt_ms(sec: Optional[float]) -> str:
    return "n/a" if sec is None else f"{sec*1e3:.3f}"


def _fmt_x(baseline: Optional[float], seconds: Optional[float]) -> str:
    """Speedup factor: baseline / seconds. '-' if either missing."""
    if baseline is None or seconds is None or seconds <= 0:
        return "-"
    return f"{baseline / seconds:.2f}x"


def _fmt_ok(ok: Optional[bool]) -> str:
    return "-" if ok is None else ("Y" if ok else "N")


def _best_approach(cfg: dict, prefix: str, approach: str) -> Optional[tuple[str, dict]]:
    """(config_key, entry) of the fastest *ok* tiled config of a given approach.

    e.g. prefix='vecrec_', approach='col' -> fastest of vecrec_col_t{...}. None if
    nothing valid was measured.
    """
    cands = {k: v for k, v in cfg.items()
             if k.startswith(f"{prefix}{approach}_t") and v.get("ok")}
    if not cands:
        return None
    bk = min(cands, key=lambda k: cands[k]["seconds"])
    return bk, cands[bk]


def _tile_of(key: str) -> str:
    """'vecrec_col_t512' -> '512'."""
    return key.rsplit("_t", 1)[-1]


# Canonical row layout for the per-size strategy table. ALWAYS emitted in this
# exact order so the table has an identical shape every run (missing data shows
# as n/a rather than dropping a row). Each tuple: (strategy, prefix, tiling).
_TABLE_LAYOUT = (
    ("Reptile (scalar)", "loops_",  "untiled"),
    ("Reptile (scalar)", "loops_",  "row"),
    ("Reptile (scalar)", "loops_",  "col"),
    ("Reptile (scalar)", "loops_",  "block"),
    ("VecRec",           "vecrec_", "untiled"),
    ("VecRec + Reptile", "vecrec_", "row"),
    ("VecRec + Reptile", "vecrec_", "col"),
    ("VecRec + Reptile", "vecrec_", "block"),
)


def _detailed_rows(cfg: dict) -> list[dict]:
    """Build the unified per-size strategy table as a FIXED 8-row structure.

    Rows are always the same, in the same order (see ``_TABLE_LAYOUT``): the
    three recurrence strategies (Reptile scalar / VecRec / VecRec+Reptile) times
    their tilings. Missing measurements become ``seconds=None`` (rendered n/a) so
    the table's shape never changes between runs. Each row carries raw time plus
    two speedup factors against fixed baselines:
      * vs Reptile = loops_untiled / time
      * vs VecRec  = vecrec_untiled / time
    For row/col/block the fastest tile found is used and reported in ``tile``.
    """
    reptile_base = cfg.get("loops_untiled", {}).get("seconds")
    vecrec_base = cfg.get("vecrec_untiled", {}).get("seconds")

    rows: list[dict] = []
    for strategy, prefix, tiling in _TABLE_LAYOUT:
        if tiling == "untiled":
            entry = cfg.get(f"{prefix}untiled")
            tile = "-"
        else:
            best = _best_approach(cfg, prefix, tiling)
            entry = best[1] if best is not None else None
            tile = _tile_of(best[0]) if best is not None else "-"
        seconds = entry["seconds"] if entry is not None else None
        ok = entry.get("ok", True) if entry is not None else None
        rows.append({
            "strategy": strategy,
            "tiling": tiling,
            "tile": tile,
            "seconds": seconds,
            "vs_reptile": _fmt_x(reptile_base, seconds),
            "vs_vecrec": _fmt_x(vecrec_base, seconds),
            "ok": ok,
            "pivot": (strategy == "VecRec"),
        })
    return rows


# Column widths shared by the stdout renderer (kept in one place so header and
# body always line up).
_COLW = {"strategy": 18, "tiling": 8, "tile": 6, "time": 10, "vsr": 11, "vsv": 10}


def _print_size_table(kind: str, row: dict, tiles) -> None:
    n = row["size"]
    cfg = row["configs"]
    sched = cfg.get("vecrec_untiled", {}).get("schedule", "?")
    w = _COLW
    print(f"\n  --- {kind} {n}x{n} (reference score {row['expected']}, "
          f"VecRec schedule {sched}) ---")
    header = (f"      {'strategy':<{w['strategy']}} {'tiling':<{w['tiling']}} "
              f"{'tile':>{w['tile']}} {'time(ms)':>{w['time']}} "
              f"{'vs Reptile':>{w['vsr']}} {'vs VecRec':>{w['vsv']}}  ok")
    print(header)
    print("      " + "-" * (len(header) - 6))
    prev_strategy = None
    for r in _detailed_rows(cfg):
        # Blank separator between strategy groups for readability.
        if prev_strategy is not None and r["strategy"] != prev_strategy:
            print()
        prev_strategy = r["strategy"]
        marker = "*" if r["pivot"] else " "
        print(f"    {marker} {r['strategy']:<{w['strategy']}} {r['tiling']:<{w['tiling']}} "
              f"{r['tile']:>{w['tile']}} {_fmt_ms(r['seconds']):>{w['time']}} "
              f"{r['vs_reptile']:>{w['vsr']}} {r['vs_vecrec']:>{w['vsv']}}  {_fmt_ok(r['ok'])}")
    print("      (* = pivot baseline; 'vs Reptile' = scalar untiled time / this time; "
          "'vs VecRec' = VecRec untiled time / this time)")


def _save(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out_dir / f"results_{stamp}.json").write_text(json.dumps(results, indent=2))
    # Markdown summary: one unified detailed table per (kind, size). Rows group
    # the three strategies (Reptile scalar / VecRec / VecRec+Reptile); columns
    # carry raw time plus two speedup factors ('vs Reptile', 'vs VecRec').
    lines = ["# REPTILE tiling results", "",
             f"Generated {stamp}",
             "",
             "`vs Reptile` = scalar (Reptile) untiled time / this time. "
             "`vs VecRec` = VecRec untiled time / this time. "
             "`*` marks the VecRec pivot.",
             ""]
    for res in results:
        kind = res["kind"]
        lines.append(f"## {kind}")
        lines.append("")
        for row in res["records"]:
            n = row["size"]
            cfg = row["configs"]
            sched = cfg.get("vecrec_untiled", {}).get("schedule", "?")
            drows = _detailed_rows(cfg)
            lines.append(f"### {kind} {n}x{n} (reference score {row['expected']}, "
                         f"VecRec schedule {sched})")
            lines.append("")
            lines.append("| Strategy | Tiling | Tile | Time (ms) | vs Reptile | vs VecRec |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for r in drows:
                strat = f"**{r['strategy']}**" if r["pivot"] else r["strategy"]
                tiling = f"**{r['tiling']}**" if r["pivot"] else r["tiling"]
                lines.append(
                    f"| {strat} | {tiling} | {r['tile']} | {_fmt_ms(r['seconds'])} | "
                    f"{r['vs_reptile']} | {r['vs_vecrec']} |")
            lines.append("")
    (out_dir / f"summary_{stamp}.md").write_text("\n".join(lines))
    print(f"\nSaved results + summary to {out_dir}/ (stamp {stamp})")


def main() -> int:
    ap = argparse.ArgumentParser(description="REPTILE tiling sweep around the VecRec black box")
    ap.add_argument("--kinds", default="nw,affine", help="comma list: nw,affine")
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--tiles", default=",".join(map(str, DEFAULT_TILES)))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--verify", action="store_true",
                    help="prove tiling correctness (bit-identical matrix vs untiled) and exit")
    ap.add_argument("--family", choices=("both", "vecrec", "loops"), default="vecrec",
                    help="which driver family to sweep. Default 'vecrec' = REPTILE "
                         "tiling (row/col/block) around the VecRec black box vs. plain "
                         "VecRec untiled. 'loops' = the "
                         "REPTILE's tiled scalar loops (no VecRec), tiled vs "
                         "untiled scalar. 'both' runs both AND reports the cross-family "
                         "comparison: best scalar-loops config vs plain VecRec untiled.")
    ap.add_argument("--score-only", action="store_true", dest="score_only",
                    help="score-only mode for the VecRec family: never materialize the "
                         "O(N^2) matrix (no DRAM writeback), keep only the final score.")
    ap.add_argument("--all-schedules", action="store_true", dest="all_schedules",
                    help="tile EVERY VecRec schedule (not just the untiled-fastest one) "
                         "and report, per size, whether the best untiled schedule is also "
                         "the best tiled schedule.")
    args = ap.parse_args()

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    tiles = [int(s) for s in args.tiles.split(",") if s.strip()]

    if args.verify:
        ok = True
        for kind in kinds:
            ok &= verify(kind, sizes, tiles, args.warmup, min(args.iterations, 2), family=args.family)
        return 0 if ok else 1

    if args.all_schedules:
        results = []
        for kind in kinds:
            results.append(sweep_all_schedules(kind, sizes, tiles, args.reps,
                                               args.warmup, args.iterations,
                                               score_only=args.score_only))
        _save_all_schedules(results, reptile.repo_root() / "Reptile" / "results")
        return 0

    results = []
    for kind in kinds:
        results.append(sweep(kind, sizes, tiles, args.reps, args.warmup,
                             args.iterations, family=args.family,
                             score_only=args.score_only))
    _save(results, reptile.repo_root() / "Reptile" / "results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
