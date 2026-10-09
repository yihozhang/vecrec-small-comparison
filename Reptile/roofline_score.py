"""Roofline test: does column tiling help score-only VecRec once the wavefront
(a single DP row) no longer fits in L1?

Score-only mode keeps only O(cols) live state, so we can push N far past the point
where one row fits in cache while total memory stays O(N). If a row exceeds L1 (48 KB
=> N > ~24K on this Core Ultra 9), the untiled kernel streams the previous row from
L2 each row; column tiling with a panel sized to fit L1 keeps that reused slice in L1.
This is the roofline/tiling argument: convert an L2-bound sweep into an L1-resident one.

We can't compute a numpy reference at these sizes (a 131072^2 matrix is ~137 GB), so
correctness is checked by CONSISTENCY: untiled/col/block must all report the same
score (tiling only reorders the traversal).

Usage (repo root, clang18 on PATH):
    uv run python Reptile/roofline_score.py --kind sw --sizes 16384,32768,65536,131072 \
        --tiles 4096,8192,16384,32768 --iters 3
"""

from __future__ import annotations

import argparse

from Reptile import bench


def _inputs(kind: str, n: int) -> dict:
    case = bench._case(kind, n)          # SmithWaterman/NeedlemanWunsch(rows=n, cols=n)
    return case.make_inputs()            # random sequences only; no O(N^2) reference DP


def _time(binary, inputs, approach, tr, tc, reps, warmup, iters):
    r = bench._median_over_reps(binary, inputs, approach, tile_rows=tr, tile_cols=tc,
                                reps=reps, warmup=warmup, iterations=iters, score_only=True)
    return r.seconds, r.score


def main() -> int:
    ap = argparse.ArgumentParser(description="roofline test for score-only tiling")
    ap.add_argument("--kind", default="sw", choices=("nw", "sw"))
    ap.add_argument("--sizes", default="16384,32768,65536,131072")
    ap.add_argument("--tiles", default="4096,8192,16384,32768")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--l1-int16", type=int, default=24576, help="L1d capacity in int16 elems")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    tiles = [int(t) for t in args.tiles.split(",") if t.strip()]

    vr = bench.build_fastest_vecrec(args.kind, bench.SELECT_SIZE, warmup=2, iterations=6)
    if vr is None:
        print(f"[{args.kind}] no schedule built"); return 1
    vr_bin, label = vr
    print(f"\n[{args.kind}] schedule {label}; L1d holds ~{args.l1_int16} int16 "
          f"(one row spills L1 at N>{args.l1_int16})\n")

    for n in sizes:
        inputs = _inputs(args.kind, n)
        row_kb = n * 2 / 1024.0
        fit = "fits L1" if n <= args.l1_int16 else "SPILLS L1"
        base_s, base_score = _time(vr_bin, inputs, "untiled", 0, 0, args.reps, args.warmup, args.iters)
        print(f"  --- {args.kind} {n}x{n}: one row = {row_kb:.0f} KB ({fit}); "
              f"score-only untiled {base_s*1e3:.2f} ms (score {base_score}) ---")
        print(f"    {'config':<16} {'ms':>10} {'vs untiled':>11} {'tile KB':>8}  score")
        print(f"    {'untiled':<16} {base_s*1e3:>10.2f} {'1.00x':>11} {'-':>8}  {base_score}")
        for approach in ("col", "block"):
            for t in tiles:
                if t > n:
                    continue
                tr = t if approach == "block" else 0
                s, sc = _time(vr_bin, inputs, approach, tr, t, args.reps, args.warmup, args.iters)
                tile_kb = t * 2 / 1024.0
                flag = "" if sc == base_score else "  <<SCORE MISMATCH"
                print(f"    {approach+' t='+str(t):<16} {s*1e3:>10.2f} "
                      f"{base_s/s:>10.2f}x {tile_kb:>7.0f}  {sc}{flag}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
