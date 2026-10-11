"""``bench`` / ``fig`` command-line entry points.

    uv run bench list
    uv run bench setup bio e2e          # clone + build third-party deps, fetch data
    uv run bench exec leetcode          # run the experiments behind one table
    uv run bench exec all               # everything (hours)
    uv run fig leetcode                 # render the table from results/leetcode/*.csv
    uv run fig bio --source "Zen 5"=results/bio/zen5.csv --source M2=results/bio/m2.csv

Groups:  wild (Table 1), leetcode (Table 2), bio (Table 3), e2e (Table 4),
         reptile (Table 5), all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ae import paths, tables
from ae.paths import host_tag

GROUPS = ["wild", "leetcode", "bio", "e2e", "reptile"]
TABLE_OF = {"wild": "Table 1", "leetcode": "Table 2", "bio": "Table 3", "e2e": "Table 4",
            "reptile": "Table 5"}


def _expand(groups: list[str]) -> list[str]:
    out: list[str] = []
    for g in groups:
        if g == "all":
            out += [x for x in GROUPS if x not in out]
        elif g in GROUPS:
            if g not in out:
                out.append(g)
        else:
            raise SystemExit(f"unknown group {g!r}; choose from {', '.join(GROUPS + ['all'])}")
    return out


# ── bench ────────────────────────────────────────────────────────────────────

def _rm(path: Path) -> None:
    import shutil

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        print(f"  removed {path.relative_to(paths.ROOT)}/")
    elif path.exists() or path.is_symlink():
        path.unlink()
        print(f"  removed {path.relative_to(paths.ROOT)}")


def _clean(group: str, deep: bool = False) -> None:
    """Delete what `bench exec <group>` rebuilds on demand.

    wild/leetcode: generated/ kernel headers + build/ binaries (forces a new
    schedule search).  bio: bin/ drivers, the CMake tree, cached VecRec driver
    binaries.  e2e: the two minimap2 binaries, the generated drop-in kernel
    and minimap2's object files.  reptile: Reptile/build/.  --deep also removes
    third_party/ and the downloaded datasets.
    """
    root = paths.ROOT
    print(f"[clean] {group}")
    if group in ("wild", "leetcode"):
        from ae.layout import MICRO_GROUPS
        from specs.registry import SPECS

        for _, name in MICRO_GROUPS[group]:
            k = SPECS[name].kernel_name
            for p in (paths.GENERATED / f"{k}.h", paths.GENERATED / f"{k}.meta.json", paths.BUILD / k):
                _rm(p)
    elif group == "bio":
        from ae.groups.bio import REQUIRED_BINARIES

        for b in REQUIRED_BINARIES:
            _rm(paths.BIN / b)
        for p in list(paths.BIN.glob("vecrec_*")) + list(paths.BIN.glob("reptile_*")):
            _rm(p)
        _rm(root / "build" / "baselines")
        if deep:
            _rm(paths.THIRD_PARTY)
    elif group == "e2e":
        import subprocess

        for p in (paths.BIN / "minimap2_original", paths.BIN / "minimap2_vecrec", paths.BIN / ".mm2_build_defines"):
            _rm(p)
        _rm(root / "end_to_end_baselines" / "minimap2" / "generated")
        mm2 = root / "end_to_end_baselines" / "minimap2" / "minimap2"
        subprocess.run(["make", "clean"], cwd=mm2, capture_output=True)
        print(f"  make clean in {mm2.relative_to(root)}/")
        if deep:
            _rm(root / "end_to_end_baselines" / "data" / "realbench")
    elif group == "reptile":
        _rm(root / "Reptile" / "build")
        for p in paths.BIN.glob("reptile_*"):
            _rm(p)

def bench_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list groups and the result files present")

    sp = sub.add_parser("setup", help="install third-party baselines / datasets for groups")
    sp.add_argument("groups", nargs="+")

    cl = sub.add_parser("clean", help="delete build products for groups (keeps third_party/ and datasets)")
    cl.add_argument("groups", nargs="+")
    cl.add_argument("--deep", action="store_true",
                    help="also delete third_party/ clones and downloaded datasets (re-fetched on next setup)")

    ex = sub.add_parser("exec", help="run the experiments for groups")
    ex.add_argument("groups", nargs="+")
    ex.add_argument("--host", help="machine label for results/<group>/<host>.* (default: $AE_HOST or hostname)")
    ex.add_argument("--only", action="append", default=[],
                    help="restrict to one benchmark/case/dataset (repeatable)")
    ex.add_argument("--regen", action="store_true", help="wild/leetcode: redo the schedule search")
    ex.add_argument("--iters", type=int, help="wild/leetcode: timed iterations per kernel")
    ex.add_argument("--warmup", type=int, help="wild/leetcode: warmup iterations per kernel")
    ex.add_argument("--size", type=int, help="bio: square matrix size (default 2048)")
    ex.add_argument("--reps", type=int, help="e2e/reptile: repetitions (default 5 / 3)")
    ex.add_argument("--rebuild", action="store_true", help="e2e: rebuild the minimap2 binaries")
    ex.add_argument("--quick", action="store_true",
                    help="smoke-test settings (small sizes / few iterations); results are NOT paper-grade")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        print(f"host tag: {host_tag()}  ({paths.cpu_model()})")
        for g in GROUPS:
            srcs = paths.result_sources(g, ".json" if g == "reptile" else ".csv")
            print(f"  {g:9s} {TABLE_OF[g]:8s} results: {', '.join(l for l, _ in srcs) or '(none)'}")
        return 0

    if args.cmd == "setup":
        paths.check_vecrec_pin()
        for g in _expand(args.groups):
            if g == "bio":
                from ae.groups import bio
                bio.setup()
            elif g == "e2e":
                from ae.groups import e2e
                e2e.setup_all()
            elif g == "reptile":
                from ae.groups import bio
                bio.setup()  # Reptile drivers reuse baselines/common + the bio cases
            else:
                print(f"[{g}] nothing to set up (only needs clang++ and the vendored vec-rec)")
        return 0

    paths.check_vecrec_pin()
    if args.cmd == "clean":
        for g in _expand(args.groups):
            _clean(g, deep=args.deep)
        return 0

    host = host_tag(args.host)
    for g in _expand(args.groups):
        print(f"\n===== bench exec {g} ({TABLE_OF[g]}) on host '{host}' =====\n", flush=True)
        if g in ("wild", "leetcode"):
            from ae.groups import micro
            micro.exec_group(g, host, regen=args.regen, only=args.only or None,
                             iters=args.iters if args.iters is not None else (5 if args.quick else None),
                             warmup=args.warmup if args.warmup is not None else (1 if args.quick else None))
        elif g == "bio":
            from ae.groups import bio
            bio.exec_group(host, size=args.size or (256 if args.quick else 2048), only=args.only or None)
        elif g == "e2e":
            from ae.groups import e2e
            e2e.exec_group(host, reps=args.reps or (1 if args.quick else 5), rebuild=args.rebuild,
                           only=args.only or None)
        elif g == "reptile":
            from ae.groups import reptile
            if args.quick:
                reptile.exec_group(host, sizes=(512, 1024), tiles=(16, 64), reps=1, warmup=1, iterations=3)
            else:
                reptile.exec_group(host, reps=args.reps or 3)
    return 0


# ── fig ──────────────────────────────────────────────────────────────────────

def fig_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fig", description="Render paper tables from results/.")
    ap.add_argument("groups", nargs="+", help=", ".join(GROUPS + ["all"]))
    ap.add_argument("--source", action="append", default=[], metavar="LABEL=FILE",
                    help="result file to use as one column group (repeatable; default: every file in results/<group>/)")
    ap.add_argument("--size", type=int, default=2048, help="bio: matrix size to tabulate")
    ap.add_argument("--vecrec-widths", default="i8,i16,i32", metavar="W,W",
                    help="bio: element widths the VecRec row may use (default i8,i16,i32 = how the "
                         "paper's numbers were produced; i16,i32 = the caption's rule)")
    ap.add_argument("--out", type=Path, default=paths.FIGURES, help="output directory (default figures/)")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for g in _expand(args.groups):
        suffix = ".json" if g == "reptile" else ".csv"
        if args.source:
            sources = []
            for s in args.source:
                label, _, file = s.partition("=")
                if not file:
                    raise SystemExit(f"--source expects LABEL=FILE, got {s!r}")
                if not file.endswith(suffix):
                    raise SystemExit(f"[{g}] --source {file!r} must be a {suffix} file; "
                                     f"pass --source only with groups of one kind")
                sources.append((label, Path(file)))
        else:
            sources = paths.result_sources(g, suffix)
        if not sources:
            print(f"[{g}] no results under {paths.RESULTS / g}; run `uv run bench exec {g}` first",
                  file=sys.stderr)
            continue
        if g == "bio":
            widths = tuple(f"_{w.strip()}_" for w in args.vecrec_widths.split(",") if w.strip())
            tex, md = tables.fig_bio(sources, size=args.size, widths=widths)
        else:
            tex, md = tables.RENDERERS[g](sources)
        (args.out / f"{g}.tex").write_text(tex + "\n")
        (args.out / f"{g}.md").write_text(md + "\n")
        print(f"## {TABLE_OF[g]} ({g}) -- sources: {', '.join(l for l, _ in sources)}\n")
        print(md)
        print(f"\nwrote {args.out / (g + '.tex')} and {args.out / (g + '.md')}\n")
    return 0
