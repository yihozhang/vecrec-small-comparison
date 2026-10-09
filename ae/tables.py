"""Render the paper's tables (LaTeX + Markdown) from ``results/<group>/*``.

Every renderer takes ``sources = [(column-group label, path), ...]`` -- one
result file per machine -- and lays the machines out side by side exactly like
the paper (which has Skylake / Zen 5 / Apple M2). With a single source you get
a single column group.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Optional

from ae import layout

# ── helpers ──────────────────────────────────────────────────────────────────

def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _f1(x: Optional[float]) -> str:
    return "--" if x is None or not math.isfinite(x) else f"{x:.1f}"


def _f2(x: Optional[float]) -> str:
    return "--" if x is None or not math.isfinite(x) else f"{x:.2f}"


def _strip_tex(s: str) -> str:
    """LaTeX cell -> plain Markdown cell."""
    if s == "--":
        return s
    return (s.replace("\\textsuperscript{$\\dagger$}", "†").replace("\\textbf{", "**")
             .replace("\\texttt{", "").replace("$\\times$", "×").replace("\\_", "_")
             .replace("\\%", "%").replace("~", " ").replace("--", "–").replace("}", "**" if "**" in s or "\\textbf" in s else ""))


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_strip_tex(c) for c in r) + " |")
    return "\n".join(out)


# ── Tables 1 & 2 ─────────────────────────────────────────────────────────────

def _load_micro(path: Path) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r["name"]: r for r in csv.DictReader(f) if r.get("name")}


def fig_micro(group: str, sources: list[tuple[str, Path]]) -> tuple[str, str]:
    """Throughput (Melem/s) of Serial / Base / Vec and the speedup of Vec over the
    faster of Serial and Base, per machine; rows and names as in the paper."""
    rows_layout = layout.MICRO_GROUPS[group]
    data = [(label, _load_micro(p)) for label, p in sources]
    n = len(data)
    cols = "l|" + "|".join(["rrrr"] * n)

    tex = [r"\begin{table}[t]", r"\resizebox{\textwidth}{!}{%", r"\centering", r"\small",
           r"\begin{tabular}{" + cols + "}", r"\toprule"]
    tex.append("Benchmark & " + " & ".join(r"\multicolumn{4}{c}{%s}" % latex_escape(l) for l, _ in data) + r" \\")
    tex.append(" ".join(r"\cmidrule(lr){%d-%d}" % (2 + 4 * i, 5 + 4 * i) for i in range(n)))
    tex.append(" & " + " & ".join(["Serial & Base & Vec & Speedup"] * n) + r" \\")
    tex.append(" & " + " & ".join([r"\multicolumn{3}{c}{(Melem/s)} & (\(\times\))"] * n) + r" \\")
    tex.append(r"\midrule")
    md_rows: list[list[str]] = []
    for label, name in rows_layout:
        cells = [label]
        for _, rows in data:
            r = rows.get(name)
            if r is None:
                cells += ["--"] * 4
                continue
            el = float(r["elements"])
            ser, base, opt = float(r["serial_us"]), float(r["baseline_us"]), float(r["opt_us"])
            best = min(ser, base)
            cells += [_f1(el / ser), _f1(el / base), _f1(el / opt), _f2(best / opt)]
        tex.append(" & ".join(cells) + r" \\")
        md_rows.append(cells)
    tex += [r"\bottomrule", r"\end{tabular}", "}"]
    caption = {
        "wild": r"Benchmarked throughputs for recurrence applications on the serial, baseline, "
                r"and vectorized implementations across architectures. Speedup is Vec over the "
                r"faster of Serial and Base.",
        "leetcode": r"Benchmarked throughputs on LeetCode problems.",
    }[group]
    tex += [r"\caption{" + caption + "}",
            r"\label{tab:%s}" % {"wild": "recurrence-throughput", "leetcode": "leetcode-throughput"}[group],
            r"\end{table}"]
    header = ["Benchmark"] + [f"{l} {c}" for l, _ in data for c in ("Serial", "Base", "Vec", "Speedup")]
    return "\n".join(tex), _md_table(header, md_rows)


# ── Table 3 ──────────────────────────────────────────────────────────────────

class _Impl:
    def __init__(self, us: float, elements: float, correct: str) -> None:
        self.us = us
        self.melem = elements / us
        self.correct = correct != "False"


def _load_bio(path: Path, size: int) -> dict[str, dict[str, _Impl]]:
    """{case: {impl: _Impl}} for rows at the requested square size (fallback: largest)."""
    by_case: dict[str, dict[str, _Impl]] = {}
    best_size: dict[str, int] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows, us, el = int(r["rows"]), float(r["median_us"]), float(r["elements"])
            except (KeyError, ValueError):
                continue
            if us <= 0 or el <= 0:
                continue
            case = r["case"]
            # Prefer exactly `size`; otherwise keep the largest size seen.
            cur = best_size.get(case)
            if cur is None or (rows == size) or (cur != size and rows > cur):
                if cur != rows:
                    by_case[case] = {}
                    best_size[case] = rows
            if best_size[case] != rows:
                continue
            by_case[case][r["impl"]] = _Impl(us, el, r.get("correct", ""))
    return by_case


def _best_vecrec(impls: dict[str, _Impl]) -> Optional[_Impl]:
    cands = [v for k, v in impls.items()
             if k.startswith("vecrec") and v.correct and any(w in k for w in layout.BIO_VECREC_WIDTHS)]
    return min(cands, key=lambda v: v.us) if cands else None


def fig_bio(sources: list[tuple[str, Path]], size: int = layout.BIO_SIZE) -> tuple[str, str]:
    data = [(label, _load_bio(p, size)) for label, p in sources]
    n = len(data)
    cols = "ll|" + "|".join(["rr"] * n)
    tex = [r"\begin{table}[t]", r"\centering", r"\resizebox{0.8\textwidth}{!}{%",
           r"\begin{tabular}{" + cols + "}", r"\toprule"]
    tex.append(" &  & " + " & ".join(r"\multicolumn{2}{c}{%s}" % latex_escape(l) for l, _ in data) + r" \\")
    tex.append(" ".join(r"\cmidrule(lr){%d-%d}" % (3 + 2 * i, 4 + 2 * i) for i in range(n)))
    tex.append("Benchmark & Implementation & " + " & ".join([r"Melem/s & \(\times\)"] * n) + r" \\")
    tex.append(r"\midrule")
    md_rows: list[list[str]] = []

    for gi, g in enumerate(layout.BIO_GROUPS):
        if gi:
            tex.append(r"\midrule")
        first = True

        def emit(label: str, cells: list[str]) -> None:
            nonlocal first
            tex.append(" & ".join([g.name if first else "", label, *cells]) + r" \\")
            md_rows.append([g.name if first else "", label, *cells])
            first = False

        # baseline rows (throughput only) + the fastest correct baseline per machine
        best_base: list[Optional[_Impl]] = [None] * n
        for label, case, impl in g.baselines:
            cells = []
            for i, (_, by_case) in enumerate(data):
                v = by_case.get(case, {}).get(impl)
                cells += [_f1(v.melem if v else None), "--"]
                if v is not None and v.correct and (best_base[i] is None or v.us < best_base[i].us):
                    best_base[i] = v
            emit(label, cells)
        # VecRec row: speedup over the fastest correct baseline
        cells = []
        for i, (_, by_case) in enumerate(data):
            v = _best_vecrec(by_case.get(g.case, {}))
            b = best_base[i]
            cells += [_f1(v.melem if v else None), _f2(b.us / v.us if (v and b) else None)]
        emit(r"\textbf{VecRec}", cells)
        # dagger row: WFA2 at 85% identity, reporting WFA2's speedup over VecRec
        if g.wfa85 is not None:
            case, impl = g.wfa85
            cells = []
            for i, (_, by_case) in enumerate(data):
                w = by_case.get(case, {}).get(impl)
                v = _best_vecrec(by_case.get(g.case, {}))
                cells += [_f1(w.melem if w else None), _f2(v.us / w.us if (v and w) else None)]
            emit(r"WFA2 (85\%)\textsuperscript{$\dagger$}", cells)

    tex += [r"\bottomrule", r"\end{tabular}", "}",
            r"\caption{Comparing \vecrec{} with state-of-the-art vectorized libraries on sequence "
            r"alignment algorithms of matrix size $%d \times %d$. The speedup is measured against "
            r"the best-performing baselines. Ksw2 is run on Apple M2 via \texttt{sse2neon}. We run "
            r"parasail with \texttt{int16} outputs, and we run \vecrec{} with both \texttt{int16} "
            r"and \texttt{int32} variants and pick the best. \textsuperscript{$\dagger$}For WFA2, "
            r"runtime is proportional to sequence similarity and the 85\%% similarity reports "
            r"WFA2's speedup over \vecrec{}.}" % (size, size),
            r"\label{tab:biothroughput}", r"\end{table}"]
    header = ["Benchmark", "Implementation"] + [f"{l} {c}" for l, _ in data for c in ("Melem/s", "×")]
    return "\n".join(tex), _md_table(header, md_rows)


# ── Table 4 ──────────────────────────────────────────────────────────────────

def fig_e2e(sources: list[tuple[str, Path]]) -> tuple[str, str]:
    data = []
    for label, p in sources:
        with open(p, newline="") as f:
            data.append((label, {r["dataset"]: r for r in csv.DictReader(f)}))
    n = len(data)
    cols = "l|" + "|".join(["rrr"] * n)
    tex = [r"\begin{table}[t]", r"\centering", r"\resizebox{0.8\textwidth}{!}{%",
           r"\begin{tabular}{" + cols + "}", r"\toprule"]
    if n > 1:
        tex.append("Dataset & " + " & ".join(r"\multicolumn{3}{c}{%s}" % latex_escape(l) for l, _ in data) + r" \\")
        tex.append(" ".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(n)))
        tex.append(" & " + " & ".join([r"Minimap2 (stock ksw2) & Minimap2 (\vecrec{}) & Speedup"] * n) + r" \\")
    else:
        tex.append(r"Dataset & Minimap2 (stock ksw2) & Minimap2 (\vecrec{}) & Speedup \\")
    tex.append(r"\midrule")
    md_rows = []
    for label, _scen in layout.E2E_ROWS:
        cells = [r"\texttt{%s}" % label]
        for _, rows in data:
            r = rows.get(label)
            if r is None:
                cells += ["--"] * 3
                continue
            o, v = float(r["original_ms"]), float(r["vecrec_ms"])
            cells += [f"{o:,.0f} ms", f"{v:,.0f} ms", f"{o / v:.2f}$\\times$"]
        tex.append(" & ".join(cells) + r" \\")
        md_rows.append([label] + [c.replace("$\\times$", "×") for c in cells[1:]])
    tex += [r"\bottomrule", r"\end{tabular}", "}",
            r"\caption{End-to-end alignment time on ONT workloads with approximate maximum, banding, "
            r"and z-drop flags disabled. \texttt{ecoli-ultralong} uses ultra-long \textit{E. coli} "
            r"reads, \texttt{ecoli-strains} uses \textit{E. coli} strain sequences, and "
            r"\texttt{pa-strains} uses \textit{Pseudomonas aeruginosa} strain sequences.}",
            r"\label{tab:ont-vecrec-speedup}", r"\end{table}"]
    header = ["Dataset"] + [f"{l} {c}" for l, _ in data for c in ("stock ksw2", "VecRec", "Speedup")]
    return "\n".join(tex), _md_table(header, md_rows)


# ── Table 5 ──────────────────────────────────────────────────────────────────

def _reptile_records(path: Path) -> dict[int, dict]:
    res = json.loads(path.read_text())
    recs: dict[int, dict] = {}
    for sweep in res:
        if sweep.get("kind") != layout.REPTILE_KIND:
            continue
        for row in sweep["records"]:
            recs[int(row["size"])] = row["configs"]
    return recs


def fig_reptile(sources: list[tuple[str, Path]]) -> tuple[str, str]:
    tex_all, md_all = [], []
    for label, p in sources:
        recs = _reptile_records(p)
        sizes = [s for s in layout.REPTILE_SIZES if s in recs] or sorted(recs)
        untiled, tiled, vec, tile_cfg = [], [], [], []
        for s in sizes:
            cfg = recs[s]
            untiled.append(cfg.get("loops_untiled", {}).get("seconds"))
            loops = {k: v for k, v in cfg.items() if k.startswith("loops_") and k != "loops_untiled" and v.get("ok")}
            if loops:
                bk = min(loops, key=lambda k: loops[k]["seconds"])
                tiled.append(loops[bk]["seconds"])
                tile_cfg.append(bk[len("loops_"):].replace("_t", " "))
            else:
                tiled.append(None)
                tile_cfg.append("--")
            vec.append(cfg.get("vecrec_untiled", {}).get("seconds"))

        def ms(x: Optional[float]) -> str:
            return "--" if x is None else (f"{x * 1e3:.3f}" if x * 1e3 < 10 else f"{x * 1e3:.2f}")

        tex = [r"\begin{wraptable}{r}{0.45\textwidth}", r"\vspace{-1em}", r"\centering", r"\small",
               r"\begin{tabular}{l" + "r" * len(sizes) + "}", r"\toprule",
               " & " + " & ".join(f"${s}$" for s in sizes) + r" \\", r"\midrule",
               "Untiled (ms) & " + " & ".join(ms(x) for x in untiled) + r" \\",
               "Reptile (ms) & " + " & ".join(ms(x) for x in tiled) + r" \\",
               r"\vecrec{} (ms) & " + " & ".join(ms(x) for x in vec) + r" \\", r"\midrule",
               "Speedup & " + " & ".join(
                   (f"${t / v:.2f}\\times$" if (t and v) else "--") for t, v in zip(tiled, vec)) + r" \\",
               r"\bottomrule", r"\end{tabular}",
               r"\caption{Smith--Waterman runtime for $N\times N$ matrices, comparing untiled code, "
               r"Reptile's best tiling, and \vecrec{}. Tile sizes were $%d$--$%d$ with the best being %s. "
               r"(%s)}" % (min(layout.REPTILE_TILES), max(layout.REPTILE_TILES),
                           ", ".join(tile_cfg), latex_escape(label)),
               r"\label{tab:reptile-vs-vecrec-sw}", r"\vspace{-2em}", r"\end{wraptable}"]
        tex_all.append("\n".join(tex))
        header = [label] + [str(s) for s in sizes]
        rows = [["Untiled (ms)"] + [ms(x) for x in untiled],
                ["Reptile (ms)"] + [ms(x) for x in tiled],
                ["VecRec (ms)"] + [ms(x) for x in vec],
                ["Speedup"] + [(f"{t / v:.2f}×" if (t and v) else "--") for t, v in zip(tiled, vec)],
                ["best tiling"] + tile_cfg]
        md_all.append(_md_table(header, rows))
    return "\n\n".join(tex_all), "\n\n".join(md_all)


# ── dispatch ─────────────────────────────────────────────────────────────────

RENDERERS = {
    "wild": lambda src: fig_micro("wild", src),
    "leetcode": lambda src: fig_micro("leetcode", src),
    "bio": fig_bio,
    "e2e": fig_e2e,
    "reptile": fig_reptile,
}
