"""Table 4: minimap2 end-to-end with VecRec as a drop-in for ksw2 (group ``e2e``).

For each ONT dataset the paper lists we (1) download the real reads/reference
(NCBI + GitHub, into ``end_to_end_baselines/data/realbench/``), (2) build two
minimap2 binaries from the vendored, lightly instrumented minimap2 copy
(``bin/minimap2_original`` = stock ksw2, ``bin/minimap2_vecrec`` = VecRec
extension kernel) and (3) time the full single-threaded pipeline under the
paper's "no approximation" regime (banding, approximate max and z-drop
disabled; scenario definitions in ``end_to_end_baselines/minimap2/scenarios.py``).

Results: ``results/e2e/<host>.csv`` (one row per dataset).
"""

from __future__ import annotations

import csv
import importlib.util
import statistics
import subprocess
import sys
import time
from pathlib import Path

from ae.layout import E2E_ROWS
from ae.paths import ROOT, merge_csv, result_file

MM2_DIR = ROOT / "end_to_end_baselines" / "minimap2"
SCRIPTS = ROOT / "end_to_end_baselines" / "scripts"

# dataset name (datasets.py) -> fetch script that materializes it
FETCH_SCRIPTS = {
    "ecoli-ultralong-real": "fetch_ecoli_ultralong.py",
    "ecoli-ultralong-full": "fetch_ecoli_ultralong_full.py",
    "ecoli-strains-real": "fetch_ecoli_strains.py",
    "pa-strains-real": "fetch_pa_strains.py",
    "salm-strains-real": "fetch_salm_strains.py",
    "ecoli-hifi-real": "fetch_ecoli_hifi.py",
}

COLUMNS = ["dataset", "scenario", "dataset_name", "preset", "reps",
           "original_ms", "vecrec_ms", "speedup", "original_alns", "vecrec_alns"]


def _load(name: str):
    """Import a module from end_to_end_baselines/minimap2 (they are plain scripts)."""
    if str(MM2_DIR) not in sys.path:
        sys.path.insert(0, str(MM2_DIR))
    spec = importlib.util.spec_from_file_location(f"mm2_{name}", MM2_DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def ensure_dataset(ds) -> None:
    if ds.present():
        return
    script = FETCH_SCRIPTS.get(ds.name)
    if script is None:
        raise SystemExit(f"dataset {ds.name} is missing and has no fetch script; see: {ds.make}")
    print(f"[e2e] fetching dataset {ds.name} via {script}", flush=True)
    subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=ROOT, check=True)
    if not ds.present():
        raise SystemExit(f"dataset {ds.name} still missing after {script}")


def time_scenario(mm2run, sc, ds, reps: int) -> dict:
    flags = mm2run._mm2_flags(sc)
    print(f"[e2e] {sc.name}: minimap2 {' '.join(flags)} {ds.ref.name} {ds.reads.name}  (reps={reps})", flush=True)
    rec: dict = {}
    for label, binary in (("original", mm2run.ORIGINAL), ("vecrec", mm2run.VECREC)):
        times, alns = [], 0
        cmd = [str(binary), *flags, str(ds.ref), str(ds.reads)]
        for _ in range(reps):
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True)
            dt = (time.perf_counter() - t0) * 1e3
            if r.returncode != 0:
                sys.stderr.write(r.stderr[-1500:])
                raise SystemExit(f"[e2e] minimap2_{label} exited {r.returncode}")
            times.append(dt)
            alns = mm2run._count_alns(r.stdout)
        rec[f"{label}_ms"] = statistics.median(times)
        rec[f"{label}_alns"] = alns
        print(f"      minimap2_{label:9s} median {rec[f'{label}_ms']:9.1f} ms  alignments {alns}", flush=True)
    rec["speedup"] = rec["original_ms"] / rec["vecrec_ms"] if rec["vecrec_ms"] else float("nan")
    print(f"      speedup (original/vecrec): {rec['speedup']:.2f}x", flush=True)
    return rec


def exec_group(host: str, *, reps: int = 5, rebuild: bool = False,
               only: list[str] | None = None) -> Path:
    datasets = _load("datasets")
    scenarios = _load("scenarios")
    mm2run = _load("run")

    out = result_file("e2e", host)
    rows: list[dict] = []
    for label, scen_name in E2E_ROWS:
        if only and label not in only:
            continue
        sc = scenarios.SCENARIOS[scen_name]
        ds = datasets.get(sc.dataset)
        ensure_dataset(ds)
        mm2run.ensure_build(sc.defines, rebuild)
        rebuild = False
        rec = time_scenario(mm2run, sc, ds, reps)
        rows.append({"dataset": label, "scenario": sc.name, "dataset_name": ds.name,
                     "preset": sc.preset, "reps": reps, **rec})
        merge_csv(out, rows, COLUMNS, key="dataset")
    print(f"wrote {out}")
    return out


def setup_all() -> None:
    """Fetch every dataset Table 4 needs and build both minimap2 binaries."""
    datasets = _load("datasets")
    scenarios = _load("scenarios")
    mm2run = _load("run")
    for _label, scen_name in E2E_ROWS:
        sc = scenarios.SCENARIOS[scen_name]
        ensure_dataset(datasets.get(sc.dataset))
    mm2run.ensure_build(scenarios.SCENARIOS[E2E_ROWS[0][1]].defines, rebuild=False)
