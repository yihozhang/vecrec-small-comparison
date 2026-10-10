"""Repository paths and the per-machine result-file naming convention."""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
BIN = ROOT / "bin"
THIRD_PARTY = ROOT / "third_party"
GENERATED = ROOT / "generated"
BUILD = ROOT / "build"


def host_tag(override: str | None = None) -> str:
    """Short machine label used to name result files (``results/<group>/<host>.csv``).

    Taken from ``--host``, else the ``AE_HOST`` environment variable, else the
    hostname. Only ``[A-Za-z0-9_-]`` survive so the tag is a safe file stem.
    """
    tag = override or os.environ.get("AE_HOST") or platform.node() or "host"
    tag = re.sub(r"[^A-Za-z0-9_-]+", "-", tag).strip("-")
    return tag or "host"


def result_file(group: str, host: str, suffix: str = ".csv") -> Path:
    d = RESULTS / group
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{host}{suffix}"


def result_sources(group: str, suffix: str = ".csv") -> list[tuple[str, Path]]:
    """All ``(label, path)`` result files for a group, label = file stem."""
    d = RESULTS / group
    if not d.is_dir():
        return []
    return sorted((p.stem, p) for p in d.glob(f"*{suffix}"))


def cpu_model() -> str:
    """Best-effort CPU model string (for the run log / README)."""
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Darwin":
        import subprocess

        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def merge_csv(path: Path, rows: list[dict], columns: list[str], key: str) -> None:
    """Write ``rows`` to ``path``, keeping rows of an existing file whose ``key``
    is not among the new ones (so ``--only X`` refreshes X without discarding
    the other benchmarks' results)."""
    import csv

    old: list[dict] = []
    if path.exists():
        with open(path, newline="") as f:
            old = [r for r in csv.DictReader(f) if r.get(key)]
    new_keys = {r[key] for r in rows}
    kept = [r for r in old if r[key] not in new_keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept + rows)
