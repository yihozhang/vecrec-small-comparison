"""Table 5: Reptile tiling vs VecRec on Smith--Waterman (group ``reptile``).

Runs ``Reptile/bench.py``'s sweep with ``--family both``: the REPTILE-style
scalar tiled loop nests (The REPTILE implementation is borrowed from REPTILE's artifact; untiled /
row / col / block, sweeping tile sizes) and the VecRec kernel, all validated
against the textbook reference score. The paper reports, per matrix size,
scalar untiled, the best scalar tiling, and VecRec untiled.

Results: ``results/reptile/<host>.json`` (the sweep's raw records).
"""

from __future__ import annotations

import json
from pathlib import Path

from ae.layout import REPTILE_KIND, REPTILE_SIZES, REPTILE_TILES
from ae.paths import result_file


def exec_group(host: str, *, sizes=REPTILE_SIZES, tiles=REPTILE_TILES, reps: int = 3,
               warmup: int = 3, iterations: int = 15) -> Path:
    from Reptile import bench

    res = bench.sweep(REPTILE_KIND, list(sizes), list(tiles), reps, warmup, iterations,
                      family="both")
    out = result_file("reptile", host, ".json")
    out.write_text(json.dumps([res], indent=2))
    print(f"wrote {out}")
    return out
