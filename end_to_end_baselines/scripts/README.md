# end_to_end_baselines/scripts/

Build and benchmark scripts for the end-to-end aligner integrations.

## Build scripts

Each build script:
1. Calls the corresponding `generate_*.py` to regenerate the VecRec kernel header
2. Clones the aligner if not already present in `third_party/`
3. Builds the **original** binary (stock kernel)
4. Compiles the VecRec wrapper object
5. Patches the static archive (or monolithic compile for Bowtie2)
6. Builds the **vecrec** binary
7. Copies both to `bin/`

| Script | Binaries produced |
|--------|-------------------|
| `build_minimap2.sh` | `bin/minimap2_original`, `bin/minimap2_vecrec` |
| `build_bwa_mem2.sh` | `bin/bwa_mem2_original`, `bin/bwa_mem2_vecrec` |
| `build_bowtie2.sh` | `bin/bowtie2_original`, `bin/bowtie2_vecrec` |
| `build_and_run_linux.sh` | All six binaries + runs all three benchmarks |

```bash
# Run from repo root:
bash end_to_end_baselines/scripts/build_minimap2.sh [schedule_index]
bash end_to_end_baselines/scripts/build_bwa_mem2.sh [schedule_index]
bash end_to_end_baselines/scripts/build_bowtie2.sh [schedule_index]
```

For minimap2, omitting `schedule_index` runs VecRec's standard schedule and
codegen search separately for the 1-piece and 2-piece int8/int16 variants on
the current host. It benchmarks the kernels on 256×256 matrices with map-ont
gap penalties (100 timed iterations by default), then writes the choices to
`minimap2/generated/vecrec_schedule_search.json`. A positional index or
`VECREC_SCHED_1P` / `VECREC_SCHED_2P` explicitly overrides that search.
`bench exec e2e --rebuild` reruns the search. This is kernel-level tuning;
the e2e timing still measures the completed minimap2 binaries afterward.
The BWA-MEM2 and Bowtie2 scripts still use their fixed defaults.

## Codegen scripts

| Script | Output | Default schedule |
|--------|--------|-----------------|
| `generate_kernel.py` | `minimap2/generated/vecrec_affine_kernel.h` | Searched per variant (1-piece/2-piece, int8/int16) |
| `generate_bwa_kernel.py` | `bwa_mem2/generated/vecrec_bwa_kernel.h` | k8 (epi=32, 1-piece affine, bw=512) |
| `generate_bowtie2_kernel.py` | `bowtie2/generated/vecrec_bt2_{ee,loc}_kernel.h` | k8 (epi=32, EE + local) |

```bash
uv run python end_to_end_baselines/scripts/generate_kernel.py [schedule_index]
uv run python end_to_end_baselines/scripts/generate_bwa_kernel.py [schedule_index]
uv run python end_to_end_baselines/scripts/generate_bowtie2_kernel.py [schedule_index]
```

## Benchmark scripts

Each benchmark script times the full pipeline (index load + seed + extend + SAM)
on a single thread (`-t 1`) and prints a comparison table.

```bash
uv run python end_to_end_baselines/scripts/run_e2e.py        --reps 10 [--reads ...] [--ref ...]
uv run python end_to_end_baselines/scripts/run_bwa_e2e.py   --reps 5  [--reads ...] [--ref ...]
uv run python end_to_end_baselines/scripts/run_bowtie2_e2e.py --reps 5 [--reads ...] [--ref ...]
```

**Defaults**:
- Reads: `data/mt_reads.fq` (200 × 1000bp, MT-human)
- Reference: `end_to_end_baselines/data/MT-human.fa` (for minimap2/BWA) or MT-human Bowtie2 index
- Index (Bowtie2): `data/mt_bt2_index` — built automatically on first run

**Output format:**
```
binary                 median    p10–p90   alignments
------------------------------------------------------
minimap2-original      194.2ms   [193–195ms]       200
minimap2-vecrec        162.7ms   [162–163ms]       200
Speedup: 1.19×
Alignment count: 200 = 200 ✓
```

## Data generation scripts

```bash
# Generate mt_reads.fq (200 × 1000bp reads from MT-human.fa)
uv run python end_to_end_baselines/scripts/generate_test_reads.py

# Generate tandem repeat stress-test data
uv run python end_to_end_baselines/scripts/generate_tandem_reads.py
```

`MT-human.fa` is committed at `end_to_end_baselines/data/MT-human.fa` (a copy of
the reference bundled with the minimap2 tree).
