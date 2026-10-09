// ksw2 driver: SIMD-only. Exposes the vectorized ksw2 kernels, one algo tag per
// kernel, each matched to a VecRec recurrence class in bio_kernel_evals:
//
// algo "gg2_sse":   ksw_gg2_sse()   — SIMD 1-piece affine GLOBAL (corner score) + CIGAR.
// algo "extz2_sse": ksw_extz2_sse() — SIMD 1-piece affine EXTENSION (ez.score corner) + CIGAR.
//                   "approx_max" (bool) sets KSW_EZ_APPROX_MAX -- minimap2's
//                   gap-fill approx-max shortcut, isolated at the kernel level.
// algo "extd2_sse": ksw_extd2_sse() — SIMD 2-piece dual-affine EXTENSION + CIGAR.
//                   gap_open2/gap_extend2 from "gap_open2"/"gap_extend2" fields.
//                   "approx_max" (bool) as above.
// algo "extf2_sse": ksw_extf2_sse() — SIMD LINEAR-gap (match/mismatch, xdrop). Score-only.
// algo "ll_i16":    ksw_ll_i16()    — SIMD LOCAL affine (Farrar striped, int16). Score-only.
// algo "extz2_sse_kalloc_batch" / "extd2_sse_kalloc_batch":
//                   Cross-check for the single-call approx-max A/B above: many
//                   calls of varying size sharing ONE long-lived kalloc pool
//                   (km_init()), mirroring map.c's `b->km` reused across an
//                   entire thread's reads, instead of a fresh km=nullptr
//                   (plain malloc/free) per call. Confirms the exact-vs-approx
//                   gap seen on a single call (algorithmic: exact's O(n)
//                   per-row max-scan vs approx's O(1)) also holds at batch
//                   scale through the real allocator pattern -- see
//                   bio_kernel_evals/cases/README.md. "score" reports total
//                   cells processed (not an alignment score -- there is no
//                   single reference for a batch).
//
// Banding: every kernel that supports it reads "band_width" (>=0 banded, -1 =>
// full matrix via w<0). ksw_ll_i16 is Farrar-striped and ignores banding.
//
// Alphabet: gg2_sse/extf2_sse/ll_i16 use the plain 4-symbol ACGT encoding
// (mat/m below). extz2_sse/extd2_sse use a separate 5-symbol encoding
// (mat5/m5) because those two kernels treat symbol (m-1) as a wildcard/"N"
// base internally -- with a plain 4-symbol alphabet that collides with T
// (encoded as 3), silently mis-scoring it. The 5th symbol is never actually
// produced by encode(), it just keeps the real bases off the wildcard slot.
//
// No scalar ksw2 variants are built or exposed (SIMD-only comparison).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include "ksw2.h"       // SIMD kernel prototypes + ksw_extz_t
#include "kalloc.h"     // km allocator (km == NULL falls back to system malloc)
}

#include "../common/driver_io.hpp"

using baselines::Input;
using baselines::TimedRun;
using baselines::Timer;

static std::vector<uint8_t> encode(const std::string& s) {
    std::vector<uint8_t> out(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        switch (s[i]) {
            case 'A': case 'a': out[i] = 0; break;
            case 'C': case 'c': out[i] = 1; break;
            case 'G': case 'g': out[i] = 2; break;
            case 'T': case 't': out[i] = 3; break;
            default:            out[i] = 0; break;
        }
    }
    return out;
}

int main(int argc, char** argv) {
    return baselines::run_driver(argc, argv, "ksw2_align", [](const char* path) -> int {
        const Input in = baselines::parse_input(path);

        const int8_t m = 4;  // ACGT
        int8_t mat[16];
        for (int a = 0; a < m; ++a)
            for (int b = 0; b < m; ++b)
                mat[a * m + b] = static_cast<int8_t>(a == b ? in.match : in.mismatch);

        // ksw_extz2_sse / ksw_extd2_sse treat symbol (m-1) as a wildcard/"N"
        // base (see ksw2_extz2_sse.c: `m1_ = m-1`, matched via cmpeq against
        // either sequence symbol and scored with `sc_N_` instead of a real
        // match/mismatch lookup). With m=4 and our 0..3 ACGT encoding, symbol
        // 3 (T) collides with that wildcard slot -- every comparison
        // involving a real T would be silently mis-scored. Use a 5-symbol
        // alphabet (dummy unused wildcard at index 4) for those two kernels
        // only, so no real base collides with the sentinel.
        const int8_t m5 = 5;
        int8_t mat5[25];
        for (int a = 0; a < m5; ++a)
            for (int b = 0; b < m5; ++b)
                mat5[a * m5 + b] = static_cast<int8_t>(a == b ? in.match : in.mismatch);

        const std::vector<uint8_t> q = encode(in.x);
        const std::vector<uint8_t> t = encode(in.y);
        const int qlen = static_cast<int>(q.size());
        const int tlen = static_cast<int>(t.size());

        // ksw2 gap cost(L) = gapo + L*gape, matching the canonical scheme.
        const int8_t gapo = static_cast<int8_t>(in.gap_open);
        const int8_t gape = static_cast<int8_t>(in.gap_extend);
        const int w = static_cast<int>(in.band_width);  // -1 => full matrix (w<0)
        const long cells = static_cast<long>(qlen) * static_cast<long>(tlen);

        void* km = nullptr;  // km == NULL => kmalloc/kfree use system malloc/free.
        long score = 0;
        baselines::AlignmentTraceback alignment;
        Timer timer;

        if (in.algo == "gg2_sse") {
            // SIMD 1-piece affine GLOBAL (corner score) + CIGAR.
            TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
                int m_cigar = 0, n_cigar = 0;
                uint32_t* cigar = nullptr;
                score = ksw_gg2_sse(km, qlen, q.data(), tlen, t.data(), m, mat,
                                    gapo, gape, w, &m_cigar, &n_cigar, &cigar);
                alignment = baselines::ksw_cigar_to_traceback(cigar, n_cigar, in.x, in.y);
                kfree(km, cigar);
            });
            baselines::print_result(score, timed, cells, alignment);
        } else if (in.algo == "extz2_sse") {
            // SIMD 1-piece affine EXTENSION; report corner score (ez.score).
            // Uses mat5/m5 (see comment above) so T doesn't collide with the
            // kernel's built-in wildcard symbol.
            //
            // "approx_max" (in.approx_max) sets KSW_EZ_APPROX_MAX -- this is
            // minimap2's gap-fill pass-1 shortcut (see align.c: MM_EXACT_ONLY=0),
            // which skips the exact 32-bit H/max bookkeeping. In gap-fill mode
            // (no KSW_EZ_EXTZ_ONLY, as here) the backtrack always starts from
            // the corner (tlen-1,qlen-1) regardless of this flag, so ez.score
            // is expected to be bit-identical to the flag=0 case -- this
            // isolates the flag's SPEED cost alone, with no correctness delta,
            // matching the real minimap2 drop-in comparison at the DP-kernel
            // level (no pipeline/index/chaining overhead).
            const int flag = in.approx_max ? KSW_EZ_APPROX_MAX : 0;
            TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
                ksw_extz_t ez{};
                ksw_extz2_sse(km, qlen, q.data(), tlen, t.data(), m5, mat5,
                              gapo, gape, w, /*zdrop=*/-1, /*end_bonus=*/0,
                              flag, &ez);
                score = ez.score == KSW_NEG_INF ? 0 : static_cast<long>(ez.score);
                alignment = baselines::ksw_cigar_to_traceback(ez.cigar, ez.n_cigar, in.x, in.y);
                kfree(km, ez.cigar);
            });
            baselines::print_result(score, timed, cells, alignment);
        } else if (in.algo == "extd2_sse") {
            // SIMD 2-piece dual-affine EXTENSION; report corner score (ez.score).
            // Uses mat5/m5 (see comment above) so T doesn't collide with the
            // kernel's built-in wildcard symbol. See "extz2_sse" above for what
            // in.approx_max (KSW_EZ_APPROX_MAX) does and why ez.score is
            // expected to be unaffected by it in this gap-fill call shape.
            const int8_t gapo2 = static_cast<int8_t>(in.gap_open2);
            const int8_t gape2 = static_cast<int8_t>(in.gap_extend2);
            const int flag = in.approx_max ? KSW_EZ_APPROX_MAX : 0;
            TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
                ksw_extz_t ez{};
                ksw_extd2_sse(km, qlen, q.data(), tlen, t.data(), m5, mat5,
                              gapo, gape, gapo2, gape2, w,
                              /*zdrop=*/-1, /*end_bonus=*/0, flag, &ez);
                score = ez.score == KSW_NEG_INF ? 0 : static_cast<long>(ez.score);
                alignment = baselines::ksw_cigar_to_traceback(ez.cigar, ez.n_cigar, in.x, in.y);
                kfree(km, ez.cigar);
            });
            baselines::print_result(score, timed, cells, alignment);
        } else if (in.algo == "extf2_sse") {
            // SIMD LINEAR-gap (match/mismatch), xdrop off. Score-only (no CIGAR).
            // mch = match (>0), mis = mismatch (<=0), e = per-base gap penalty.
            const int8_t mch = static_cast<int8_t>(in.match);
            const int8_t mis = static_cast<int8_t>(in.mismatch);
            const int8_t e   = static_cast<int8_t>(in.gap_extend);
            TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
                ksw_extz_t ez{};
                ksw_extf2_sse(km, qlen, q.data(), tlen, t.data(), mch, mis, e, w,
                              /*xdrop=*/-1, &ez);
                score = ez.score == KSW_NEG_INF ? 0 : static_cast<long>(ez.score);
            });
            baselines::print_result(score, timed, cells);
        } else if (in.algo == "ll_i16") {
            // SIMD LOCAL affine (Farrar striped, int16). Score-only (no CIGAR).
            // The query profile (ksw_ll_qinit) is built INSIDE the timed region so
            // the timed span matches VecRec's (profile build + kernel + score).
            TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
                int qe = -1, te = -1;
                void* qp = ksw_ll_qinit(km, /*size=*/2, qlen, q.data(), m, mat);
                score = static_cast<long>(
                    ksw_ll_i16(qp, tlen, t.data(),
                               static_cast<int>(gapo), static_cast<int>(gape),
                               &qe, &te));
                kfree(km, qp);
            });
            baselines::print_result(score, timed, cells);
        } else if (in.algo == "extz2_sse_kalloc_batch" || in.algo == "extd2_sse_kalloc_batch") {
            // Reproduce minimap2's real allocation pattern instead of the
            // single-call km=nullptr path the other algos above use: many
            // calls of VARYING size sharing ONE long-lived kalloc pool
            // (km_init()), mirroring map.c's `b->km` reused across an entire
            // thread's worth of reads (see kalloc.c: kmalloc/kfree are a
            // linear-search circular free list, not a size-class allocator
            // like glibc -- its cost is sensitive to how fragmented/long that
            // free list has become, which a fresh malloc/free per repeated
            // identical-size call never exercises).
            //
            // Size histogram mirrors the real [extend-size] histogram measured
            // on the sv_core2048 e2e dataset (MM_COUNT_MATRIX; see
            // end_to_end_baselines/minimap2/CLAUDE.md): 259 calls total, mostly
            // small (<256) gap-fills between consecutive exact-flank seeds plus
            // a long tail of big divergent-core gap-fills.
            struct SizeBucket { int lo, hi, count; };
            static const SizeBucket kHist[] = {
                {8, 31, 59}, {32, 63, 3}, {64, 127, 1}, {128, 255, 154},
                {256, 511, 6}, {512, 1023, 5}, {1024, 2047, 10}, {2048, 2300, 21},
            };
            std::mt19937 rng(12345);
            std::vector<std::pair<int, int>> sizes;
            for (const auto& b : kHist) {
                std::uniform_int_distribution<int> d(b.lo, b.hi);
                for (int i = 0; i < b.count; ++i) {
                    int n = d(rng);
                    sizes.emplace_back(n, n);
                }
            }
            std::shuffle(sizes.begin(), sizes.end(), rng); // interleave sizes like real seed order

            std::uniform_int_distribution<int> base(0, 3);
            std::vector<std::vector<uint8_t>> qseqs, tseqs;
            qseqs.reserve(sizes.size());
            tseqs.reserve(sizes.size());
            for (const auto& sz : sizes) {
                std::vector<uint8_t> qs(static_cast<std::size_t>(sz.first));
                std::vector<uint8_t> ts(static_cast<std::size_t>(sz.second));
                for (auto& c : qs) c = static_cast<uint8_t>(base(rng));
                for (auto& c : ts) c = static_cast<uint8_t>(base(rng));
                qseqs.push_back(std::move(qs));
                tseqs.push_back(std::move(ts));
            }

            long total_cells = 0;
            for (const auto& sz : sizes)
                total_cells += static_cast<long>(sz.first) * static_cast<long>(sz.second);

            const bool dual = (in.algo == "extd2_sse_kalloc_batch");
            const int8_t gapo2 = static_cast<int8_t>(in.gap_open2);
            const int8_t gape2 = static_cast<int8_t>(in.gap_extend2);
            const int flag = in.approx_max ? KSW_EZ_APPROX_MAX : 0;
            TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
                void* pool = km_init(); // ONE pool for the whole batch, like b->km
                for (std::size_t i = 0; i < sizes.size(); ++i) {
                    const int ql = sizes[i].first, tl = sizes[i].second;
                    ksw_extz_t ez{};
                    if (dual) {
                        ksw_extd2_sse(pool, ql, qseqs[i].data(), tl, tseqs[i].data(), m5, mat5,
                                      gapo, gape, gapo2, gape2, w,
                                      /*zdrop=*/-1, /*end_bonus=*/0, flag, &ez);
                    } else {
                        ksw_extz2_sse(pool, ql, qseqs[i].data(), tl, tseqs[i].data(), m5, mat5,
                                      gapo, gape, w, /*zdrop=*/-1, /*end_bonus=*/0, flag, &ez);
                    }
                    kfree(pool, ez.cigar);
                }
                km_destroy(pool);
            });
            score = total_cells;
            baselines::print_result(score, timed, total_cells);
        } else {
            std::fprintf(stderr, "ksw2_align: unknown algo %s\n", in.algo.c_str());
            return 2;
        }
        return 0;
    });
}
