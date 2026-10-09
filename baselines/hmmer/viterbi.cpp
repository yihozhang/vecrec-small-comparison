// Profile-HMM Viterbi driver backed by the vendored HMMER 3.4 (third_party/hmmer-3.4).
//
// This is the "library" side of the profile-hmm-viterbi bio-eval. It is the single
// source of the model + scores so vecrec/numpy never have to re-derive HMMER's
// ProfileConfig transforms (occupancy entry, wing retraction, null model):
//
//   algo="hmm_prepare":  sample a seeded model, configure UNIGLOCAL, ZERO the
//       special-state ribbon (so the Viterbi optimum is exactly the core M/I/D
//       glocal path: sequence fully consumed, model fully traversed), and dump the
//       exact scores HMMER reads via its TSC/MSC/ISC macros, plus the (seeded)
//       observation sequence. numpy + the vecrec driver consume this JSON.
//   algo="hmm_run"  :  rebuild the same seeded model and self-time
//       gmx alloc + p7_GViterbi + p7_GTrace (generic exact, scalar).
//
// HMMER's exact glocal Viterbi available in this build is the generic (scalar) one:
// the int16 ViterbiFilter is local-only + saturating (an MSV-style pre-filter, not an
// exact aligner), and p7_ViterbiScore (the old float-SIMD full Viterbi) has a rotted
// NEON port in 3.4 that no longer compiles, so it is excluded from libhmmer.a.
//
// Score is a Viterbi log-odds value in nats (float); it is printed as a float (the
// alignment cases print an integer score, hence a dedicated print here).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "driver_io.hpp"  // baselines::detail::{read_file,get_int,get_string}, Timer

extern "C" {
#include <p7_config.h>

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_random.h"
#include "esl_randomseq.h"

#include "hmmer.h"
}

using baselines::TimedRun;
using baselines::Timer;

namespace {

struct Model {
    ESL_ALPHABET* abc = nullptr;
    P7_HMM* hmm = nullptr;
    P7_BG* bg = nullptr;
    P7_PROFILE* gm = nullptr;
    ESL_DSQ* dsq = nullptr;
    int M = 0;
    int L = 0;
};

// Zero the special-state ribbon so p7_GViterbi's optimum == the core M/I/D glocal
// path. Forbid N/C flanking (LOOP=-inf) so the whole sequence is consumed; free
// E->C->T moves (MOVE=0); J unused (unihit). With glocal `esc=-inf`, the exit is
// only from node M, so the model is fully traversed. Set AFTER p7_ProfileConfig
// (which itself sets these via the length model) so our values win.
void zero_ribbon(P7_PROFILE* gm) {
    gm->xsc[p7P_N][p7P_LOOP] = -eslINFINITY;
    gm->xsc[p7P_N][p7P_MOVE] = 0.0f;
    gm->xsc[p7P_C][p7P_LOOP] = -eslINFINITY;
    gm->xsc[p7P_C][p7P_MOVE] = 0.0f;
    gm->xsc[p7P_J][p7P_LOOP] = -eslINFINITY;
    gm->xsc[p7P_J][p7P_MOVE] = -eslINFINITY;
    gm->xsc[p7P_E][p7P_MOVE] = 0.0f;
    gm->xsc[p7P_E][p7P_LOOP] = -eslINFINITY;
}

// Deterministic: same (seed, M, L) -> identical model AND observation in every mode,
// because prepare/run issue the same RNG calls in the same order.
Model build_model(int seed, int M, int L) {
    Model m;
    m.M = M;
    m.L = L;
    ESL_RANDOMNESS* r = esl_randomness_Create(seed);
    m.abc = esl_alphabet_Create(eslAMINO);
    if (p7_hmm_Sample(r, M, m.abc, &m.hmm) != eslOK) p7_Die("p7_hmm_Sample failed");
    m.bg = p7_bg_Create(m.abc);
    m.gm = p7_profile_Create(m.hmm->M, m.abc);
    p7_bg_SetLength(m.bg, L);
    if (p7_ProfileConfig(m.hmm, m.bg, m.gm, L, p7_UNIGLOCAL) != eslOK)
        p7_Die("p7_ProfileConfig failed");
    zero_ribbon(m.gm);
    m.dsq = static_cast<ESL_DSQ*>(malloc(sizeof(ESL_DSQ) * (static_cast<size_t>(L) + 2)));
    if (esl_rsq_xfIID(r, m.bg->f, m.abc->K, L, m.dsq) != eslOK) p7_Die("esl_rsq_xfIID failed");
    esl_randomness_Destroy(r);
    return m;
}

void free_model(Model& m) {
    if (m.dsq) free(m.dsq);
    if (m.gm) p7_profile_Destroy(m.gm);
    if (m.bg) p7_bg_Destroy(m.bg);
    if (m.hmm) p7_hmm_Destroy(m.hmm);
    if (m.abc) esl_alphabet_Destroy(m.abc);
}

// HMMER stores impossible transitions/emissions as -inf. Emit a large finite
// sentinel instead so the JSON is valid and the C++ vecrec reader needs no inf
// parsing; a sentinel never lies on a valid path (max() discards it), so the
// optimal score is unaffected.
double dump_num(double v) { return (std::isfinite(v) && v > -1e30) ? v : -1e30; }

// Emit the configured scores HMMER actually reads (p7P_TSC/MSC/ISC), plus the
// observation, as a flat JSON the Python side and the vecrec driver both parse.
int do_prepare(const Model& m) {
    const P7_PROFILE* gm = m.gm;
    const int M = m.M, L = m.L, K = m.abc->K;

    std::printf("{\"M\": %d, \"L\": %d, \"K\": %d,\n", M, L, K);

    // obs: residue indices 1..L (dsq[0]/dsq[L+1] are sentinels).
    std::printf("\"obs\": [");
    for (int i = 1; i <= L; ++i) std::printf("%s%d", i > 1 ? ", " : "", (int)m.dsq[i]);
    std::printf("],\n");

    // tsc: flat (M+1)*8, tsc[k*8+s] for k=0..M, s in {MM,IM,DM,BM,MD,DD,MI,II}.
    std::printf("\"tsc\": [");
    for (int k = 0; k <= M; ++k)
        for (int s = 0; s < p7P_NTRANS; ++s)
            std::printf("%s%.9g", (k || s) ? ", " : "", dump_num((double)p7P_TSC(gm, k, s)));
    std::printf("],\n");

    // emitM / emitI: flat K*(M+1), emit[x*(M+1)+k] for x=0..K-1, k=0..M (k=0 unused).
    std::printf("\"emitM\": [");
    for (int x = 0; x < K; ++x)
        for (int k = 0; k <= M; ++k)
            std::printf("%s%.9g", (x || k) ? ", " : "", dump_num((double)p7P_MSC(gm, k, x)));
    std::printf("],\n");
    std::printf("\"emitI\": [");
    for (int x = 0; x < K; ++x)
        for (int k = 0; k <= M; ++k)
            std::printf("%s%.9g", (x || k) ? ", " : "", dump_num((double)p7P_ISC(gm, k, x)));
    std::printf("]}\n");
    return 0;
}

// Build an M/I/D state-path string from a Viterbi trace (core states only).
std::string trace_path(const P7_TRACE* tr) {
    std::string s;
    for (int z = 0; z < tr->N; ++z) {
        switch (tr->st[z]) {
            case p7T_M: s.push_back('M'); break;
            case p7T_I: s.push_back('I'); break;
            case p7T_D: s.push_back('D'); break;
            default: break;  // skip S/N/B/E/C/T/J
        }
    }
    return s;
}

void print_float_result(double score, const TimedRun& timed, long cells,
                        const std::string& path) {
    const double seconds = timed.median;
    const double gcups = seconds > 0.0 ? (double)cells / seconds / 1e9 : 0.0;
    std::printf("{\"score\": %.9g, \"seconds\": %.9g, \"gcups\": %.6g, \"cells\": %ld, \"samples\": [",
                score, seconds, gcups, cells);
    for (size_t i = 0; i < timed.samples.size(); ++i)
        std::printf("%s%.9g", i ? ", " : "", timed.samples[i]);
    std::printf("], \"alignment\": {\"query\": \"%s\", \"ref\": \"\", \"comp\": \"\"}}\n",
                path.c_str());
}

int do_run_generic(const Model& m, int warmup, int iters) {
    float score = 0.0f;
    std::string path;
    Timer timer;
    const TimedRun timed = timer.timed_run(warmup, iters, [&] {
        P7_GMX* gx = p7_gmx_Create(m.gm->M, m.L);
        p7_GViterbi(m.dsq, m.L, m.gm, gx, &score);
        P7_TRACE* tr = p7_trace_Create();
        p7_GTrace(m.dsq, m.L, m.gm, gx, tr);
        path = trace_path(tr);
        p7_trace_Destroy(tr);
        p7_gmx_Destroy(gx);
    });
    print_float_result((double)score, timed, (long)m.L * (long)m.M, path);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s input.json\n", argv[0]);
        return 2;
    }
    try {
        const std::string s = baselines::detail::read_file(argv[1]);
        const std::string algo = baselines::detail::get_string(s, "algo");
        const int seed = (int)baselines::detail::get_int(s, "seed");
        const int M = (int)baselines::detail::get_int(s, "M");
        const int L = (int)baselines::detail::get_int(s, "L");
        const int warmup = (int)baselines::detail::get_int_or(s, "warmup", 3);
        const int iters = (int)baselines::detail::get_int_or(s, "iterations", 20);

        Model m = build_model(seed, M, L);
        int rc;
        if (algo == "hmm_prepare") rc = do_prepare(m);
        else if (algo == "hmm_run") rc = do_run_generic(m, warmup, iters);
        else { std::fprintf(stderr, "unknown algo: %s\n", algo.c_str()); rc = 2; }
        free_model(m);
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hmmer_viterbi error: %s\n", e.what());
        return 1;
    }
}
