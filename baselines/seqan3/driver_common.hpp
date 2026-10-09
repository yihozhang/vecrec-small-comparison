// Shared SeqAn3 driver logic for the SW (method_local) and NW (method_global)
// binaries. Both share the scoring scheme, affine gap config, timing, and JSON
// output; only the alignment method differs.

#pragma once

#include <seqan3/alignment/configuration/align_config_gap_cost_affine.hpp>
#include <seqan3/alignment/configuration/align_config_output.hpp>
#include <seqan3/alignment/configuration/align_config_scoring_scheme.hpp>
#include <seqan3/alignment/pairwise/align_pairwise.hpp>
#include <seqan3/alignment/scoring/nucleotide_scoring_scheme.hpp>
#include <seqan3/alphabet/concept.hpp>
#include <seqan3/alphabet/nucleotide/dna4.hpp>
#include <seqan3/utility/concept.hpp>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../common/driver_io.hpp"

namespace seqan3_driver {

inline std::vector<seqan3::dna4> to_dna4(const std::string& s) {
    std::vector<seqan3::dna4> out;
    out.reserve(s.size());
    for (char c : s) {
        seqan3::dna4 base;
        base.assign_char(c);
        out.push_back(base);
    }
    return out;
}

template <class GappedSeq>
std::string gapped_to_string(const GappedSeq& seq) {
    std::string out;
    out.reserve(seq.size());
    for (auto const& sym : seq) {
        out.push_back(seqan3::to_char(sym));
    }
    return out;
}

template <class AlignmentT>
void fill_traceback_from_alignment(const AlignmentT& gapped, baselines::AlignmentTraceback& out) {
    if constexpr (seqan3::tuple_like<AlignmentT>) {
        out.query = gapped_to_string(std::get<0>(gapped));
        out.ref = gapped_to_string(std::get<1>(gapped));
    } else {
        out.query = gapped_to_string(gapped[0]);
        out.ref = gapped_to_string(gapped[1]);
    }
    out.comp.clear();
    out.comp.reserve(out.query.size());
    for (std::size_t i = 0; i < out.query.size(); ++i) {
        baselines::append_comp_char(out.comp, out.query[i], out.ref[i]);
    }
}

template <class Method>
std::pair<long, baselines::AlignmentTraceback> align_with_traceback(
    const std::vector<seqan3::dna4>& s1,
    const std::vector<seqan3::dna4>& s2,
    const baselines::Input& in,
    Method method) {
    auto cfg = method
        | seqan3::align_cfg::scoring_scheme{seqan3::nucleotide_scoring_scheme{
              seqan3::match_score{static_cast<int8_t>(in.match)},
              seqan3::mismatch_score{static_cast<int8_t>(in.mismatch)}}}
        | seqan3::align_cfg::gap_cost_affine{
              seqan3::align_cfg::open_score{static_cast<int32_t>(-in.gap_open)},
              seqan3::align_cfg::extension_score{static_cast<int32_t>(-in.gap_extend)}};

    baselines::AlignmentTraceback out;
    long score = 0;
    const std::pair<std::vector<seqan3::dna4>, std::vector<seqan3::dna4>> pair{s1, s2};
    auto results = seqan3::align_pairwise(pair, cfg);
    for (auto const& res : results) {
        score = res.score();
        fill_traceback_from_alignment(res.alignment(), out);
    }
    return {score, out};
}

// `method` is either seqan3::align_cfg::method_local{} or ...::method_global{}.
// Gap mapping: the harness passes positive penalty magnitudes; SeqAn3 wants
// negative scores (open_score is the cost of opening, extension_score per gap
// character).
template <class Method>
int run(const char* path, Method method) {
    const baselines::Input in = baselines::parse_input(path);
    const std::vector<seqan3::dna4> s1 = to_dna4(in.x);
    const std::vector<seqan3::dna4> s2 = to_dna4(in.y);

    long score = 0;
    baselines::AlignmentTraceback alignment;
    baselines::Timer timer;
    const baselines::TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
        std::tie(score, alignment) = align_with_traceback(s1, s2, in, method);
    });

    baselines::print_result(score, timed, in.cells(), alignment);
    return 0;
}

}  // namespace seqan3_driver
