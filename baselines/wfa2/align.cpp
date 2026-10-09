// WFA2-lib driver: exact end-to-end wavefront alignment.
//
// WFA2 is a *distance* library (minimize cost; match <= 0, penalties > 0).
// Our harness uses *similarity* scores (MATCH=2, MISMATCH=-1, ...). Mapping:
//
//   algo "edit":   distance_metric = edit (Levenshtein). Scoring fields ignored.
//                  cigar->score is the edit distance (report as-is).
//   algo "nw":     distance_metric = gap_linear. Penalties:
//                    match = -in.match, mismatch = -in.mismatch,
//                    indel = in.gap_extend  (linear NW uses gap_open=0).
//   algo "affine": distance_metric = gap_affine. Penalties:
//                    match = -in.match, mismatch = -in.mismatch,
//                    gap_opening = in.gap_open, gap_extension = in.gap_extend.
//
// With match < 0, WFA2's wavefront_compute_classic_score already converts the
// internal distance into a Smith-Waterman/Needleman-Wunsch *similarity* score
// (Eizenga's formula), so cigar->score is directly comparable to nw_reference /
// affine_reference. Heuristics are forced off (wf_heuristic_none). The aligner
// object is created once and reused across warmup/iters.

#include <cstdio>
#include <stdexcept>
#include <string>

extern "C" {
#include "wavefront/wavefront_align.h"
}

#include "../common/driver_io.hpp"

using baselines::AlignmentTraceback;
using baselines::Input;
using baselines::TimedRun;
using baselines::Timer;

static AlignmentTraceback wfa_ops_to_traceback(const cigar_t* cigar,
                                               const std::string& pattern,
                                               const std::string& text) {
    AlignmentTraceback out;
    std::size_t p_pos = 0;
    std::size_t t_pos = 0;
    for (int i = cigar->begin_offset; i < cigar->end_offset; ++i) {
        const char op = cigar->operations[i];
        switch (op) {
            case 'M':
            case 'X':
            case '=':
                out.query.push_back(pattern[p_pos++]);
                out.ref.push_back(text[t_pos++]);
                baselines::append_comp_char(out.comp, out.query.back(), out.ref.back());
                break;
            case 'I':
                // WFA2: I = insertion into pattern = gap in pattern / consume text.
                out.query.push_back('-');
                out.ref.push_back(text[t_pos++]);
                baselines::append_comp_char(out.comp, out.query.back(), out.ref.back());
                break;
            case 'D':
                // WFA2: D = deletion from pattern = gap in text / consume pattern.
                out.query.push_back(pattern[p_pos++]);
                out.ref.push_back('-');
                baselines::append_comp_char(out.comp, out.query.back(), out.ref.back());
                break;
            default:
                break;
        }
    }
    return out;
}

int main(int argc, char** argv) {
    return baselines::run_driver(argc, argv, "wfa2_align", [](const char* path) -> int {
        const Input in = baselines::parse_input(path);

        wavefront_aligner_attr_t attributes = wavefront_aligner_attr_default;
        attributes.alignment_scope = compute_alignment;
        attributes.alignment_form.span = alignment_end2end;
        attributes.heuristic.strategy = wf_heuristic_none;

        if (in.algo == "edit") {
            attributes.distance_metric = edit;
        } else if (in.algo == "nw") {
            attributes.distance_metric = gap_linear;
            attributes.linear_penalties.match = static_cast<int>(-in.match);
            attributes.linear_penalties.mismatch = static_cast<int>(-in.mismatch);
            attributes.linear_penalties.indel = static_cast<int>(in.gap_extend);
        } else if (in.algo == "affine") {
            attributes.distance_metric = gap_affine;
            attributes.affine_penalties.match = static_cast<int>(-in.match);
            attributes.affine_penalties.mismatch = static_cast<int>(-in.mismatch);
            attributes.affine_penalties.gap_opening = static_cast<int>(in.gap_open);
            attributes.affine_penalties.gap_extension = static_cast<int>(in.gap_extend);
        } else {
            throw std::runtime_error(
                "wfa2_align: unknown algo '" + in.algo +
                "' (expected edit|nw|affine)");
        }

        wavefront_aligner_t* const wf_aligner = wavefront_aligner_new(&attributes);
        if (wf_aligner == nullptr) {
            throw std::runtime_error("wfa2_align: wavefront_aligner_new failed");
        }

        long score = 0;
        AlignmentTraceback alignment;
        Timer timer;
        const TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
            const int status = wavefront_align(
                wf_aligner,
                in.x.c_str(), static_cast<int>(in.x.size()),
                in.y.c_str(), static_cast<int>(in.y.size()));
            if (status != WF_STATUS_ALG_COMPLETED) {
                throw std::runtime_error(
                    std::string("wfa2_align: wavefront_align failed: ") +
                    wavefront_align_strerror(status));
            }
            cigar_t* const cigar = wf_aligner->cigar;
            score = cigar->score;
            alignment = wfa_ops_to_traceback(cigar, in.x, in.y);
        });

        wavefront_aligner_delete(wf_aligner);
        baselines::print_result(score, timed, in.cells(), alignment);
        return 0;
    });
}
