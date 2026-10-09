// I/O for the profile-HMM Viterbi vecrec driver.
//
// Reads the compact model JSON produced by `hmmer_viterbi hmm_prepare` (the same
// file the numpy reference consumes): scalars M/L/K, the observation sequence, and
// the configured score arrays HMMER reads (tsc, emitM, emitI). The vecrec driver
// materializes its per-cell kernel inputs from these inside its timed region.
//
// Scores are floats (Viterbi log-odds in nats); -inf is stored as the sentinel
// -1e30 in the JSON, mapped back to -infinity here. A float-score result printer
// mirrors driver_io.hpp's print_result (which emits an integer score).
#pragma once

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "driver_io.hpp"  // baselines::detail::{read_file,get_int,get_string}, Timer, TimedRun

namespace baselines {

struct HmmModel {
    int M = 0;            // model length (nodes 1..M)
    int L = 0;            // observation length
    int K = 0;            // alphabet size
    std::vector<int> obs;          // length L, residue indices 0..K-1
    std::vector<float> tsc;        // (M+1)*8, p7P_TSC layout: tsc[k*8+s]
    std::vector<float> emitM;      // K*(M+1), emitM[x*(M+1)+k] = MSC(k,x)
    std::vector<float> emitI;      // K*(M+1), emitI[x*(M+1)+k] = ISC(k,x)
};

namespace detail {

// Parse a JSON array of numbers starting at the value for `key`. Maps the dump's
// -1e30 sentinel back to -infinity for float arrays.
inline std::vector<double> get_num_array(const std::string& s, const std::string& key) {
    std::size_t p = value_start(s, key);
    if (p >= s.size() || s[p] != '[') throw std::runtime_error("expected array for key: " + key);
    ++p;
    std::vector<double> out;
    while (p < s.size() && s[p] != ']') {
        while (p < s.size() && (s[p] == ' ' || s[p] == ',' || s[p] == '\n' || s[p] == '\t')) ++p;
        if (p < s.size() && s[p] == ']') break;
        const std::size_t start = p;
        if (s[p] == '-' || s[p] == '+') ++p;
        while (p < s.size() && (std::isdigit(static_cast<unsigned char>(s[p])) ||
                                s[p] == '.' || s[p] == 'e' || s[p] == 'E' ||
                                s[p] == '-' || s[p] == '+')) ++p;
        out.push_back(std::stod(s.substr(start, p - start)));
    }
    return out;
}

}  // namespace detail

inline HmmModel parse_hmm(const char* path) {
    const std::string s = detail::read_file(path);
    HmmModel m;
    m.M = static_cast<int>(detail::get_int(s, "M"));
    m.L = static_cast<int>(detail::get_int(s, "L"));
    m.K = static_cast<int>(detail::get_int(s, "K"));
    for (double v : detail::get_num_array(s, "obs")) m.obs.push_back(static_cast<int>(v));
    auto to_f = [](const std::vector<double>& v) {
        std::vector<float> out;
        out.reserve(v.size());
        for (double x : v) out.push_back(x <= -1e30 ? -std::numeric_limits<float>::infinity()
                                                     : static_cast<float>(x));
        return out;
    };
    m.tsc = to_f(detail::get_num_array(s, "tsc"));
    m.emitM = to_f(detail::get_num_array(s, "emitM"));
    m.emitI = to_f(detail::get_num_array(s, "emitI"));
    return m;
}

// Like print_result, but the score is a float (Viterbi log-odds in nats).
inline void print_hmm_result(double score, const TimedRun& timed, long cells,
                             const std::string& path) {
    const double seconds = timed.median;
    const double gcups = seconds > 0.0 ? static_cast<double>(cells) / seconds / 1e9 : 0.0;
    std::printf("{\"score\": %.9g, \"seconds\": %.9g, \"gcups\": %.6g, \"cells\": %ld, \"samples\": [",
                score, seconds, gcups, cells);
    for (std::size_t i = 0; i < timed.samples.size(); ++i)
        std::printf("%s%.9g", i ? ", " : "", timed.samples[i]);
    std::printf("], \"alignment\": {\"query\": \"%s\", \"ref\": \"\", \"comp\": \"\"}}\n",
                path.c_str());
}

}  // namespace baselines
