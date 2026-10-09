// Shared I/O + timing contract for all native baseline drivers.
//
// Each driver:
//   1. reads an input.json path from argv[1]  -> parse_input(...)
//   2. self-times only the alignment call     -> Timer::median_seconds(...)
//   3. prints one JSON line to stdout          -> print_result(...)
//
// The Python side (evals_common/cpp.py) writes the input file and parses the
// single result line, so the JSON shapes here and there must stay in sync.
//
// The input is small and machine-generated, so a tiny hand-rolled JSON reader
// (flat object, string/number values) is sufficient and keeps drivers
// dependency-free.

#pragma once

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "traceback.hpp"

namespace baselines {

struct Input {
    std::string algo;       // "sw" | "nw" | "sg" | "edit" | "sw_banded" | ...
    std::string x;          // first sequence  (rows)
    std::string y;          // second sequence (cols)
    long match = 0;
    long mismatch = 0;
    long gap_open = 0;      // positive penalty magnitude (piece 1)
    long gap_extend = 0;    // positive penalty magnitude (piece 1)
    long gap_open2 = 0;     // piece 2 gap open   (0 = same as gap_open)
    long gap_extend2 = 0;   // piece 2 gap extend (0 = same as gap_extend)
    int warmup = 3;
    int iterations = 20;
    long band_width = -1;   // -1 = no band; >= 0 = diagonal half-width
    bool approx_max = false; // ksw2 KSW_EZ_APPROX_MAX: skip exact max/H bookkeeping

    long cells() const {
        return static_cast<long>(x.size()) * static_cast<long>(y.size());
    }
};

namespace detail {

inline std::string read_file(const char* path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("cannot open input file: ") + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline std::size_t value_start(const std::string& s, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    std::size_t p = s.find(pat);
    if (p == std::string::npos) throw std::runtime_error("missing key: " + key);
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) throw std::runtime_error("malformed value for key: " + key);
    ++p;
    while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
    return p;
}

inline std::string get_string(const std::string& s, const std::string& key) {
    std::size_t p = value_start(s, key);
    if (p >= s.size() || s[p] != '"')
        throw std::runtime_error("expected string for key: " + key);
    ++p;
    std::string out;
    while (p < s.size() && s[p] != '"') {
        if (s[p] == '\\' && p + 1 < s.size()) ++p;  // minimal escape handling
        out.push_back(s[p]);
        ++p;
    }
    return out;
}

inline long get_int(const std::string& s, const std::string& key) {
    std::size_t p = value_start(s, key);
    const std::size_t start = p;
    if (p < s.size() && (s[p] == '-' || s[p] == '+')) ++p;
    while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) ++p;
    if (p == start) throw std::runtime_error("expected integer for key: " + key);
    return std::stol(s.substr(start, p - start));
}

inline long get_int_or(const std::string& s, const std::string& key, long fallback) {
    if (s.find("\"" + key + "\"") == std::string::npos) return fallback;
    return get_int(s, key);
}

}  // namespace detail

inline Input parse_input(const char* path) {
    const std::string s = detail::read_file(path);
    Input in;
    in.algo = detail::get_string(s, "algo");
    in.x = detail::get_string(s, "x");
    in.y = detail::get_string(s, "y");
    in.match = detail::get_int(s, "match");
    in.mismatch = detail::get_int(s, "mismatch");
    in.gap_open = detail::get_int(s, "gap_open");
    in.gap_extend = detail::get_int(s, "gap_extend");
    in.gap_open2   = detail::get_int_or(s, "gap_open2",   in.gap_open);
    in.gap_extend2 = detail::get_int_or(s, "gap_extend2", in.gap_extend);
    in.warmup = static_cast<int>(detail::get_int_or(s, "warmup", 3));
    in.iterations = static_cast<int>(detail::get_int_or(s, "iterations", 20));
    in.band_width = detail::get_int_or(s, "band_width", -1);
    in.approx_max = detail::get_int_or(s, "approx_max", 0) != 0;
    return in;
}

struct TimedRun {
    double median = 0.0;
    std::vector<double> samples;
};

struct Timer {
    // Runs `run` `warmup` times (untimed), then `iters` times (timed).
    template <class F>
    TimedRun timed_run(int warmup, int iters, F&& run) {
        for (int i = 0; i < warmup; ++i) run();
        TimedRun out;
        out.samples.reserve(static_cast<std::size_t>(std::max(iters, 0)));
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            run();
            const auto t1 = std::chrono::steady_clock::now();
            out.samples.push_back(std::chrono::duration<double>(t1 - t0).count());
        }
        if (out.samples.empty()) return out;
        std::sort(out.samples.begin(), out.samples.end());
        const std::size_t n = out.samples.size();
        out.median = (n % 2) ? out.samples[n / 2]
                               : (out.samples[n / 2 - 1] + out.samples[n / 2]) / 2.0;
        return out;
    }

    // Backward-compatible median-only helper.
    template <class F>
    double median_seconds(int warmup, int iters, F&& run) {
        return timed_run(warmup, iters, std::forward<F>(run)).median;
    }
};

inline void print_result(long score, const TimedRun& timed, long cells,
                         const AlignmentTraceback& alignment) {
    const double seconds = timed.median;
    const double gcups = seconds > 0.0 ? static_cast<double>(cells) / seconds / 1e9 : 0.0;
    std::printf("{\"score\": %ld, \"seconds\": %.9g, \"gcups\": %.6g, \"cells\": %ld, \"samples\": [",
                score, seconds, gcups, cells);
    for (std::size_t i = 0; i < timed.samples.size(); ++i) {
        if (i) std::printf(", ");
        std::printf("%.9g", timed.samples[i]);
    }
    std::printf("], \"alignment\": {");
    print_json_string("query", alignment.query);
    std::printf(", ");
    print_json_string("ref", alignment.ref);
    std::printf(", ");
    print_json_string("comp", alignment.comp);
    std::printf("}}\n");
}

inline void print_result(long score, const TimedRun& timed, long cells) {
    print_result(score, timed, cells, AlignmentTraceback{});
}

inline void print_result(long score, double seconds, long cells) {
    TimedRun timed;
    timed.median = seconds;
    timed.samples = {seconds};
    print_result(score, timed, cells);
}

// Run a driver body with argv[1] as the input path; catch std::exception and
// print a diagnostic to stderr (surfaced by the Python harness on failure).
template <class F>
int run_driver(int argc, char** argv, const char* prog, F&& body) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s input.json\n", prog);
        return 2;
    }
    try {
        return body(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "baseline error in %s: %s\n", prog, e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "baseline error in %s: unknown exception\n", prog);
        return 1;
    }
}

}  // namespace baselines
