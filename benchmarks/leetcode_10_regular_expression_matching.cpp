// Regular Expression Matching (LeetCode 10): dp[r][c] = "does s[0..r) match p[0..c)",
// where '.' matches any single char and '*' matches zero-or-more of the preceding element.
// The classic DP is a 2D boolean OR(=max)/AND(=min) recurrence, seeded dp[0][0]=true:
//   dp[r][c] = (M_back1 & dp[r][c-1]) | (M_back2 & dp[r][c-2])     (same row)
//            | (M_up    & dp[r-1][c]) | (M_diag  & dp[r-1][c-1])   (row back)
// Grid row r indexes the string (transition char s[r-1]), column c the pattern (char
// p[c-1]). The four DP branches decide which taps fire per cell, so each tap coefficient is
// a time-varying 0/1 mask (the branch predicate). The masks are emitted on the fly by
// self-describing ComputedSignal2D structs that read s and p from two external int8 buffers.
// We fill those buffers here from a fixed string/pattern (tiled across the grid) and register
// them via the generated header's regex_set_inputs() before running; the serial loop below
// reads the same regex_s / regex_p buffers, so it evaluates the identical recurrence. The
// buffers are registered out-of-band (not via a ComputedSignal2D make_args constructor arg)
// so the kernel constructor stays `(int row_size)` regardless of the autotuned schedule.
#include "bench.hpp"
#include "leetcode_10_regular_expression_matching.h"

// The external string (s) and pattern (p). MUST match specs/registry.py's REGEX_S_INPUT /
// REGEX_P_INPUT so opt/baseline/serial all agree. Letters 'a'..'e'; the pattern mixes '*'
// (zero-or-more of the preceding element) and '.' (any single char) so all four branches fire.
static const char REGEX_S_BASE[] = "abcabcddeeabcde";
static const char REGEX_P_BASE[] = "a*b.c*d.e*b.a*";

static void serial(int32_t* out, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      size_t idx = (size_t)r * cols + c;
      int32_t v = (r == 0 && c == 0) ? 1 : 0;  // dp[0][0]=true (Impulse seed); else false
      char sc  = (r >= 1) ? (char)regex_s[r - 1] : REGEX_SENTINEL;
      char pc  = (c >= 1) ? (char)regex_p[c - 1] : REGEX_SENTINEL;
      char pc2 = (c >= 2) ? (char)regex_p[c - 2] : REGEX_SENTINEL;
      // M_diag: dp[r-1][c-1], active iff p[c-1] matches s[r-1] ('.'/literal).
      if (r >= 1 && c >= 1 && regex_pat_matches(pc, sc))
        v = std::max(v, out[(size_t)(r - 1) * cols + (c - 1)]);
      // M_back2: dp[r][c-2] (zero occurrences), active iff p[c-1]=='*'.
      if (c >= 2 && pc == '*')
        v = std::max(v, out[idx - 2]);
      // M_back1: dp[r][c-1] (one-or-more), active iff p[c-1]=='*' && p[c-2] matches s[r-1].
      if (c >= 1 && pc == '*' && regex_pat_matches(pc2, sc))
        v = std::max(v, out[idx - 1]);
      // M_up: dp[r-1][c] (one-or-more), same condition as M_back1.
      if (r >= 1 && pc == '*' && regex_pat_matches(pc2, sc))
        v = std::max(v, out[(size_t)(r - 1) * cols + c]);
      out[idx] = v;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_10_regular_expression_matching;
  using Kb = leetcode_10_regular_expression_matching_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/2048, /*cols=*/2048);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);

  // Tile the fixed string/pattern across the grid: s is row-indexed (read at s[r-1]), p is
  // column-indexed (read at p[c-1]/p[c-2]). The +64 halo covers the kernel's lane over-read.
  std::vector<int8_t> s(d.rows + 64), p(d.cols + 64);
  for (size_t i = 0; i < s.size(); ++i) s[i] = (int8_t)REGEX_S_BASE[i % (sizeof(REGEX_S_BASE) - 1)];
  for (size_t i = 0; i < p.size(); ++i) p[i] = (int8_t)REGEX_P_BASE[i % (sizeof(REGEX_P_BASE) - 1)];
  regex_set_inputs(s.data(), p.data());  // masks cache these pointers at construction; must precede runs

  std::vector<int32_t> out_opt(d.n), out_base(d.n), out_serial(d.n);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);  // boolean int32: exact match
  bench::report3("leetcode-10-regular-expression-matching", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
