// Longest Common Subsequence (LeetCode 1143), max-form as a 2D (max,+) recurrence:
//   dp(i,j) = max(dp(i-1,j), dp(i,j-1), dp(i-1,j-1) + m(i,j)),  m = 1 if s1[i]==s2[j] else 0.
// The match reward m rides on the diagonal tap (materialized matrix m); row 0 / column 0 are
// the empty-prefix boundary (dp = 0), seeded by Impulse(0).
#include "bench.hpp"
#include "leetcode_1143_lcs.h"

#include <limits>

static constexpr int32_t NEG = std::numeric_limits<int32_t>::min() / 8;  // TropMax zero (-inf)
static constexpr int ALPHA = 4;  // alphabet size (smaller => more matches => longer LCS)

// Build the match matrix m[y][x] = (s1[y] == s2[x]) ? 1 : 0 from two random strings.
static void build_inputs(int32_t* m, int rows, int cols, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> ch(0, ALPHA - 1);
  std::vector<int> s1(rows), s2(cols);
  for (int y = 0; y < rows; ++y) s1[y] = ch(rng);
  for (int x = 0; x < cols; ++x) s2[x] = ch(rng);
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      m[(size_t)y * cols + x] = (s1[y] == s2[x]) ? 1 : 0;
}

static void serial(const int32_t* m, int32_t* out, int rows, int cols) {
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      int32_t best = (y == 0 && x == 0) ? 0 : NEG;                 // G = Impulse(0)
      if (y >= 1) best = std::max(best, out[(size_t)(y - 1) * cols + x]);
      if (x >= 1) best = std::max(best, out[(size_t)y * cols + (x - 1)]);
      if (y >= 1 && x >= 1)
        best = std::max(best, out[(size_t)(y - 1) * cols + (x - 1)] + m[(size_t)y * cols + x]);
      out[(size_t)y * cols + x] = best;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_1143_lcs;
  using Kb = leetcode_1143_lcs_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/2048, /*cols=*/2048);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int32_t> m(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(m.data(), d.rows, d.cols, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(m.data(), d.cols, d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(m.data(), d.cols, d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(m.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);
  bench::report3("leetcode-1143-lcs", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
