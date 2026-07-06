// Longest Palindromic Subsequence (LeetCode 516): f(l,r) = 2 + f(l+1,r-1) if s[l]==s[r] else
// max(f(l+1,r), f(l,r-1)). Reversing the l axis (i = L-1-l) turns (l+1,r-1) into the standard
// (i-1,r-1) diagonal, giving the LCS max-form over one string vs itself with diagonal reward
//   m = 2 on match else 0.
#include "bench.hpp"
#include "leetcode_516_longest_palindromic_subseq.h"

#include <limits>

static constexpr int32_t NEG = std::numeric_limits<int32_t>::min() / 8;
static constexpr int ALPHA = 4;

// m[y][x] = 2 if s[rows-1-y] == s[x] else 0 (one string; row axis reversed).
static void build_inputs(int32_t* m, int rows, int cols, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> ch(0, ALPHA - 1);
  int L = std::max(rows, cols);
  std::vector<int> s(L);
  for (int i = 0; i < L; ++i) s[i] = ch(rng);
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      m[(size_t)y * cols + x] = (s[rows - 1 - y] == s[x]) ? 2 : 0;
}

static void serial(const int32_t* m, int32_t* out, int rows, int cols) {
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      int32_t best = (y == 0 && x == 0) ? 0 : NEG;
      if (y >= 1) best = std::max(best, out[(size_t)(y - 1) * cols + x]);
      if (x >= 1) best = std::max(best, out[(size_t)y * cols + (x - 1)]);
      if (y >= 1 && x >= 1)
        best = std::max(best, out[(size_t)(y - 1) * cols + (x - 1)] + m[(size_t)y * cols + x]);
      out[(size_t)y * cols + x] = best;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_516_longest_palindromic_subseq;
  using Kb = leetcode_516_longest_palindromic_subseq_baseline;
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
  bench::report3("leetcode-516-longest-palindromic-subseq", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
