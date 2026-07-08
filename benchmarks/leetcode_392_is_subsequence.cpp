// Is Subsequence (LeetCode 392): is string a a subsequence of string b?
//   f(i,j) = f(i,j-1) OR (a[i-1]==b[j-1] AND f(i-1,j-1))
// Boolean OR/AND semiring (OR=max, AND=min on {0,1}), 2D. Left tap passes through (skip b[j-1]),
// diagonal tap is the match mask (consume a[i-1]); seeded true at (0,0). Values 0/1 so it is exact.
#include "bench.hpp"
#include "leetcode_392_is_subsequence.h"

static constexpr int ALPHA = 4;

// m[i][j] = (a[i-1]==b[j-1]) for i,j>=1; seed = 1 at (0,0) else 0.
static void build_inputs(int32_t* m, int32_t* seed, int rows, int cols, uint64_t rng_seed) {
  std::mt19937_64 rng(rng_seed);
  std::uniform_int_distribution<int> ch(0, ALPHA - 1);
  std::vector<int> a(rows - 1), b(cols - 1);
  for (int i = 0; i < rows - 1; ++i) a[i] = ch(rng);
  for (int j = 0; j < cols - 1; ++j) b[j] = ch(rng);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) {
      size_t idx = (size_t)i * cols + j;
      seed[idx] = (i == 0 && j == 0) ? 1 : 0;
      m[idx] = (i >= 1 && j >= 1 && a[i - 1] == b[j - 1]) ? 1 : 0;
    }
}

static void serial(const int32_t* m, const int32_t* seed, int32_t* out, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      size_t idx = (size_t)i * cols + j;
      int32_t v = seed[idx];
      if (j >= 1) v = std::max(v, out[idx - 1]);
      if (i >= 1 && j >= 1) v = std::max(v, std::min(m[idx], out[(size_t)(i - 1) * cols + (j - 1)]));
      out[idx] = v;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_392_is_subsequence;
  using Kb = leetcode_392_is_subsequence_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/2048, /*cols=*/2048);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int32_t> m(d.n), seed(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(m.data(), seed.data(), d.rows, d.cols, cfg.seed);
  // Constructor: K(m, m_stride, s, s_stride, row_size).
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(m.data(), d.cols, seed.data(), d.cols, d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(m.data(), d.cols, seed.data(), d.cols, d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(m.data(), seed.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);
  bench::report3("leetcode-392-is-subsequence", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
