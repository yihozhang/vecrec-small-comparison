// Maximal Square (LeetCode 221): side of the largest all-ones square ending at (i,j):
//   dp(i,j) = min(dp(i-1,j), dp(i,j-1), dp(i-1,j-1)) + 1  if grid==1 else 0.
// 2D TropMin with constant +1 taps (including the diagonal). The mask g is the input: 0 where
// grid==0 (forces dp=0), 1 on a grid==1 boundary cell (single-cell square), and a large finite
// "+inf" sentinel on grid==1 interior cells so min(neighbors)+1 wins. Values <= min(rows,cols),
// integer, so it is exact.
#include "bench.hpp"
#include "leetcode_221_maximal_square.h"

static constexpr float BIG = 1e9f;  // stand-in for the TropMin zero (+inf) on interior cells

static void build_inputs(float* g, int rows, int cols, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> p(0, 99);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) {
      size_t idx = (size_t)i * cols + j;
      bool one = p(rng) < 70;  // ~70% grid==1
      g[idx] = !one ? 0.0f : ((i == 0 || j == 0) ? 1.0f : BIG);
    }
}

static void serial(const float* g, float* out, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      size_t idx = (size_t)i * cols + j;
      float best = g[idx];
      if (i >= 1) best = std::min(best, out[(size_t)(i - 1) * cols + j] + 1.0f);
      if (j >= 1) best = std::min(best, out[idx - 1] + 1.0f);
      if (i >= 1 && j >= 1) best = std::min(best, out[(size_t)(i - 1) * cols + (j - 1)] + 1.0f);
      out[idx] = best;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_221_maximal_square;
  using Kb = leetcode_221_maximal_square_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/2048, /*cols=*/2048);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<float> g(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(g.data(), d.rows, d.cols, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(g.data(), d.cols, d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(g.data(), d.cols, d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(g.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);
  bench::report3("leetcode-221-maximal-square", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
