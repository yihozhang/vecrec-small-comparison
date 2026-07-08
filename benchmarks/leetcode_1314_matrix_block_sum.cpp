// Matrix Block Sum (LeetCode 1314): each answer[i][j] is the sum of a (2k+1)x(2k+1) block, which
// is computed in O(1) from a 2D prefix sum (summed-area table) of the matrix — the same recurrence
// as 304:  f[i][j] = m[i][j] + f[i-1][j] + f[i][j-1] - f[i-1][j-1].
#include "bench.hpp"
#include "leetcode_1314_matrix_block_sum.h"

// m[i][j] = small non-negative integer; the summed-area table over a 2048x2048 grid
// stays well within int32 (max ~9 * N cells).
static void build_inputs(int32_t* m, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> v(0, 9);
  for (size_t i = 0; i < N; ++i) m[i] = (int32_t)v(rng);
}

static void serial(const int32_t* m, int32_t* out, int rows, int cols) {
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      int32_t up = (y >= 1) ? out[(size_t)(y - 1) * cols + x] : 0;
      int32_t left = (x >= 1) ? out[(size_t)y * cols + (x - 1)] : 0;
      int32_t upleft = (y >= 1 && x >= 1) ? out[(size_t)(y - 1) * cols + (x - 1)] : 0;
      out[(size_t)y * cols + x] = m[(size_t)y * cols + x] + up + left - upleft;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_1314_matrix_block_sum;
  using Kb = leetcode_1314_matrix_block_sum_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/2048, /*cols=*/2048);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int32_t> m(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  // int32 SAT: addition is associative, so VecRec's reassociation is exact (no rounding).
  build_inputs(m.data(), d.n, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(m.data(), d.cols, d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(m.data(), d.cols, d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(m.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);  // int32: exact integer match
  bench::report3("leetcode-1314-matrix-block-sum", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
