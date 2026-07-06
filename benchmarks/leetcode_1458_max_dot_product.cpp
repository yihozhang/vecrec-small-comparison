// Max Dot Product of Two Subsequences (LeetCode 1458):
//   dp(i,j) = max(dp(i-1,j), dp(i,j-1), dp(i-1,j-1) + p(i,j), p(i,j)),  p(i,j)=nums1[i]*nums2[j].
// LCS-shaped 2D TropMax recurrence: +0 up/left taps, the product p on the diagonal tap, and the
// standalone p (start a new pairing at (i,j)) as the ⊕ input. p (materialized, may be negative)
// is used in both roles.
#include "bench.hpp"
#include "leetcode_1458_max_dot_product.h"

static void build_inputs(int32_t* p, int rows, int cols, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> v(-100, 100);
  std::vector<int> a(rows), b(cols);
  for (int y = 0; y < rows; ++y) a[y] = v(rng);
  for (int x = 0; x < cols; ++x) b[x] = v(rng);
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      p[(size_t)y * cols + x] = a[y] * b[x];
}

static void serial(const int32_t* p, int32_t* out, int rows, int cols) {
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      size_t idx = (size_t)y * cols + x;
      int32_t best = p[idx];  // G: start a new pairing at (i,j)
      if (y >= 1) best = std::max(best, out[(size_t)(y - 1) * cols + x]);
      if (x >= 1) best = std::max(best, out[(size_t)y * cols + (x - 1)]);
      if (y >= 1 && x >= 1) best = std::max(best, out[(size_t)(y - 1) * cols + (x - 1)] + p[idx]);
      out[idx] = best;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_1458_max_dot_product;
  using Kb = leetcode_1458_max_dot_product_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/2048, /*cols=*/2048);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int32_t> p(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(p.data(), d.rows, d.cols, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(p.data(), d.cols, d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(p.data(), d.cols, d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(p.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);
  bench::report3("leetcode-1458-max-dot-product", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
