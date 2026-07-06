// Maximum Energy Boost From Two Drinks (LeetCode 3259): each hour drink A or B; switching
// drink loses the next hour. Mutual max-plus DP (output = dp_A, the energy ending on A):
//   dp_A[i] = A[i] + max(dp_A[i-1], dp_B[i-2])   (stay on A, or switch from B losing hour i-1)
//   dp_B[i] = B[i] + max(dp_B[i-1], dp_A[i-2])
// base cases dp_A[0]=A[0], dp_B[0]=B[0]. Each hour's energy rides on the time-varying taps;
// seeds sa, sb carry the i=0 base and are a large -inf sentinel (the TropMax zero) elsewhere.
// One long stream (num_rows defaults to 1). Integer energies keep the winning max-plus sums
// exact (< 2^53), so correctness is exact — only the sentinel-poisoned losing terms are
// inexact, and those never win the max.
#include "bench.hpp"
#include "leetcode_3259_max_energy_two_drinks.h"

static constexpr double SENTINEL = -1e18;  // stands in for the TropMax zero (-inf)

static void build_inputs(double* a, double* b, double* sa, double* sb, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> e(0, 100);
  for (size_t i = 0; i < N; ++i) {
    a[i] = (double)e(rng);
    b[i] = (double)e(rng);
    sa[i] = (i == 0) ? a[i] : SENTINEL;   // dp_A[0] = A[0]
    sb[i] = (i == 0) ? b[i] : SENTINEL;   // dp_B[0] = B[0]
  }
}

static void serial(const double* a, const double* b, const double* sa, const double* sb,
                   double* outA, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  std::vector<double> outB(N);
  for (size_t i = 0; i < N; ++i) {
    double dA1 = (i >= 1) ? outA[i - 1] : SENTINEL;
    double dB1 = (i >= 1) ? outB[i - 1] : SENTINEL;
    double dA2 = (i >= 2) ? outA[i - 2] : SENTINEL;
    double dB2 = (i >= 2) ? outB[i - 2] : SENTINEL;
    outA[i] = std::max(sa[i], std::max(a[i] + dA1, a[i] + dB2));
    outB[i] = std::max(sb[i], std::max(b[i] + dB1, b[i] + dA2));
  }
}

int main(int argc, char** argv) {
  using K = leetcode_3259_max_energy_two_drinks;
  using Kb = leetcode_3259_max_energy_two_drinks_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1, /*cols=*/1 << 22);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<double> a(d.n), b(d.n), sa(d.n), sb(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(a.data(), b.data(), sa.data(), sb.data(), d.n, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(a.data(), b.data(), sa.data(), sb.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(a.data(), b.data(), sa.data(), sb.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(a.data(), b.data(), sa.data(), sb.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);
  bench::report3("leetcode-3259-max-energy-two-drinks", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
