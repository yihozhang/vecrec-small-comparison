// Delete and Earn (LeetCode 740): after bucketing, total[v] = v * count(v), and taking value
// v deletes v-1 and v+1 — so over the value axis it is exactly House Robber:
//   dp(v) = max(dp(v-1), dp(v-2) + total[v]),  base cases dp(0)=total[0], dp(1)=max(total[0],total[1]).
// In (max, +): +0 skip tap on dp(v-1), +total tap on dp(v-2); seed s carries the two base cases
// and is a large -inf sentinel elsewhere. One long value axis. Integer totals keep sums exact.
#include "bench.hpp"
#include "leetcode_740_delete_and_earn.h"

static constexpr double SENTINEL = -1e18;

// Inputs in alphabetical order: s (seed), t (total).
static void build_inputs(double* s, double* t, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> total(0, 100);
  for (size_t i = 0; i < N; ++i) {
    t[i] = (double)total(rng);
    s[i] = (i < 2) ? t[i] : SENTINEL;   // dp(0)=total[0], dp(1)=total[1] seed the base cases
  }
}

static void serial(const double* s, const double* t, double* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  for (size_t i = 0; i < N; ++i) {
    double f;
    if (i == 0) f = s[0];                                  // total[0]
    else if (i == 1) f = std::max(out[0], s[1]);           // max(total[0], total[1])
    else f = std::max(out[i - 1], out[i - 2] + t[i]);
    out[i] = f;
  }
}

int main(int argc, char** argv) {
  using K = leetcode_740_delete_and_earn;
  using Kb = leetcode_740_delete_and_earn_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1, /*cols=*/1 << 22);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<double> s(d.n), t(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(s.data(), t.data(), d.n, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(s.data(), t.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(s.data(), t.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(s.data(), t.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);
  bench::report3("leetcode-740-delete-and-earn", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
