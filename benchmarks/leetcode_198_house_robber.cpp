// House Robber (LeetCode 198): f(i) = max(f(i-1), f(i-2) + nums[i]), i.e. the max
// sum of a subset of a row of houses with no two adjacent. Base cases f(0)=nums[0],
// f(1)=max(nums[0],nums[1]). In the (max, +) tropical semiring the "skip" edge f(i-1)
// has weight +0 and the "rob" edge f(i-2) has the time-varying weight +nums[i]; a seed
// stream s carries the two base cases and is a large -inf sentinel (the TropMax zero)
// elsewhere. One long row of houses (num_rows defaults to 1).
//
// A large FINITE sentinel replaces literal -inf because the build uses -ffast-math;
// integer rewards keep every partial path sum exact (< 2^53), so correctness is exact.
#include "bench.hpp"
#include "leetcode_198_house_robber.h"

static constexpr double SENTINEL = -1e18;  // stands in for the TropMax zero (-inf) in s

static void build_inputs(double* n, double* s, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> reward(0, 100);
  for (size_t i = 0; i < N; ++i) {
    n[i] = (double)reward(rng);
    s[i] = (i < 2) ? n[i] : SENTINEL;   // s[0]=nums[0], s[1]=nums[1] seed the base cases
  }
}

static void serial(const double* n, const double* s, double* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  for (size_t i = 0; i < N; ++i) {
    double f;
    if (i == 0) f = s[0];                                  // f(0) = nums[0]
    else if (i == 1) f = std::max(out[0], s[1]);           // max(nums[0], nums[1])
    else f = std::max(out[i - 1], out[i - 2] + n[i]);
    out[i] = f;
  }
}

int main(int argc, char** argv) {
  using K = leetcode_198_house_robber;
  using Kb = leetcode_198_house_robber_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1, /*cols=*/1 << 22);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<double> n(d.n), s(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(n.data(), s.data(), d.n, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(n.data(), s.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(n.data(), s.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(n.data(), s.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);
  bench::report3("leetcode-198-house-robber", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
