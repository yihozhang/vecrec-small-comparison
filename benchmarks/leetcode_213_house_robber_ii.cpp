// House Robber II (LeetCode 213): circular houses, same per-house recurrence as 198
//   f(i) = max(f(i-1), f(i-2) + nums[i])
// (the circular case runs this linear DP twice over two ranges). (max, +): +0 skip tap,
// +nums rob tap; seed s carries the base cases, -inf elsewhere.
#include "bench.hpp"
#include "leetcode_213_house_robber_ii.h"

static constexpr int32_t SENTINEL = INT32_MIN / 8;  // TropMax zero (-inf)

static void build_inputs(int32_t* n, int32_t* s, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> reward(0, 100);
  for (size_t i = 0; i < N; ++i) {
    n[i] = (int32_t)reward(rng);
    s[i] = (i < 2) ? n[i] : SENTINEL;
  }
}

static void serial(const int32_t* n, const int32_t* s, int32_t* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  for (size_t i = 0; i < N; ++i) {
    int32_t f;
    if (i == 0) f = s[0];
    else if (i == 1) f = std::max(out[0], s[1]);
    else f = std::max(out[i - 1], out[i - 2] + n[i]);
    out[i] = f;
  }
}

int main(int argc, char** argv) {
  using K = leetcode_213_house_robber_ii;
  using Kb = leetcode_213_house_robber_ii_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1, /*cols=*/1 << 22);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int32_t> n(d.n), s(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
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
  bench::report3("leetcode-213-house-robber-ii", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
