// Count Ways to Build Good Strings (LeetCode 2466):
//   f(i) = f(i-1) + f(i-3),  seeded by s (a unit impulse at each segment's first step).
// The counts grow ~1.47^n and overflow a single long stream, so we run many short
// *independent* segments (one per row): the recurrence resets at each row boundary and s
// re-seeds it, keeping every segment's counts well within int64. G is the segment length
// (== row_size).
#include "bench.hpp"
#include "leetcode_2466_count_good_strings.h"

static constexpr int ZERO = 1;
static constexpr int ONE = 3;
static constexpr int G = 64;  // steps per segment (== row_size)

// Per-row seed stream: s = 1 at each segment's first step (t == 0), else 0.
static void build_inputs(int64_t* s, size_t N) {
  for (size_t x = 0; x < N; ++x) s[x] = ((int)(x % G) == 0) ? 1 : 0;
}

// Serial reference: one independent sequence per row, history reset at each row start.
static void serial(const int64_t* s, int64_t* out, int num_rows, int row_size) {
  for (int r = 0; r < num_rows; ++r) {
    for (int t = 0; t < row_size; ++t) {
      size_t x = (size_t)r * row_size + t;
      int64_t v = s[x];                       // seed (1 at t == 0)
      if (t >= ZERO) v += out[x - ZERO];
      if (t >= ONE)  v += out[x - ONE];
      out[x] = v;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_2466_count_good_strings;             // optimized schedule
  using Kb = leetcode_2466_count_good_strings_baseline;   // un-optimized reference schedule
  // 65536 independent segments of 64 steps each => 2^22 elements.
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1 << 16, /*cols=*/G);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int64_t> s(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(s.data(), d.n);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(s.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(s.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(s.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);  // integer counts, no overflow: exact
  bench::report3("leetcode-2466-count-good-strings", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
