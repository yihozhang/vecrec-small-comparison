// Decode Ways II (LeetCode 639): like 91 but with '*' wildcards, so the coefficients can exceed 1
//   f(i) = a(i)*f(i-1) + b(i)*f(i-2)
// where a(i) is the number of valid single decodings of char i (up to 9 for '*') and b(i) the
// valid two-char decodings. Same 2nd-order time-varying arithmetic recurrence as 91; counts grow
// faster, so shorter (32-step) independent segments per row, seeded f=1 at each segment start. int64.
// G below MUST match the segment length used to build the seed.
#include "bench.hpp"
#include "leetcode_639_decode_ways_ii.h"

static constexpr int G = 32;

// Per-row 0..2 coefficient streams (small so a 32-step segment stays within int64) + seed.
// a(0), b(0), b(1) multiply the zeroed post-reset history so their value is irrelevant; set 0.
static void build_inputs(int64_t* a, int64_t* b, int64_t* s, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> pa(1, 2);   // single-char ways
  std::uniform_int_distribution<int> pb(0, 2);   // two-char ways
  for (size_t x = 0; x < N; ++x) {
    int t = (int)(x % G);
    a[x] = (t == 0) ? 0 : pa(rng);
    b[x] = (t <= 1) ? 0 : pb(rng);
    s[x] = (t == 0) ? 1 : 0;
  }
}

static void serial(const int64_t* a, const int64_t* b, const int64_t* s, int64_t* out,
                   int num_rows, int row_size) {
  for (int r = 0; r < num_rows; ++r) {
    int64_t f1 = 0, f2 = 0;
    for (int t = 0; t < row_size; ++t) {
      size_t x = (size_t)r * row_size + t;
      int64_t f = a[x] * f1 + b[x] * f2 + s[x];
      out[x] = f;
      f2 = f1;
      f1 = f;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_639_decode_ways_ii;
  using Kb = leetcode_639_decode_ways_ii_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1 << 17, /*cols=*/G);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int64_t> a(d.n), b(d.n), s(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(a.data(), b.data(), s.data(), d.n, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(a.data(), b.data(), s.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(a.data(), b.data(), s.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(a.data(), b.data(), s.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);
  bench::report3("leetcode-639-decode-ways-ii", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
