// Decode Ways (LeetCode 91): number of ways to decode a digit string, where
//   f(i) = a(i)*f(i-1) + b(i)*f(i-2)
// a(i)=1 iff digit i is a valid single (1..9), b(i)=1 iff digits (i-1,i) form a valid
// pair (10..26). The counts grow ~Fibonacci and overflow a single long stream, so we run
// many independent length-G strings (one per row): the recurrence resets at each row
// boundary and a seed s injects the f=1 empty-prefix base at each segment's first step.
// int64 holds the counts within a 64-step segment.
//
// G below MUST match specs.registry.DECODE_SEGMENT.
#include "bench.hpp"
#include "leetcode_91_decode_ways.h"

static constexpr int G = 64;  // digits per string (== row_size == DECODE_SEGMENT)

// Per-row 0/1 coefficient streams and the seed. a(t) is mostly 1 (few '0' digits); b(t)
// is a valid pair ~half the time. a(0), b(0), b(1) multiply the zeroed post-reset history
// so their value is irrelevant; set to 0 for a clean match with serial. s = 1 at each
// segment's first step (the empty-prefix base), 0 elsewhere.
static void build_inputs(int64_t* a, int64_t* b, int64_t* s, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> pa(0, 9);   // digit-is-nonzero ~90%
  std::uniform_int_distribution<int> pb(0, 1);   // valid pair ~50%
  for (size_t x = 0; x < N; ++x) {
    int t = (int)(x % G);
    a[x] = (t == 0) ? 0 : (pa(rng) != 0 ? 1 : 0);
    b[x] = (t <= 1) ? 0 : (pb(rng) == 1 ? 1 : 0);
    s[x] = (t == 0) ? 1 : 0;
  }
}

// Serial reference: one independent decoding count per row, history reset at each row start.
static void serial(const int64_t* a, const int64_t* b, const int64_t* s, int64_t* out,
                   int num_rows, int row_size) {
  for (int r = 0; r < num_rows; ++r) {
    int64_t f1 = 0, f2 = 0;  // f(t-1), f(t-2): reset per row
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
  using K = leetcode_91_decode_ways;
  using Kb = leetcode_91_decode_ways_baseline;
  // 65536 independent strings of 64 digits each => 2^22 elements.
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1 << 16, /*cols=*/G);
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
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);  // integer counts, no overflow: exact
  bench::report3("leetcode-91-decode-ways", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
