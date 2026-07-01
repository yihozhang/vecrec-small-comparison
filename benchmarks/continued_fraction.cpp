// Continued fraction via the Wallis-Euler recurrence (float64):
//   h(x) = a(x)*h(x-1) + b(x)*h(x-2),   f(x) = h(x)/h(x-1)
//
// h grows/decays geometrically, so a single long stream overflows quickly even
// in double. We therefore run many *independent* continued fractions of G=512
// steps each: one CF per harness row (row_size = G). The recurrence state
// resets at each row boundary, so every row is a fresh CF; a seed input s
// injects h=1 at each row's first step (after reset the history is zero, so
// h(0)=s(0)). Both coefficients a(x), b(x) vary per step (generalized continued
// fraction), drawn from [1,2] — kept positive so h stays positive and finite
// across a 512-step segment. VecRec dilates this genuinely lag-1 recurrence to
// vectorize within each segment.
//
// G below MUST match specs.registry.CF_SEGMENT.
#include "bench.hpp"
#include "continued_fraction.h"

static constexpr int G = 512;  // steps per continued fraction (== row_size == CF_SEGMENT)

// Build the per-row (contiguous) a/b/s streams; element x is step t = x % G of
// its continued fraction. a(x), b(x) are the (per-step, positive) generalized-CF
// coefficients; s is the structural seed (1 at each segment's first step). a(0),
// b(0), b(1) multiply the zeroed post-reset history so their value is irrelevant;
// we set them to 0 for a clean match with serial.
static void build_inputs(double* a, double* b, double* s, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> coef(1.0, 2.0);
  for (size_t x = 0; x < N; ++x) {
    int t = (int)(x % G);
    a[x] = (t == 0) ? 0.0 : coef(rng);
    b[x] = (t <= 1) ? 0.0 : coef(rng);
    s[x] = (t == 0) ? 1.0 : 0.0;
  }
}

// Serial reference: one independent CF per row, history reset at each row start.
static void serial(const double* a, const double* b, const double* s, double* out,
                   int num_rows, int row_size) {
  for (int r = 0; r < num_rows; ++r) {
    double h1 = 0.0, h2 = 0.0;  // h(x-1), h(x-2): reset per row
    for (int t = 0; t < row_size; ++t) {
      size_t x = (size_t)r * row_size + t;
      double h = a[x] * h1 + b[x] * h2 + s[x];
      out[x] = h / h1;  // f = h(x)/h(x-1); at t=0, h1=0 => inf (excluded below)
      h2 = h1;
      h1 = h;
    }
  }
}

// Max abs/rel error, skipping each row's t=0 (the ratio there divides by the
// zero post-reset history and is not a real CF output).
static bench::ErrStats masked_err(const double* vr, const double* ref, size_t n, int row_size) {
  bench::ErrStats e;
  for (size_t i = 0; i < n; ++i) {
    if ((int)(i % row_size) == 0) continue;
    double d = std::fabs(vr[i] - ref[i]);
    if (d > e.max_abs) { e.max_abs = d; e.argmax_abs = i; e.a_at = vr[i]; e.b_at = ref[i]; }
    double rel = d / (std::fabs(ref[i]) + 1e-12);
    if (rel > e.max_rel) e.max_rel = rel;
  }
  return e;
}

int main(int argc, char** argv) {
  using K = continued_fraction;            // optimized schedule
  using Kb = continued_fraction_baseline;  // un-optimized reference schedule
  // 8192 independent continued fractions of 512 steps each => 2^22 elements.
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1 << 13, /*cols=*/G);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<double> a(d.n), b(d.n), s(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(a.data(), b.data(), s.data(), d.n, cfg.seed);
  // Constructor takes 1D inputs in alphabetical order (a, b, s), then row_size.
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(a.data(), b.data(), s.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(a.data(), b.data(), s.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(a.data(), b.data(), s.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = masked_err(out_base.data(), out_serial.data(), d.n, d.cols);
  bench::ErrStats eo = masked_err(out_opt.data(), out_serial.data(), d.n, d.cols);
  bool ok = (eo.max_abs <= 1e-3 || eo.max_rel <= 1e-3) && (eb.max_abs <= 1e-3 || eb.max_rel <= 1e-3);
  bench::report3("continued-fraction", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
