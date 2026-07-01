// Non-uniform (time-varying) recursive filter:
//   f(x) = a(x)*f(x-1) + b(x)*f(x-2) + g(x)      (1D)
// The feedback coefficients a(x), b(x) vary per sample. We draw them from
// [-0.4, 0.4] so |a|+|b| < 1 everywhere, keeping the time-varying recurrence
// BIBO-stable (bounded output) over the whole stream.
#include "bench.hpp"
#include "nonuniform_recursive_filter.h"

static void serial(const float* a, const float* b, const float* g, float* out,
                   int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  float f1 = 0.0f, f2 = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float f = a[i] * f1 + b[i] * f2 + g[i];
    out[i] = f;
    f2 = f1;
    f1 = f;
  }
}

int main(int argc, char** argv) {
  using K = nonuniform_recursive_filter;            // optimized schedule
  using Kb = nonuniform_recursive_filter_baseline;  // un-optimized reference schedule
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1, /*cols=*/1 << 22);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<float> a(d.n), b(d.n), g(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  bench::fill_uniform(a.data(), d.n, -0.4f, 0.4f, cfg.seed);
  bench::fill_uniform(b.data(), d.n, -0.4f, 0.4f, cfg.seed ^ 0x1111u);
  bench::fill_uniform(g.data(), d.n, -1.0f, 1.0f, cfg.seed ^ 0x2222u);
  // Constructor takes 1D inputs in alphabetical order (a, b, g), then row_size.
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(a.data(), b.data(), g.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(a.data(), b.data(), g.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(a.data(), b.data(), g.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs <= 1e-2 || eo.max_rel <= 1e-2) && (eb.max_abs <= 1e-2 || eb.max_rel <= 1e-2);
  bench::report3("nonuniform-recursive-filter", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
