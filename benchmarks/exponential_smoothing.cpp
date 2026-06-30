// Exponential smoothing: f(x) = a*f(x-1) + (1-a)*g(x), a = 0.8  (1D).
#include "bench.hpp"
#include "exponential_smoothing.h"

static void serial(const float* g, float* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  const float a = 0.8f, oma = 0.2f;
  float f = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    f = a * f + oma * g[i];
    out[i] = f;
  }
}

int main(int argc, char** argv) {
  using K = exponential_smoothing;            // optimized schedule
  using Kb = exponential_smoothing_baseline;  // un-optimized reference schedule
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1, /*cols=*/1 << 22);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<float> g(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  bench::fill_uniform(g.data(), d.n, -1.0f, 1.0f, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(g.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(g.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(g.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs <= 1e-2 || eo.max_rel <= 1e-2) && (eb.max_abs <= 1e-2 || eb.max_rel <= 1e-2);
  bench::report3("exponential-smoothing", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
