// Second-order high pass: f(x) = 1.6*f(x-1) - 0.64*f(x-2) + g(x) - 2*g(x-1) + g(x-2)  (1D).
#include "bench.hpp"
#include "second_order_high_pass.h"

static void serial(const float* g, float* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  float f1 = 0.0f, f2 = 0.0f, g1 = 0.0f, g2 = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float g0 = g[i];
    float f = 1.6f * f1 - 0.64f * f2 + g0 - 2.0f * g1 + g2;
    out[i] = f;
    f2 = f1;
    f1 = f;
    g2 = g1;
    g1 = g0;
  }
}

int main(int argc, char** argv) {
  using K = second_order_high_pass;
  using Kb = second_order_high_pass_baseline;
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
  bench::report3("second-order-high-pass", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
