// Second-order IIR low pass: f(x) = 1.8*f(x-1) - 0.9*f(x-2) + g(x)
// 1D benchmark: a single long continuous IIR (num_rows defaults to 1).
#include "bench.hpp"
#include "second_order_low_pass.h"

// Efficient serial loop. For a 1D recurrence the VecRec kernel treats the input
// as one contiguous stream (reset_and_next_row is a no-op), so the baseline is a
// single continuous IIR over the whole buffer.
static void serial(const float* g, float* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  float f1 = 0.0f, f2 = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float y = 1.8f * f1 - 0.9f * f2 + g[i];
    out[i] = y;
    f2 = f1;
    f1 = y;
  }
}

int main(int argc, char** argv) {
  using K = second_order_low_pass;
  using Kb = second_order_low_pass_baseline;
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
  bench::report3("second-order-low-pass", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
