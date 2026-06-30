// Double exponential smoothing (Holt's), output = level l(x).  (1D)
// Rewritten so both states depend only on t-1 (matches the VecRec mutual recurrence):
//   l_t = (1-a) l_{t-1} + (1-a) b_{t-1} + a    g_t
//   b_t = -a*B l_{t-1} + (1-a*B) b_{t-1} + a*B g_t      (a=alpha, B=beta)
#include "bench.hpp"
#include "double_exp_smoothing.h"

static void serial(const float* g, float* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  const float alpha = 0.5f, beta = 0.3f;
  const float oma = 1.0f - alpha;  // 0.5
  const float ab = alpha * beta;   // 0.15
  float l = 0.0f, b = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float ln = oma * l + oma * b + alpha * g[i];
    float bn = -ab * l + (1.0f - ab) * b + ab * g[i];
    l = ln;
    b = bn;
    out[i] = l;
  }
}

int main(int argc, char** argv) {
  using K = double_exp_smoothing;
  using Kb = double_exp_smoothing_baseline;
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
  bench::report3("double-exp-smoothing", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
