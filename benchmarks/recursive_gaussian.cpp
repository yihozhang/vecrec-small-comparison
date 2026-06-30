// Causal recursive Gaussian (Young & van Vliet 1995), sigma = 2.0  (1D):
//   f(x) = a1*f(x-1) + a2*f(x-2) + a3*f(x-3) + B*g(x)
// Coefficients computed exactly as in specs/registry.py so they match the kernel.
#include "bench.hpp"
#include "recursive_gaussian.h"

struct Coef {
  float a1, a2, a3, b;
};
static Coef gaussian_coef(double sigma) {
  double q = (sigma < 2.5) ? 1.31564 * (std::sqrt(1.0 + 0.490811 * sigma * sigma) - 1.0)
                           : 0.98711 * sigma - 0.96330;
  double b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
  double b1 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
  double b2 = -1.4281 * q * q - 1.26661 * q * q * q;
  double b3 = 0.422205 * q * q * q;
  double a1 = b1 / b0, a2 = b2 / b0, a3 = b3 / b0;
  double B = 1.0 - (a1 + a2 + a3);
  return Coef{(float)a1, (float)a2, (float)a3, (float)B};
}

static void serial(const float* g, float* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  Coef c = gaussian_coef(2.0);
  float f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float f = c.a1 * f1 + c.a2 * f2 + c.a3 * f3 + c.b * g[i];
    out[i] = f;
    f3 = f2;
    f2 = f1;
    f1 = f;
  }
}

int main(int argc, char** argv) {
  using K = recursive_gaussian;
  using Kb = recursive_gaussian_baseline;
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
  bench::report3("recursive-gaussian", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
