// 2D distance transform in the TropMin (min, +) semiring with a knight-move kernel:
//   f[y][x] = min(g[y][x],
//                 f[y][x-1]+1, f[y-1][x]+1, f[y-1][x-1]+sqrt2,
//                 f[y-1][x-2]+sqrt5, f[y-2][x-1]+sqrt5)
#include "bench.hpp"
#include "distance_transform.h"

#include <limits>

static void serial(const float* g, float* out, int rows, int cols) {
  const float s2 = std::sqrt(2.0f), s5 = std::sqrt(5.0f);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      float best = g[(size_t)y * cols + x];
      auto rd = [&](int yy, int xx) { return out[(size_t)yy * cols + xx]; };
      if (x >= 1) best = std::min(best, rd(y, x - 1) + 1.0f);
      if (y >= 1) best = std::min(best, rd(y - 1, x) + 1.0f);
      if (y >= 1 && x >= 1) best = std::min(best, rd(y - 1, x - 1) + s2);
      if (y >= 1 && x >= 2) best = std::min(best, rd(y - 1, x - 2) + s5);
      if (y >= 2 && x >= 1) best = std::min(best, rd(y - 2, x - 1) + s5);
      out[(size_t)y * cols + x] = best;
    }
  }
}

int main(int argc, char** argv) {
  using K = distance_transform;
  using Kb = distance_transform_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/2048, /*cols=*/2048);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<float> g(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  bench::fill_uniform(g.data(), d.n, 0.0f, 1.0f, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(g.data(), d.cols, d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(g.data(), d.cols, d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(g.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs <= 1e-1 || eo.max_rel <= 1e-3) && (eb.max_abs <= 1e-1 || eb.max_rel <= 1e-3);
  bench::report3("distance-transform", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
