// Box filter (D=11) via a 2D difference kernel + summed-area-table recurrence.
//   fir[y][x] = g[y][x] + g[y-d][x-d] - g[y-d][x] - g[y][x-d]   (d = D-1)
//   sat[y][x] = sat[y-1][x] + sat[y][x-1] - sat[y-1][x-1] + fir[y][x]
// The SAT of the differenced image equals the box-filter result (telescoping).
#include "bench.hpp"
#include "box_filter_sat.h"

static constexpr int D = 11;  // must match specs/registry.py::_box_filter_sat

// `fir` is a caller-owned rows*cols scratch buffer (allocated outside the timed region).
static void serial(const float* g, float* out, float* fir, int rows, int cols) {
  const int d = D - 1;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      float v = g[(size_t)y * cols + x];
      if (y >= d && x >= d) v += g[(size_t)(y - d) * cols + (x - d)];
      if (y >= d) v -= g[(size_t)(y - d) * cols + x];
      if (x >= d) v -= g[(size_t)y * cols + (x - d)];
      fir[(size_t)y * cols + x] = v;
    }
  }
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      float up = (y >= 1) ? out[(size_t)(y - 1) * cols + x] : 0.0f;
      float left = (x >= 1) ? out[(size_t)y * cols + (x - 1)] : 0.0f;
      float upleft = (y >= 1 && x >= 1) ? out[(size_t)(y - 1) * cols + (x - 1)] : 0.0f;
      out[(size_t)y * cols + x] = up + left - upleft + fir[(size_t)y * cols + x];
    }
  }
}

int main(int argc, char** argv) {
  using K = box_filter_sat;
  using Kb = box_filter_sat_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/2048, /*cols=*/2048);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<float> g(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n), fir(d.n);
  bench::fill_uniform(g.data(), d.n, -1.0f, 1.0f, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(g.data(), d.cols, d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(g.data(), d.cols, d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(g.data(), out_serial.data(), fir.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs <= 1e-2 || eo.max_rel <= 1e-2) && (eb.max_abs <= 1e-2 || eb.max_rel <= 1e-2);
  bench::report3("box-filter-sat", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
