// Welford running mean:
//   M(x) = M(x-1) + (a(x) - M(x-1)) / x
//        = (1 - w(x))*M(x-1) + w(x)*a(x),   w(x) = 1/x      (1D)
// The reciprocal weight w(x)=1/x is a pure function of stream position, so it is
// NOT fed as an input array: VecRec computes it on the fly via a ComputedSignal2D
// (WelfordWeight2D, defined in the generated header) that tracks the column cursor
// and emits a fully-vectorized reciprocal. a(x) is the only data input; the result
// is the cumulative average of a. (x is 1-based: element i has x = i+1.)
#include "bench.hpp"
#include "welford_mean.h"

static void serial(const float* a, float* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  float M = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float w = 1.0f / (float)(i + 1);  // w(x) = 1/x, x = i+1
    M = M + (a[i] - M) * w;           // == (1-w)*M + w*a
    out[i] = M;
  }
}

int main(int argc, char** argv) {
  using K = welford_mean;            // optimized schedule
  using Kb = welford_mean_baseline;  // un-optimized reference schedule
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1, /*cols=*/1 << 22);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<float> a(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  bench::fill_uniform(a.data(), d.n, -1.0f, 1.0f, cfg.seed);
  // The on-the-fly weight signal w(x)=1/x carries no external buffers, so it is
  // default-constructed inside the kernel; the constructor just takes a and row_size.
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(a.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(a.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(a.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs <= 1e-2 || eo.max_rel <= 1e-2) && (eb.max_abs <= 1e-2 || eb.max_rel <= 1e-2);
  bench::report3("welford-mean", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
