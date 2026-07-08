// Welford running (population) variance:
//   M(x)  = M(x-1) + (a(x) - M(x-1)) / x                 running mean
//   M2(x) = M2(x-1) + ((x-1)/x)*(a(x) - M(x-1))^2          running sum of squared devs
//   var(x) = M2(x) / x
// The M2 increment is Welford's delta*delta2 rewritten via delta2 = ((x-1)/x)*delta, so
// it depends only on the *previous* mean M(x-1). VecRec runs this as two coupled 1st-order
// recurrences (the mean, plus a prefix sum for M2) joined by pointwise ops. The reciprocal
// weight w(x)=1/x is a pure function of stream position, so it is NOT fed as an input array:
// VecRec computes it on the fly via a ComputedSignal2D (WelfordWeight2D, defined in the
// generated header). a(x) is the only data input. (x is 1-based: element i has x = i+1.)
#include "bench.hpp"
#include "welford_variance.h"

static void serial(const float* a, float* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  float M = 0.0f, M2 = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float x = (float)(i + 1);        // 1-based count, x = i+1
    float delta = a[i] - M;          // a(x) - M(x-1)
    M = M + delta / x;               // update mean to M(x)
    float delta2 = a[i] - M;         // a(x) - M(x)
    M2 = M2 + delta * delta2;        // == ((x-1)/x)*delta^2
    out[i] = M2 / x;                 // population variance
  }
}

int main(int argc, char** argv) {
  using K = welford_variance;            // optimized schedule
  using Kb = welford_variance_baseline;  // un-optimized reference schedule
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
  bench::report3("welford-variance", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
