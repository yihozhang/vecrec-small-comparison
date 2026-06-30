// Centralized C++ benchmark harness for comparing VecRec-generated kernels
// against efficient hand-written serial loops.
//
// A per-benchmark .cpp includes this header plus the VecRec-generated kernel
// header (generated/<kernel_name>.h, which is self-contained), defines a serial
// baseline loop, and calls into the functions below. Nothing here depends on
// VecRec Python at runtime.
//
// The VecRec kernel contract this harness drives (all exposed by the generated
// struct after our additive codegen tweak):
//   struct K {
//     static constexpr int ele_per_iter;   // elements produced per run() per block-row
//     static constexpr int rows_per_iter;  // block-rows produced per run()
//     using vector_type = ...;             // SIMD lane type of the output
//     K(<inputs...>, int row_size);        // constructor (caller supplies via a factory)
//     void run(vector_type** out_per_row); // writes ele_per_iter elems into out_per_row[r]
//     void reset_and_next_row();           // advance to next (block of) row(s)
//   };
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace bench {

// ── Config / CLI ────────────────────────────────────────────────────────────
struct Config {
  int num_rows;
  int row_size;
  int warmup = 10;
  int iters = 100;
  uint64_t seed = 0x9e3779b97f4a7c15ULL;
};

inline Config parse(int argc, char** argv, int default_rows, int default_cols) {
  Config c;
  c.num_rows = default_rows;
  c.row_size = default_cols;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> long { return (i + 1 < argc) ? std::atol(argv[++i]) : 0; };
    if (a == "--rows") c.num_rows = (int)next();
    else if (a == "--cols" || a == "--row-size") c.row_size = (int)next();
    else if (a == "--iters") c.iters = (int)next();
    else if (a == "--warmup") c.warmup = (int)next();
    else if (a == "--seed") c.seed = (uint64_t)next();
  }
  return c;
}

// ── Input generation ─────────────────────────────────────────────────────────
inline void fill_uniform(float* p, size_t n, float lo, float hi, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
}

// ── Physical dimensions ──────────────────────────────────────────────────────
// A VecRec kernel always produces ele_per_iter elements per run() and processes
// rows_per_iter rows per block, so we round the logical dimensions up to exact
// multiples. This keeps the driver loop free of partial tiles and prevents any
// out-of-bounds read/write. The rounded grid is a perfectly valid (slightly
// larger) problem instance — both the kernel and the serial baseline run on it.
inline int round_up(int x, int m) { return ((x + m - 1) / m) * m; }

template <class K>
struct Dims {
  int rows;
  int cols;
  size_t n;
};

template <class K>
inline Dims<K> physical_dims(int rows, int cols) {
  int pcols = round_up(cols, K::ele_per_iter);
  int prows = round_up(rows, K::rows_per_iter);
  return Dims<K>{prows, pcols, (size_t)prows * (size_t)pcols};
}

constexpr int cgcd(int a, int b) { return b == 0 ? a : cgcd(b, a % b); }
constexpr int clcm(int a, int b) { return a / cgcd(a, b) * b; }

// Dimensions valid for two kernels at once (e.g. the optimized kernel and its
// baseline), rounded up to a common multiple of both kernels' ele_per_iter /
// rows_per_iter so a single shared buffer works for both.
template <class K1, class K2>
inline Dims<K1> physical_dims2(int rows, int cols) {
  int pcols = round_up(cols, clcm(K1::ele_per_iter, K2::ele_per_iter));
  int prows = round_up(rows, clcm(K1::rows_per_iter, K2::rows_per_iter));
  return Dims<K1>{prows, pcols, (size_t)prows * (size_t)pcols};
}

// ── Timing ───────────────────────────────────────────────────────────────────
struct Timing {
  double best_ns = 0.0;
  double median_ns = 0.0;
};

namespace detail {
template <class Fn>
inline Timing time_loop(Fn&& body, int warmup, int iters) {
  using clk = std::chrono::high_resolution_clock;
  for (int i = 0; i < warmup; ++i) body();
  std::vector<double> samples;
  samples.reserve((size_t)std::max(1, iters));
  for (int i = 0; i < iters; ++i) {
    auto t0 = clk::now();
    body();
    auto t1 = clk::now();
    samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  Timing t;
  t.best_ns = samples.empty() ? 0.0 : samples.front();
  t.median_ns = samples.empty() ? 0.0 : samples[samples.size() / 2];
  return t;
}
}  // namespace detail

// ── Generic VecRec runner ────────────────────────────────────────────────────
// `make` is a factory returning a freshly-constructed kernel. The kernel carries
// recurrence state, so we reconstruct it at the start of every timed iteration
// (construction is cheap — it just stores input pointers). `out` must hold
// num_rows*row_size elements; `row_size` must be a multiple of K::ele_per_iter
// and `num_rows` a multiple of K::rows_per_iter (use physical_dims()).
template <class KFactory, class OutT>
inline Timing run_vecrec(KFactory&& make, int num_rows, int row_size, OutT* out,
                         int warmup, int iters) {
  using K = decltype(make());
  using VT = typename K::vector_type;
  constexpr int EPI = K::ele_per_iter;
  constexpr int RPI = K::rows_per_iter;
  constexpr int LANES = (int)(sizeof(VT) / sizeof(OutT));
  static_assert(EPI % LANES == 0, "ele_per_iter must be a multiple of the lane count");
  constexpr int VEC_PER_ITER = EPI / LANES;

  const size_t N = (size_t)num_rows * (size_t)row_size;
  // Per-block-row scratch (reused across iterations). std::vector<VT> is aligned
  // to alignof(VT); the kernel writes into it via memcpy regardless.
  std::vector<VT> scratch((size_t)RPI * (size_t)VEC_PER_ITER);
  VT* per_row[RPI];

  auto body = [&]() {
    K k = make();
    size_t row_base = 0;
    while (row_base < N) {
      for (int r = 0; r < RPI; ++r) per_row[r] = scratch.data() + (size_t)r * VEC_PER_ITER;
      for (int col = 0; col < row_size; col += EPI) {
        k.run(per_row);
        for (int r = 0; r < RPI; ++r) {
          size_t dst = row_base + (size_t)r * row_size + (size_t)col;
          std::memcpy(out + dst, per_row[r], sizeof(OutT) * (size_t)EPI);
        }
      }
      for (int r = 0; r < RPI; ++r) k.reset_and_next_row();
      row_base += (size_t)RPI * (size_t)row_size;
    }
  };
  return detail::time_loop(body, warmup, iters);
}

// ── Serial baseline runner ───────────────────────────────────────────────────
// `fill` writes the full output once; we time it. After the call returns the
// output buffer holds a valid result, usable for verification.
template <class Fn>
inline Timing run_serial(Fn&& fill, int warmup, int iters) {
  return detail::time_loop(std::forward<Fn>(fill), warmup, iters);
}

// ── Verification ─────────────────────────────────────────────────────────────
struct ErrStats {
  double max_abs = 0.0;
  double max_rel = 0.0;
  size_t argmax_abs = 0;
  double a_at = 0.0;  // value from the first array at argmax_abs
  double b_at = 0.0;  // value from the second array at argmax_abs
};

template <class T>
inline ErrStats compute_err(const T* a, const T* b, size_t n) {
  ErrStats e;
  for (size_t i = 0; i < n; ++i) {
    double av = (double)a[i], bv = (double)b[i];
    double d = std::fabs(av - bv);
    if (d > e.max_abs) {
      e.max_abs = d;
      e.argmax_abs = i;
      e.a_at = av;
      e.b_at = bv;
    }
    double rel = d / (std::fabs(bv) + 1e-12);
    if (rel > e.max_rel) e.max_rel = rel;
  }
  return e;
}

// ── Reporting ────────────────────────────────────────────────────────────────
inline void report(const char* name, const Timing& vr, const Timing& serial,
                   size_t n, const ErrStats& e, bool ok) {
  auto gelem = [&](double ns) { return ns > 0 ? (double)n / ns : 0.0; };  // ns -> Gelem/s
  double speedup = (vr.best_ns > 0) ? serial.best_ns / vr.best_ns : 0.0;
  std::printf("\n==== %s ====\n", name);
  std::printf("  elements        : %zu\n", n);
  std::printf("  VecRec  (best)  : %10.1f us   %7.3f Gelem/s\n", vr.best_ns / 1e3, gelem(vr.best_ns));
  std::printf("  serial  (best)  : %10.1f us   %7.3f Gelem/s\n", serial.best_ns / 1e3, gelem(serial.best_ns));
  std::printf("  VecRec  (median): %10.1f us\n", vr.median_ns / 1e3);
  std::printf("  serial  (median): %10.1f us\n", serial.median_ns / 1e3);
  std::printf("  speedup (VecRec vs serial, best): %.2fx\n", speedup);
  std::printf("  max abs err     : %.3e  (vecrec=%.6g serial=%.6g @ idx %zu)\n",
              e.max_abs, e.a_at, e.b_at, e.argmax_abs);
  std::printf("  max rel err     : %.3e\n", e.max_rel);
  std::printf("  correctness     : %s\n", ok ? "PASS" : "FAIL");
  // Machine-readable line for scripts/run_all.py to aggregate.
  std::printf("RESULT,%s,%zu,%.3f,%.3f,%.4f,%.3e,%.3e,%s\n", name, n, vr.best_ns / 1e3,
              serial.best_ns / 1e3, speedup, e.max_abs, e.max_rel, ok ? "PASS" : "FAIL");
  std::fflush(stdout);
}

// Three-way report: serial loop vs VecRec baseline (un-optimized schedule, codegen
// only) vs VecRec optimized (best schedule). `e_base`/`e_best` are each kernel's error
// vs the serial loop.
inline void report3(const char* name, size_t n, const Timing& serial, const Timing& baseline,
                    const Timing& best, const ErrStats& e_base, const ErrStats& e_best, bool ok) {
  auto gelem = [&](double ns) { return ns > 0 ? (double)n / ns : 0.0; };
  auto sp = [&](double ref, double t) { return t > 0 ? ref / t : 0.0; };
  std::printf("\n==== %s ====\n", name);
  std::printf("  elements                : %zu\n", n);
  std::printf("  serial          (best)  : %10.1f us  %7.3f Gelem/s\n", serial.best_ns / 1e3, gelem(serial.best_ns));
  std::printf("  VecRec baseline (best)  : %10.1f us  %7.3f Gelem/s\n", baseline.best_ns / 1e3, gelem(baseline.best_ns));
  std::printf("  VecRec optimized(best)  : %10.1f us  %7.3f Gelem/s\n", best.best_ns / 1e3, gelem(best.best_ns));
  std::printf("  speedup  opt  vs serial : %.2fx\n", sp(serial.best_ns, best.best_ns));
  std::printf("  speedup  opt  vs base   : %.2fx\n", sp(baseline.best_ns, best.best_ns));
  std::printf("  speedup  base vs serial : %.2fx\n", sp(serial.best_ns, baseline.best_ns));
  std::printf("  max abs err base / opt  : %.3e / %.3e\n", e_base.max_abs, e_best.max_abs);
  std::printf("  correctness             : %s\n", ok ? "PASS" : "FAIL");
  // RESULT,<name>,<n>,<serial_us>,<base_us>,<opt_us>,<opt/serial>,<opt/base>,<base/serial>,<absErrBase>,<absErrOpt>,<status>
  std::printf("RESULT,%s,%zu,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.3e,%.3e,%s\n", name, n,
              serial.best_ns / 1e3, baseline.best_ns / 1e3, best.best_ns / 1e3,
              sp(serial.best_ns, best.best_ns), sp(baseline.best_ns, best.best_ns),
              sp(serial.best_ns, baseline.best_ns), e_base.max_abs, e_best.max_abs,
              ok ? "PASS" : "FAIL");
  std::fflush(stdout);
}

}  // namespace bench
