// Seam carving DP:  D[y][x] = E[y][x] + min(D[y-1][x-1], D[y-1][x], D[y-1][x+1])
//
// NOTE: This benchmark has NO VecRec kernel. The centered 3-way min reads a *future*
// column (x+1) of the previous row, i.e. a negative column lag. VecRec's TIKernel2D
// only represents non-negative column lags (x-j), so the recurrence is not expressible
// without extending the kernel representation. We provide the serial baseline for
// reference and time it; there is nothing to compare against.
#include "bench.hpp"

#include <limits>

static void serial(const float* E, float* D, int rows, int cols) {
  const float INF = std::numeric_limits<float>::infinity();
  for (int x = 0; x < cols; ++x) D[x] = E[x];  // row 0
  for (int y = 1; y < rows; ++y) {
    const float* prev = D + (size_t)(y - 1) * cols;
    float* cur = D + (size_t)y * cols;
    const float* e = E + (size_t)y * cols;
    for (int x = 0; x < cols; ++x) {
      float m = prev[x];
      if (x >= 1) m = std::min(m, prev[x - 1]);
      if (x + 1 < cols) m = std::min(m, prev[x + 1]);
      cur[x] = e[x] + m;
      (void)INF;
    }
  }
}

int main(int argc, char** argv) {
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/2048, /*cols=*/2048);
  const size_t n = (size_t)cfg.num_rows * (size_t)cfg.row_size;
  std::vector<float> E(n), D(n);
  bench::fill_uniform(E.data(), n, 0.0f, 1.0f, cfg.seed);

  bench::Timing ts = bench::run_serial(
      [&] { serial(E.data(), D.data(), cfg.num_rows, cfg.row_size); }, cfg.warmup, cfg.iters);

  std::printf("\n==== seam-carving ====\n");
  std::printf("  elements        : %zu\n", n);
  std::printf("  serial  (best)  : %10.1f us   %7.3f Gelem/s\n",
              ts.best_ns / 1e3, ts.best_ns > 0 ? (double)n / ts.best_ns : 0.0);
  std::printf("  VecRec          : UNSUPPORTED\n");
  std::printf("  reason          : centered min needs a future-column (x+1) tap in the\n");
  std::printf("                    previous row; VecRec's TIKernel2D only supports\n");
  std::printf("                    non-negative column lags (x-j).\n");
  // Columns match bench::report3's RESULT line; serial_us is real, VecRec fields are nan.
  std::printf("RESULT,seam-carving,%zu,%.3f,nan,nan,nan,nan,nan,nan,nan,UNSUPPORTED\n", n, ts.best_ns / 1e3);
  std::fflush(stdout);
  return 0;
}
