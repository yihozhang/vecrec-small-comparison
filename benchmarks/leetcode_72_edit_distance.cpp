// Edit Distance (LeetCode 72), Levenshtein, as a 2D (min,+) recurrence:
//   D(i,j) = min(D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1) + c(i,j)),  c = 0 if s1[i]==s2[j] else 1.
// Insert/delete are the +1 up/left taps; substitution is the diagonal tap (materialized cost c).
// Impulse(0) plus the +1 taps give the D(0,j)=j / D(i,0)=i boundaries.
#include "bench.hpp"
#include "leetcode_72_edit_distance.h"

#include <limits>

static constexpr int32_t POS = std::numeric_limits<int32_t>::max() / 8;  // TropMin zero (+inf)
static constexpr int ALPHA = 4;

// c[y][x] = substitution cost = 0 if s1[y]==s2[x] else 1.
static void build_inputs(int32_t* c, int rows, int cols, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> ch(0, ALPHA - 1);
  std::vector<int> s1(rows), s2(cols);
  for (int y = 0; y < rows; ++y) s1[y] = ch(rng);
  for (int x = 0; x < cols; ++x) s2[x] = ch(rng);
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      c[(size_t)y * cols + x] = (s1[y] == s2[x]) ? 0 : 1;
}

static void serial(const int32_t* c, int32_t* out, int rows, int cols) {
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      int32_t best = (y == 0 && x == 0) ? 0 : POS;
      if (y >= 1) best = std::min(best, out[(size_t)(y - 1) * cols + x] + 1);
      if (x >= 1) best = std::min(best, out[(size_t)y * cols + (x - 1)] + 1);
      if (y >= 1 && x >= 1)
        best = std::min(best, out[(size_t)(y - 1) * cols + (x - 1)] + c[(size_t)y * cols + x]);
      out[(size_t)y * cols + x] = best;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_72_edit_distance;
  using Kb = leetcode_72_edit_distance_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/2048, /*cols=*/2048);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int32_t> c(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(c.data(), d.rows, d.cols, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(c.data(), d.cols, d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(c.data(), d.cols, d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(c.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);
  bench::report3("leetcode-72-edit-distance", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
