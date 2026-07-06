// Delete Operation for Two Strings (LeetCode 583): min deletions to make two strings equal —
// edit distance with insert/delete only. Same 2D (min,+) recurrence as 72, but the diagonal
// (substitution) cost c is +inf on a mismatch so it is never taken (c = 0 if match else +inf).
#include "bench.hpp"
#include "leetcode_583_delete_two_strings.h"

#include <limits>

static constexpr int32_t POS = std::numeric_limits<int32_t>::max() / 8;  // TropMin zero (+inf)
static constexpr int ALPHA = 4;

// c[y][x] = 0 if match else POS (substitution forbidden — must insert/delete instead).
static void build_inputs(int32_t* c, int rows, int cols, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> ch(0, ALPHA - 1);
  std::vector<int> s1(rows), s2(cols);
  for (int y = 0; y < rows; ++y) s1[y] = ch(rng);
  for (int x = 0; x < cols; ++x) s2[x] = ch(rng);
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      c[(size_t)y * cols + x] = (s1[y] == s2[x]) ? 0 : POS;
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
  using K = leetcode_583_delete_two_strings;
  using Kb = leetcode_583_delete_two_strings_baseline;
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
  bench::report3("leetcode-583-delete-two-strings", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
