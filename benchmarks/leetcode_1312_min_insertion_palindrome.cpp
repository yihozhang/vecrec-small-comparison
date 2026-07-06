// Minimum Insertion Steps to Make a String Palindrome (LeetCode 1312): f(l,r) = f(l+1,r-1) if
// s[l]==s[r] else 1 + min(f(l+1,r), f(l,r-1)). Reversing the l axis (i = L-1-l) makes it the
// standard edit-distance 2D TropMin recurrence over one string vs itself:
//   D(i,r) = min(D(i-1,r)+1, D(i,r-1)+1, D(i-1,r-1) + c(i,r)),  c = 0 on match else +inf.
// Insert = the +1 up/left taps; the diagonal is free on a match and forbidden (+inf) on a
// mismatch. Deep up/left/diag dependency so the optimizer wins.
#include "bench.hpp"
#include "leetcode_1312_min_insertion_palindrome.h"

#include <limits>

static constexpr int32_t POS = std::numeric_limits<int32_t>::max() / 8;
static constexpr int ALPHA = 4;

// c[y][x] = 0 if s[rows-1-y]==s[x] else POS (one string; row axis reversed).
static void build_inputs(int32_t* c, int rows, int cols, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> ch(0, ALPHA - 1);
  int L = std::max(rows, cols);
  std::vector<int> s(L);
  for (int i = 0; i < L; ++i) s[i] = ch(rng);
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      c[(size_t)y * cols + x] = (s[rows - 1 - y] == s[x]) ? 0 : POS;
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
  using K = leetcode_1312_min_insertion_palindrome;
  using Kb = leetcode_1312_min_insertion_palindrome_baseline;
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
  bench::report3("leetcode-1312-min-insertion-palindrome", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
