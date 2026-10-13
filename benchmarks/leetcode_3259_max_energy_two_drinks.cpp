// Maximum Energy Boost From Two Drinks (LeetCode 3259): each hour drink A or B; switching drinks
// loses the next hour. 2x2 mutual (max,+) DP (output dp_A):
//   dp_A[i] = A[i] + max(dp_A[i-1], dp_B[i-2])   (stay on A / switch from B, losing hour i-1)
//   dp_B[i] = B[i] + max(dp_B[i-1], dp_A[i-2])
// Energy rides on the taps; seeds sa, sb carry the i=0 base, -inf elsewhere. Integer energies
// keep the max-plus sums exact.
#include "bench.hpp"
#include "leetcode_3259_max_energy_two_drinks.h"

static constexpr int32_t SENTINEL = INT32_MIN / 8;  // TropMax zero (-inf)

static void build_inputs(int32_t* a, int32_t* b, int32_t* sa, int32_t* sb, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> e(0, 100);
  for (size_t i = 0; i < N; ++i) {
    a[i] = (int32_t)e(rng);
    b[i] = (int32_t)e(rng);
    sa[i] = (i == 0) ? a[i] : SENTINEL;   // dp_A[0] = A[0]
    sb[i] = (i == 0) ? b[i] : SENTINEL;   // dp_B[0] = B[0]
  }
}

// `outB` is a caller-owned num_rows*row_size scratch buffer (allocated outside the timed region).
static void serial(const int32_t* a, const int32_t* b, const int32_t* sa, const int32_t* sb,
                   int32_t* outA, int32_t* outB, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  for (size_t i = 0; i < N; ++i) {
    int32_t dA1 = (i >= 1) ? outA[i - 1] : SENTINEL;
    int32_t dB1 = (i >= 1) ? outB[i - 1] : SENTINEL;
    int32_t dA2 = (i >= 2) ? outA[i - 2] : SENTINEL;
    int32_t dB2 = (i >= 2) ? outB[i - 2] : SENTINEL;
    outA[i] = std::max(sa[i], std::max(a[i] + dA1, a[i] + dB2));
    outB[i] = std::max(sb[i], std::max(b[i] + dB1, b[i] + dA2));
  }
}

int main(int argc, char** argv) {
  using K = leetcode_3259_max_energy_two_drinks;
  using Kb = leetcode_3259_max_energy_two_drinks_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1, /*cols=*/1 << 22);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int32_t> a(d.n), b(d.n), sa(d.n), sb(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  std::vector<int32_t> outB(d.n);  // serial scratch
  build_inputs(a.data(), b.data(), sa.data(), sb.data(), d.n, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(a.data(), b.data(), sa.data(), sb.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(a.data(), b.data(), sa.data(), sb.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(a.data(), b.data(), sa.data(), sb.data(), out_serial.data(), outB.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);
  bench::report3("leetcode-3259-max-energy-two-drinks", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
