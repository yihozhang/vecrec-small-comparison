// O(rows+cols) alignment tracebacks on a *full* padded DP matrix.
//
// The vecrec alignment kernels compute the textbook DP over a grid whose cell
// (0,0) is the boundary origin (consumes no characters) and whose diagonal move
// into (i,j) scores the character pair (x[i-1], y[j-1]). Feeding the kernel an
// (rows+1) x (cols+1) score matrix therefore makes its output the complete
// (rows+1) x (cols+1) textbook matrix `mat`, with `mat[rows][cols]` the optimum.
//
// These tracebacks walk that full matrix directly from (rows, cols) -- no
// boundary reconstruction. `stride` is the *physical* row width (>= cols+1; the
// driver rounds it up to a multiple of the kernel's ele_per_iter), so `mat` and
// the score matrix `s` are indexed `[i*stride + j]`. `x`/`y` are the original
// sequences (lengths rows/cols); the character pair at logical cell (i,j) is
// (x[i-1], y[j-1]). Callers allocate q_out/r_out/c_out of size rows+cols+1.
//
// Templated on element type so they run on the kernel's int16/int32 output with
// no widening copy. The affine variant is greedy (diagonal > up > left) since
// H = max(M, Dx, Dy) does not record which state produced each cell; it yields a
// valid optimal alignment, not necessarily the canonical one.

#pragma once

#include <cstdint>

namespace baselines {

inline char tb_comp_char(char q, char r) {
    if (q == '-' || r == '-') return ' ';
    return (q == r) ? '|' : '.';
}

inline void tb_reverse_inplace(char* s, int n) {
    for (int lo = 0, hi = n - 1; lo < hi; ++lo, --hi) {
        char t = s[lo]; s[lo] = s[hi]; s[hi] = t;
    }
}

// Global Needleman-Wunsch traceback over the full padded matrix. gap < 0.
template <class H, class S>
int nw_traceback_full(
    const H* mat, const S* s, int rows, int cols, int stride,
    const char* x, const char* y, int gap,
    char* q_out, char* r_out, char* c_out)
{
    int n = 0, i = rows, j = cols;
    while (i > 0 || j > 0) {
        if (i == 0) {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        } else if (j == 0) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else if ((int64_t)mat[i*stride+j] ==
                   (int64_t)mat[(i-1)*stride+(j-1)] + (int64_t)s[i*stride+j]) {
            q_out[n] = x[i-1]; r_out[n] = y[j-1];
            c_out[n] = tb_comp_char(x[i-1], y[j-1]); ++n; --i; --j;
        } else if ((int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+j] + gap) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        }
    }
    q_out[n] = r_out[n] = c_out[n] = '\0';
    tb_reverse_inplace(q_out, n);
    tb_reverse_inplace(r_out, n);
    tb_reverse_inplace(c_out, n);
    return n;
}

// Global affine (Gotoh) traceback over the full padded H = max(M, Dx, Dy).
// gap_open/gap_extend < 0. Greedy.
template <class H, class S>
int affine_traceback_full(
    const H* mat, const S* s, int rows, int cols, int stride,
    const char* x, const char* y, int gap_open, int gap_extend,
    char* q_out, char* r_out, char* c_out)
{
    const int64_t GO_GE = (int64_t)gap_open + gap_extend;
    const int64_t GE    = (int64_t)gap_extend;

    int n = 0, i = rows, j = cols;
    while (i > 0 || j > 0) {
        if (i == 0) {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        } else if (j == 0) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else if ((int64_t)mat[i*stride+j] ==
                   (int64_t)mat[(i-1)*stride+(j-1)] + (int64_t)s[i*stride+j]) {
            q_out[n] = x[i-1]; r_out[n] = y[j-1];
            c_out[n] = tb_comp_char(x[i-1], y[j-1]); ++n; --i; --j;
        } else if ((int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+j] + GO_GE ||
                   (int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+j] + GE) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        }
    }
    q_out[n] = r_out[n] = c_out[n] = '\0';
    tb_reverse_inplace(q_out, n);
    tb_reverse_inplace(r_out, n);
    tb_reverse_inplace(c_out, n);
    return n;
}

// Local Smith-Waterman traceback over the full padded matrix. gap < 0.
// Finds the global max cell (the local-alignment optimum) by scanning the
// logical interior, writes it to *score_out, then walks back to the first
// non-positive (restart) cell. The global max is the driver-side "max score"
// tracking -- a full reduction that does not fit the streaming kernel model.
template <class H, class S>
int sw_traceback_full(
    const H* mat, const S* s, int rows, int cols, int stride,
    const char* x, const char* y, int gap,
    char* q_out, char* r_out, char* c_out, int64_t* score_out)
{
    // Find the global max (local-alignment optimum) as a flat SIMD reduction.
    // Boundary row/col and dead columns are 0, and SW scores are >= 0, so the max
    // over the whole contiguous buffer equals the max over the interior; for a
    // positive max the located cell is interior.
    //
    // Pass 1: value max -- a plain reduction the compiler auto-vectorizes (memory
    // bandwidth bound, ~tens of µs). Pass 2: locate -- a *vectorized-compare*
    // first-match with early exit. The naive scalar `if (mat[k]==best) idx=k`
    // last-match pass does NOT vectorize and runs ~4-8x slower; comparing a whole
    // SIMD chunk to the broadcast max and only descending to scalar on a hit keeps
    // the locate near bandwidth too.
    const int total = (rows + 1) * stride;
    H vbest = 0;
    for (int k = 0; k < total; ++k) vbest = mat[k] > vbest ? mat[k] : vbest;
    *score_out = (int64_t)vbest;

    constexpr int L = (int)(16 / sizeof(H));   // int16->8, int32->4, int64->2 (128-bit NEON/SSE)
    typedef H hvec __attribute__((vector_size(16)));
    int idx = 0, k = 0;
    bool found = false;
    hvec bv;
    for (int l = 0; l < L; ++l) bv[l] = vbest;
    for (; k + L <= total; k += L) {
        hvec c;
        __builtin_memcpy(&c, &mat[k], sizeof(hvec));
        hvec eq = (c == bv);
        int any = 0;
        for (int l = 0; l < L; ++l) any |= (eq[l] != 0);
        if (any) {
            for (int l = 0; l < L; ++l) if (mat[k + l] == vbest) { idx = k + l; found = true; break; }
            if (found) break;
        }
    }
    if (!found) { for (; k < total; ++k) if (mat[k] == vbest) { idx = k; break; } }
    int bi = idx / stride, bj = idx % stride;

    int n = 0, i = bi, j = bj;
    while (i > 0 && j > 0 && (int64_t)mat[i*stride+j] > 0) {
        if ((int64_t)mat[i*stride+j] ==
            (int64_t)mat[(i-1)*stride+(j-1)] + (int64_t)s[i*stride+j]) {
            q_out[n] = x[i-1]; r_out[n] = y[j-1];
            c_out[n] = tb_comp_char(x[i-1], y[j-1]); ++n; --i; --j;
        } else if ((int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+j] + gap) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        }
    }
    q_out[n] = r_out[n] = c_out[n] = '\0';
    tb_reverse_inplace(q_out, n);
    tb_reverse_inplace(r_out, n);
    tb_reverse_inplace(c_out, n);
    return n;
}

// Unit-cost Levenshtein traceback over the full padded matrix.
template <class H>
int levenshtein_traceback_full(
    const H* mat, int rows, int cols, int stride,
    const char* x, const char* y,
    char* q_out, char* r_out, char* c_out)
{
    int n = 0, i = rows, j = cols;
    while (i > 0 || j > 0) {
        if (i == 0) {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        } else if (j == 0) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else {
            const int64_t cost = (x[i-1] == y[j-1]) ? 0 : 1;
            if ((int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+(j-1)] + cost) {
                q_out[n] = x[i-1]; r_out[n] = y[j-1];
                c_out[n] = tb_comp_char(x[i-1], y[j-1]); ++n; --i; --j;
            } else if ((int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+j] + 1) {
                q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
            } else {
                q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
            }
        }
    }
    q_out[n] = r_out[n] = c_out[n] = '\0';
    tb_reverse_inplace(q_out, n);
    tb_reverse_inplace(r_out, n);
    tb_reverse_inplace(c_out, n);
    return n;
}

// ---------------------------------------------------------------------------
// Inline-substitution variants for the profile-kernel path: the substitution
// matrix s is never materialized, so the traceback recomputes the diagonal score
// from the sequences (subst = (x==y) ? match : mismatch) along the O(path) walk.
// ---------------------------------------------------------------------------

template <class H>
int nw_traceback_full_inline(
    const H* mat, int rows, int cols, int stride,
    const char* x, const char* y, int gap, int match, int mismatch,
    char* q_out, char* r_out, char* c_out)
{
    int n = 0, i = rows, j = cols;
    while (i > 0 || j > 0) {
        if (i == 0) { q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j; }
        else if (j == 0) { q_out[n]=x[i-1]; r_out[n]='-'; c_out[n]=' '; ++n; --i; }
        else {
            const int64_t sub = (x[i-1] == y[j-1]) ? match : mismatch;
            if ((int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+(j-1)] + sub) {
                q_out[n]=x[i-1]; r_out[n]=y[j-1]; c_out[n]=tb_comp_char(x[i-1],y[j-1]); ++n; --i; --j;
            } else if ((int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+j] + gap) {
                q_out[n]=x[i-1]; r_out[n]='-'; c_out[n]=' '; ++n; --i;
            } else { q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j; }
        }
    }
    q_out[n]=r_out[n]=c_out[n]='\0';
    tb_reverse_inplace(q_out,n); tb_reverse_inplace(r_out,n); tb_reverse_inplace(c_out,n);
    return n;
}

template <class H>
int affine_traceback_full_inline(
    const H* mat, int rows, int cols, int stride,
    const char* x, const char* y, int gap_open, int gap_extend, int match, int mismatch,
    char* q_out, char* r_out, char* c_out)
{
    // Traceback of a Gotoh (affine-gap) alignment from the collapsed H matrix
    // H = max(M, E, F). A gap state is not recoverable from a single H
    // predecessor (an "extend" step is only valid inside an open gap), but
    // E[i][j] = max_{k>=1} H[i-k][j] + gap_open + k*gap_extend (and likewise F
    // along a row), so a gap of length k is identified by scanning k until the
    // opening cell is found and then emitted whole. The old heuristic that
    // accepted H[i][j] == H[i-1][j] + gap_extend as a step produced alignments
    // whose re-computed score was far below the reported optimum.
    const int64_t GO=(int64_t)gap_open, GE=(int64_t)gap_extend;
    int n = 0, i = rows, j = cols;
    auto at = [&](int a, int b) -> int64_t { return (int64_t)mat[(size_t)a*stride+b]; };
    while (i > 0 || j > 0) {
        if (i == 0) { q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j; continue; }
        if (j == 0) { q_out[n]=x[i-1]; r_out[n]='-'; c_out[n]=' '; ++n; --i; continue; }
        const int64_t h = at(i, j);
        const int64_t sub = (x[i-1] == y[j-1]) ? match : mismatch;
        if (h == at(i-1, j-1) + sub) {
            q_out[n]=x[i-1]; r_out[n]=y[j-1]; c_out[n]=tb_comp_char(x[i-1],y[j-1]); ++n; --i; --j;
            continue;
        }
        int k = 0;
        for (int t = 1; t <= i; ++t) if (h == at(i-t, j) + GO + (int64_t)t*GE) { k = t; break; }
        if (k > 0) {   // vertical gap of length k (consumes x)
            for (int t = 0; t < k; ++t) { q_out[n]=x[i-1]; r_out[n]='-'; c_out[n]=' '; ++n; --i; }
            continue;
        }
        for (int t = 1; t <= j; ++t) if (h == at(i, j-t) + GO + (int64_t)t*GE) { k = t; break; }
        if (k > 0) {   // horizontal gap of length k (consumes y)
            for (int t = 0; t < k; ++t) { q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j; }
            continue;
        }
        // No consistent predecessor (e.g. a saturated cell): fall back to one
        // horizontal step so the traceback always terminates.
        q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j;
    }
    q_out[n]=r_out[n]=c_out[n]='\0';
    tb_reverse_inplace(q_out,n); tb_reverse_inplace(r_out,n); tb_reverse_inplace(c_out,n);
    return n;
}

template <class H>
int sw_traceback_full_inline(
    const H* mat, int rows, int cols, int stride,
    const char* x, const char* y, int gap, int match, int mismatch,
    char* q_out, char* r_out, char* c_out, int64_t* score_out)
{
    const int total = (rows + 1) * stride;
    H vbest = 0;
    for (int k = 0; k < total; ++k) vbest = mat[k] > vbest ? mat[k] : vbest;
    *score_out = (int64_t)vbest;
    int idx = 0; for (int k = 0; k < total; ++k) if (mat[k] == vbest) { idx = k; break; }
    int n = 0, i = idx / stride, j = idx % stride;
    while (i > 0 && j > 0 && (int64_t)mat[i*stride+j] > 0) {
        const int64_t sub = (x[i-1] == y[j-1]) ? match : mismatch;
        if ((int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+(j-1)] + sub) {
            q_out[n]=x[i-1]; r_out[n]=y[j-1]; c_out[n]=tb_comp_char(x[i-1],y[j-1]); ++n; --i; --j;
        } else if ((int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+j] + gap) {
            q_out[n]=x[i-1]; r_out[n]='-'; c_out[n]=' '; ++n; --i;
        } else { q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j; }
    }
    q_out[n]=r_out[n]=c_out[n]='\0';
    tb_reverse_inplace(q_out,n); tb_reverse_inplace(r_out,n); tb_reverse_inplace(c_out,n);
    return n;
}

// ---------------------------------------------------------------------------
// Banded variants: store only the in-band strip (n_col columns per row).
//
// col_starts[i] = first physical column stored for logical row i
//   = (max(0, i - bw) / EPI) * EPI  (same formula as _band_col_bounds in vecrec_cpp.py)
// n_col = width of each row's slice in band_buf (constant across all rows)
//   = ceil((2*bw + EPI) / EPI) * EPI  (covers the widest in-band range + rounding)
//
// band_buf[(size_t)i * n_col + (j - col_starts[i])] holds the score at (i,j)
// for col_starts[i] <= j < col_starts[i] + n_col; out-of-range positions hold
// the sentinel value (TropMax zero = INT16_MIN/8 for NW/affine, 0 for SW).
// ---------------------------------------------------------------------------

template <class H>
inline H band_get(const H* band_buf, const int* col_starts,
                  int n_col, int i, int j, H sentinel) {
    int lo = col_starts[i];
    if (j < lo || j >= lo + n_col) return sentinel;
    return band_buf[static_cast<std::size_t>(i) * n_col + (j - lo)];
}

// Banded Needleman-Wunsch traceback. gap < 0. sentinel = TropMax zero (INT16_MIN/8).
template <class H>
int nw_traceback_banded_inline(
    const H* band_buf, const int* col_starts, int n_col,
    int rows, int cols,
    const char* x, const char* y, int gap, int match, int mismatch, H sentinel,
    char* q_out, char* r_out, char* c_out)
{
    int n = 0, i = rows, j = cols;
    while (i > 0 || j > 0) {
        if (i == 0) { q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j; }
        else if (j == 0) { q_out[n]=x[i-1]; r_out[n]='-'; c_out[n]=' '; ++n; --i; }
        else {
            const int64_t sub = (x[i-1] == y[j-1]) ? match : mismatch;
            const int64_t cur  = static_cast<int64_t>(band_get(band_buf, col_starts, n_col, i,   j,   sentinel));
            const int64_t diag = static_cast<int64_t>(band_get(band_buf, col_starts, n_col, i-1, j-1, sentinel));
            const int64_t up   = static_cast<int64_t>(band_get(band_buf, col_starts, n_col, i-1, j,   sentinel));
            if (cur == diag + sub) {
                q_out[n]=x[i-1]; r_out[n]=y[j-1]; c_out[n]=tb_comp_char(x[i-1],y[j-1]); ++n; --i; --j;
            } else if (cur == up + gap) {
                q_out[n]=x[i-1]; r_out[n]='-'; c_out[n]=' '; ++n; --i;
            } else { q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j; }
        }
    }
    q_out[n]=r_out[n]=c_out[n]='\0';
    tb_reverse_inplace(q_out,n); tb_reverse_inplace(r_out,n); tb_reverse_inplace(c_out,n);
    return n;
}

// Banded affine (Gotoh) traceback. gap_open/gap_extend < 0. Greedy on collapsed H.
template <class H>
int affine_traceback_banded_inline(
    const H* band_buf, const int* col_starts, int n_col,
    int rows, int cols,
    const char* x, const char* y, int gap_open, int gap_extend, int match, int mismatch, H sentinel,
    char* q_out, char* r_out, char* c_out)
{
    const int64_t GO_GE = static_cast<int64_t>(gap_open) + gap_extend;
    const int64_t GE    = static_cast<int64_t>(gap_extend);
    int n = 0, i = rows, j = cols;
    while (i > 0 || j > 0) {
        if (i == 0) { q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j; }
        else if (j == 0) { q_out[n]=x[i-1]; r_out[n]='-'; c_out[n]=' '; ++n; --i; }
        else {
            const int64_t sub  = (x[i-1] == y[j-1]) ? match : mismatch;
            const int64_t cur  = static_cast<int64_t>(band_get(band_buf, col_starts, n_col, i,   j,   sentinel));
            const int64_t diag = static_cast<int64_t>(band_get(band_buf, col_starts, n_col, i-1, j-1, sentinel));
            const int64_t up   = static_cast<int64_t>(band_get(band_buf, col_starts, n_col, i-1, j,   sentinel));
            if (cur == diag + sub) {
                q_out[n]=x[i-1]; r_out[n]=y[j-1]; c_out[n]=tb_comp_char(x[i-1],y[j-1]); ++n; --i; --j;
            } else if (cur == up + GO_GE || cur == up + GE) {
                q_out[n]=x[i-1]; r_out[n]='-'; c_out[n]=' '; ++n; --i;
            } else { q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j; }
        }
    }
    q_out[n]=r_out[n]=c_out[n]='\0';
    tb_reverse_inplace(q_out,n); tb_reverse_inplace(r_out,n); tb_reverse_inplace(c_out,n);
    return n;
}

// Banded SW walk: gmax/gargmax already found by fused in-sweep reduce. Converts the
// flat gargmax (computed as dst + lane during the sweep, where dst = row_base + rr*phys_w + col)
// to (bi, bj) using phys_w, then walks back through the band strip.
template <class H>
int sw_walk_banded_inline(
    const H* band_buf, const int* col_starts, int n_col,
    int rows, int cols,
    const char* x, const char* y, int gap, int match, int mismatch, H sentinel,
    int bi, int bj,
    char* q_out, char* r_out, char* c_out)
{
    int n = 0, i = bi, j = bj;
    while (i > 0 && j > 0) {
        const int64_t cur = static_cast<int64_t>(band_get(band_buf, col_starts, n_col, i, j, sentinel));
        if (cur <= 0) break;
        const int64_t sub  = (x[i-1] == y[j-1]) ? match : mismatch;
        const int64_t diag = static_cast<int64_t>(band_get(band_buf, col_starts, n_col, i-1, j-1, sentinel));
        const int64_t up   = static_cast<int64_t>(band_get(band_buf, col_starts, n_col, i-1, j,   sentinel));
        if (cur == diag + sub) {
            q_out[n]=x[i-1]; r_out[n]=y[j-1]; c_out[n]=tb_comp_char(x[i-1],y[j-1]); ++n; --i; --j;
        } else if (cur == up + gap) {
            q_out[n]=x[i-1]; r_out[n]='-'; c_out[n]=' '; ++n; --i;
        } else { q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j; }
    }
    q_out[n]=r_out[n]=c_out[n]='\0';
    tb_reverse_inplace(q_out,n); tb_reverse_inplace(r_out,n); tb_reverse_inplace(c_out,n);
    return n;
}

// Walk-only SW traceback: the global max value + start cell (bi,bj) were already
// found by a fused in-sweep reduce, so this just walks back (no scan), inline subst.
template <class H>
int sw_walk_inline(
    const H* mat, int rows, int cols, int stride,
    const char* x, const char* y, int gap, int match, int mismatch,
    int bi, int bj, char* q_out, char* r_out, char* c_out)
{
    int n = 0, i = bi, j = bj;
    while (i > 0 && j > 0 && (int64_t)mat[i*stride+j] > 0) {
        const int64_t sub = (x[i-1] == y[j-1]) ? match : mismatch;
        if ((int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+(j-1)] + sub) {
            q_out[n]=x[i-1]; r_out[n]=y[j-1]; c_out[n]=tb_comp_char(x[i-1],y[j-1]); ++n; --i; --j;
        } else if ((int64_t)mat[i*stride+j] == (int64_t)mat[(i-1)*stride+j] + gap) {
            q_out[n]=x[i-1]; r_out[n]='-'; c_out[n]=' '; ++n; --i;
        } else { q_out[n]='-'; r_out[n]=y[j-1]; c_out[n]=' '; ++n; --j; }
    }
    q_out[n]=r_out[n]=c_out[n]='\0';
    tb_reverse_inplace(q_out,n); tb_reverse_inplace(r_out,n); tb_reverse_inplace(c_out,n);
    return n;
}

}  // namespace baselines
