// O(rows+cols) alignment tracebacks on pre-computed DP score matrices.
//
// Each function takes the flat (rows x cols) score matrix H produced by the
// corresponding Python reference DP, reconstructs the boundary row/col inline,
// and walks backwards from the optimal cell to produce three alignment strings.
//
// All four variants share the same greedy priority: diagonal > up > left.
// For affine gap (Gotoh) this may produce a different valid optimum than
// the canonical one when multiple optimal paths exist, since H = max(M,Dx,Dy)
// does not encode which state generated each cell.

#include <algorithm>
#include <cstdint>
#include <cstring>

static inline char comp_char(char q, char r) {
    if (q == '-' || r == '-') return ' ';
    return (q == r) ? '|' : '.';
}

static void reverse_inplace(char* s, int n) {
    for (int lo = 0, hi = n - 1; lo < hi; ++lo, --hi) {
        char t = s[lo]; s[lo] = s[hi]; s[hi] = t;
    }
}

extern "C" {

// Global NW traceback.  gap < 0.  Returns alignment length.
int nw_traceback_c(
    const int64_t* H, int rows, int cols,
    const int32_t* s, const char* x, const char* y, int gap,
    char* q_out, char* r_out, char* c_out)
{
    auto full = [&](int i, int j) -> int64_t {
        if (i == 0 && j == 0) return 0;
        if (i == 0) return (int64_t)j * gap;
        if (j == 0) return (int64_t)i * gap;
        return H[(i - 1) * cols + (j - 1)];
    };

    int n = 0, i = rows, j = cols;
    while (i > 0 || j > 0) {
        if (i == 0) {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        } else if (j == 0) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else if (full(i, j) == full(i-1, j-1) + (int64_t)s[(i-1)*cols+(j-1)]) {
            q_out[n] = x[i-1]; r_out[n] = y[j-1];
            c_out[n] = comp_char(x[i-1], y[j-1]); ++n; --i; --j;
        } else if (full(i, j) == full(i-1, j) + gap) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        }
    }
    q_out[n] = r_out[n] = c_out[n] = '\0';
    reverse_inplace(q_out, n);
    reverse_inplace(r_out, n);
    reverse_inplace(c_out, n);
    return n;
}

// Local SW traceback (stops when cell == 0).  gap < 0.  Returns alignment length.
int sw_traceback_c(
    const int64_t* H, int rows, int cols,
    const int32_t* s, const char* x, const char* y, int gap,
    char* q_out, char* r_out, char* c_out)
{
    int64_t best = 0;
    int bi = 0, bj = 0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            if (H[r * cols + c] > best) {
                best = H[r * cols + c]; bi = r + 1; bj = c + 1;
            }

    auto full = [&](int i, int j) -> int64_t {
        if (i == 0 || j == 0) return 0;
        return H[(i - 1) * cols + (j - 1)];
    };

    int n = 0, i = bi, j = bj;
    while (i > 0 && j > 0 && full(i, j) > 0) {
        if (full(i, j) == full(i-1, j-1) + (int64_t)s[(i-1)*cols+(j-1)]) {
            q_out[n] = x[i-1]; r_out[n] = y[j-1];
            c_out[n] = comp_char(x[i-1], y[j-1]); ++n; --i; --j;
        } else if (full(i, j) == full(i-1, j) + gap) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        }
    }
    q_out[n] = r_out[n] = c_out[n] = '\0';
    reverse_inplace(q_out, n);
    reverse_inplace(r_out, n);
    reverse_inplace(c_out, n);
    return n;
}

// Global affine (Gotoh) traceback.  gap_open < 0, gap_extend < 0.
// Greedy from H = max(M, Dx, Dy).  Returns alignment length.
int affine_traceback_c(
    const int64_t* H, int rows, int cols,
    const int32_t* s, const char* x, const char* y,
    int gap_open, int gap_extend,
    char* q_out, char* r_out, char* c_out)
{
    const int64_t GO_GE = (int64_t)gap_open + gap_extend;
    const int64_t GE    = (int64_t)gap_extend;

    auto full = [&](int i, int j) -> int64_t {
        if (i == 0 && j == 0) return 0;
        if (i == 0) return (int64_t)gap_open + (int64_t)j * gap_extend;
        if (j == 0) return (int64_t)gap_open + (int64_t)i * gap_extend;
        return H[(i - 1) * cols + (j - 1)];
    };

    int n = 0, i = rows, j = cols;
    while (i > 0 || j > 0) {
        if (i == 0) {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        } else if (j == 0) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else if (full(i, j) == full(i-1, j-1) + (int64_t)s[(i-1)*cols+(j-1)]) {
            q_out[n] = x[i-1]; r_out[n] = y[j-1];
            c_out[n] = comp_char(x[i-1], y[j-1]); ++n; --i; --j;
        } else if (full(i, j) == full(i-1, j) + GO_GE ||
                   full(i, j) == full(i-1, j) + GE) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        }
    }
    q_out[n] = r_out[n] = c_out[n] = '\0';
    reverse_inplace(q_out, n);
    reverse_inplace(r_out, n);
    reverse_inplace(c_out, n);
    return n;
}

// Unit-cost Levenshtein traceback.  Returns alignment length.
int levenshtein_traceback_c(
    const int64_t* D, int rows, int cols,
    const char* x, const char* y,
    char* q_out, char* r_out, char* c_out)
{
    auto full = [&](int i, int j) -> int64_t {
        if (i == 0) return (int64_t)j;
        if (j == 0) return (int64_t)i;
        return D[(i - 1) * cols + (j - 1)];
    };

    int n = 0, i = rows, j = cols;
    while (i > 0 || j > 0) {
        if (i == 0) {
            q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
        } else if (j == 0) {
            q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
        } else {
            const int64_t cost = (x[i-1] == y[j-1]) ? 0 : 1;
            if (full(i, j) == full(i-1, j-1) + cost) {
                q_out[n] = x[i-1]; r_out[n] = y[j-1];
                c_out[n] = comp_char(x[i-1], y[j-1]); ++n; --i; --j;
            } else if (full(i, j) == full(i-1, j) + 1) {
                q_out[n] = x[i-1]; r_out[n] = '-'; c_out[n] = ' '; ++n; --i;
            } else {
                q_out[n] = '-'; r_out[n] = y[j-1]; c_out[n] = ' '; ++n; --j;
            }
        }
    }
    q_out[n] = r_out[n] = c_out[n] = '\0';
    reverse_inplace(q_out, n);
    reverse_inplace(r_out, n);
    reverse_inplace(c_out, n);
    return n;
}

} // extern "C"
