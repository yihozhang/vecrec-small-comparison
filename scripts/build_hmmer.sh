#!/usr/bin/env bash
# Build the vendored HMMER (+ bundled Easel) static libraries.
#
# HMMER uses an autotools build that generates config headers (p7_config.h,
# esl_config.h) and selects a SIMD DP backend (SSE/AVX/NEON) for the host. We
# build it in-place once; the profile-HMM Viterbi driver (baselines/hmmer/) then
# links against third_party/hmmer-3.4/src/libhmmer.a + easel/libeasel.a.
#
# The release tarball ships a pre-generated ./configure, so this needs only a C
# compiler + make (no autoconf). Re-run after a clean checkout.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HMMER_DIR="$HERE/third_party/hmmer-3.4"
HMMER_VERSION="3.4"
TARBALL_URL="http://eddylab.org/software/hmmer/hmmer-${HMMER_VERSION}.tar.gz"

if [ ! -d "$HMMER_DIR" ]; then
  echo "==> downloading hmmer-${HMMER_VERSION}"
  curl -sSL -o "$HERE/third_party/hmmer.tar.gz" "$TARBALL_URL"
  tar xzf "$HERE/third_party/hmmer.tar.gz" -C "$HERE/third_party"
  rm -f "$HERE/third_party/hmmer.tar.gz"
fi

cd "$HMMER_DIR"
if [ ! -f src/libhmmer.a ] || [ ! -f easel/libeasel.a ]; then
  echo "==> configuring"
  ./configure --quiet
  echo "==> building (this takes a few minutes)"
  make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

echo "==> done:"
ls -la "$HMMER_DIR"/src/libhmmer.a "$HMMER_DIR"/easel/libeasel.a
