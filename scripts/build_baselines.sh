#!/usr/bin/env bash
# Configure and build the native baseline drivers, then copy the resulting
# binaries into bin/ (gitignored) where the harness looks for them.
#
# Usage:
#   scripts/build_baselines.sh
#
# Only libraries that were cloned (third_party/<lib>) AND have a driver
# (baselines/<lib>/CMakeLists.txt) are built; the rest are skipped. Re-running is
# incremental (CMake rebuilds only what changed).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/baselines"
BIN_DIR="${REPO_ROOT}/bin"

mkdir -p "${BIN_DIR}"

# Build the vendored HMMER static libs first (idempotent: skips if already built),
# so CMake can enable the profile-HMM Viterbi driver. Non-fatal if it fails — the
# CMake guard then skips hmmer_viterbi and the HMM case reports SKIP.
if [ "${AE_SKIP_HMMER:-0}" = "1" ]; then
  echo ">> AE_SKIP_HMMER=1: skipping HMMER (not used by any paper table)"
else
  bash "${REPO_ROOT}/scripts/build_hmmer.sh" || \
    echo ">> warning: HMMER build failed; profile-hmm-viterbi case will be skipped"
fi

# SeqAn3 requires Clang >= 17. Prefer an explicit modern clang (Homebrew LLVM on
# macOS) over the system default, which may be too old. Override via CC/CXX.
: "${CXX:=$(command -v clang++ || true)}"
: "${CC:=$(command -v clang || true)}"
# CMAKE_POLICY_VERSION_MINIMUM lets us configure vendored libs (e.g. parasail)
# whose cmake_minimum_required predates CMake's supported floor.
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
[ -n "${CXX}" ] && CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="${CXX}")
[ -n "${CC}" ] && CMAKE_ARGS+=(-DCMAKE_C_COMPILER="${CC}")

# A cached build configured with a different compiler can't be reused.
if [ -f "${BUILD_DIR}/CMakeCache.txt" ] && [ -n "${CXX}" ] && \
   ! grep -qF "CMAKE_CXX_COMPILER:FILEPATH=${CXX}" "${BUILD_DIR}/CMakeCache.txt"; then
  echo ">> compiler changed; reconfiguring from scratch"
  rm -rf "${BUILD_DIR}"
fi

cmake -S "${REPO_ROOT}/baselines" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j

# Collect every executable produced under the CMake runtime output dir.
copied=0
if [ -d "${BUILD_DIR}/bin" ]; then
  for f in "${BUILD_DIR}/bin"/*; do
    [ -f "${f}" ] && [ -x "${f}" ] || continue
    cp -f "${f}" "${BIN_DIR}/"
    echo ">> installed bin/$(basename "${f}")"
    copied=$((copied + 1))
  done
fi

if [ "${copied}" -eq 0 ]; then
  echo "No drivers built. Did you run scripts/setup_baselines.sh first?"
fi
echo "Done. ${copied} driver(s) in ${BIN_DIR}"
