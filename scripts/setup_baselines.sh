#!/usr/bin/env bash
# Clone the external C++ alignment libraries into third_party/ (gitignored).
#
# Usage:
#   scripts/setup_baselines.sh            # clone all libraries
#   scripts/setup_baselines.sh seqan3     # clone only the named libraries
#
# Refs are pinned below for reproducibility; override per-lib via env vars, e.g.
#   SEQAN3_REF=3.3.0 scripts/setup_baselines.sh seqan3
#
# Re-running is safe: an existing clone is checked out to the pinned ref rather
# than re-cloned.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"
mkdir -p "${THIRD_PARTY}"

# Pinned refs (tags/branches). Pin to commits for fully reproducible numbers.
SEQAN3_REF="${SEQAN3_REF:-3.4.0}"
PARASAIL_REF="${PARASAIL_REF:-v2.6.2}"
KSW2_REF="${KSW2_REF:-master}"
SSW_REF="${SSW_REF:-master}"
EDLIB_REF="${EDLIB_REF:-v1.2.7}"
WFA2_REF="${WFA2_REF:-v2.3.5}"
MINIMAP2_REF="${MINIMAP2_REF:-v2.28}"

# name|url|ref|recurse_submodules
LIBS=(
  "seqan3|https://github.com/seqan/seqan3|${SEQAN3_REF}|1"
  "parasail|https://github.com/jeffdaily/parasail|${PARASAIL_REF}|0"
  "ksw2|https://github.com/lh3/ksw2|${KSW2_REF}|0"
  "ssw|https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library|${SSW_REF}|0"
  "edlib|https://github.com/Martinsos/edlib|${EDLIB_REF}|0"
  "wfa2|https://github.com/smarco/WFA2-lib|${WFA2_REF}|0"
  # minimap2 supplies the sse2neon shim + ksw2 SIMD .c files the ARM ksw2 driver
  # compiles against (see baselines/ksw2/CMakeLists.txt); needed on Apple Silicon.
  "minimap2|https://github.com/lh3/minimap2|${MINIMAP2_REF}|0"
)

want() {
  # want <name> [filter...]; no filters -> want everything.
  local name="$1"
  shift
  [ "$#" -eq 0 ] && return 0
  for w in "$@"; do [ "$w" = "${name}" ] && return 0; done
  return 1
}

clone_one() {
  local name="$1" url="$2" ref="$3" recurse="$4"
  local dest="${THIRD_PARTY}/${name}"

  if [ -d "${dest}/.git" ]; then
    echo ">> ${name}: already cloned, checking out ${ref}"
    git -C "${dest}" fetch --tags --quiet origin || true
    git -C "${dest}" checkout --quiet "${ref}" || echo "   (could not checkout ${ref}; leaving as-is)"
    [ "${recurse}" = "1" ] && git -C "${dest}" submodule update --init --recursive --quiet || true
    return 0
  fi

  echo ">> ${name}: cloning ${url} @ ${ref}"
  if [ "${recurse}" = "1" ]; then
    git clone --recurse-submodules --quiet "${url}" "${dest}"
  else
    git clone --quiet "${url}" "${dest}"
  fi
  git -C "${dest}" checkout --quiet "${ref}" || echo "   (could not checkout ${ref}; using default branch)"
  [ "${recurse}" = "1" ] && git -C "${dest}" submodule update --init --recursive --quiet || true
}

for entry in "${LIBS[@]}"; do
  IFS='|' read -r name url ref recurse <<< "${entry}"
  if want "${name}" "$@"; then
    clone_one "${name}" "${url}" "${ref}" "${recurse}"
  fi
done

echo "Done. Next: scripts/build_baselines.sh"
