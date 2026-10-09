cmake_minimum_required(VERSION 3.20)
project(vecrec_baselines CXX C)

# All driver binaries land here; scripts/build_baselines.sh copies them to ../bin.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Cloned C++ libraries live here (gitignored, populated by setup_baselines.sh).
set(THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/../third_party")

# Shared contract: include path for driver_io.hpp + portable optimization flags.
add_library(baselines_common INTERFACE)
target_include_directories(baselines_common INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/common")
target_compile_features(baselines_common INTERFACE cxx_std_20)

include(CheckCXXCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  # Apple/ARM clang rejects -march=native; prefer -mcpu=native when available.
  check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
  if(HAS_MCPU_NATIVE)
    target_compile_options(baselines_common INTERFACE -O3 -mcpu=native)
  else()
    target_compile_options(baselines_common INTERFACE -O3)
  endif()
else()
  target_compile_options(baselines_common INTERFACE -O3 -march=native)
endif()

# Build a driver only when both its clone and its CMakeLists are present, so a
# partial setup (or a not-yet-written driver) never blocks the rest.
foreach(lib seqan3 parasail ksw2 ssw edlib wfa2)
  if(EXISTS "${THIRD_PARTY}/${lib}" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${lib}/CMakeLists.txt")
    message(STATUS "baselines: enabling ${lib}")
    add_subdirectory(${lib})
  else()
    message(STATUS "baselines: skipping ${lib} (clone or driver missing)")
  endif()
endforeach()

# Hand-written dilated NW is NEON-only.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  add_subdirectory(handwritten_nw)
else()
  message(STATUS "baselines: skipping handwritten_nw (ARM/NEON only)")
endif()

# HMMER lives under hmmer-3.4/ and is built into static libs by scripts/build_hmmer.sh;
# enable the profile-HMM Viterbi driver only once those libs exist.
if(EXISTS "${THIRD_PARTY}/hmmer-3.4/src/libhmmer.a" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hmmer/CMakeLists.txt")
  message(STATUS "baselines: enabling hmmer")
  add_subdirectory(hmmer)
else()
  message(STATUS "baselines: skipping hmmer (run scripts/build_hmmer.sh first)")
endif()
