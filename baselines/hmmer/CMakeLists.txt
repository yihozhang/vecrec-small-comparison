# Profile-HMM Viterbi driver linking the vendored HMMER 3.4 static libraries
# (third_party/hmmer-3.4, built by scripts/build_hmmer.sh).
#
# HMMER's NEON/SSE headers use the `register` storage class, which C++17 removed,
# so this one target compiles as C++14 (where it is merely deprecated) instead of
# the suite-wide C++20. It therefore does NOT link `baselines_common` (whose
# cxx_std_20 INTERFACE feature would force C++17+); it pulls in the same
# common/ include dir and optimization flags directly.
set(HMMER_DIR "${THIRD_PARTY}/hmmer-3.4")

add_executable(hmmer_viterbi viterbi.cpp)
target_include_directories(hmmer_viterbi PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../common"
  "${HMMER_DIR}/src"
  "${HMMER_DIR}/easel")
set_target_properties(hmmer_viterbi PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)
target_compile_options(hmmer_viterbi PRIVATE -O3 -Wno-write-strings -Wno-deprecated-register)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  check_cxx_compiler_flag("-mcpu=native" HMMER_HAS_MCPU_NATIVE)
  if(HMMER_HAS_MCPU_NATIVE)
    target_compile_options(hmmer_viterbi PRIVATE -mcpu=native)
  endif()
else()
  target_compile_options(hmmer_viterbi PRIVATE -march=native)
endif()

target_link_libraries(hmmer_viterbi PRIVATE
  "${HMMER_DIR}/src/libhmmer.a"
  "${HMMER_DIR}/easel/libeasel.a"
  pthread)
find_library(M_LIB m)
if(M_LIB)
  target_link_libraries(hmmer_viterbi PRIVATE ${M_LIB})
endif()
