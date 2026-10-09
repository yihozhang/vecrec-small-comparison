# SeqAn3 is header-only and requires C++23. sdsl-lite is bundled under
# include/seqan3/contrib; cereal/zlib/bzip2 are optional and unused by these
# drivers, so a plain include path is enough (no find_package / CPM fetch).
set(SEQAN3_INCLUDE "${THIRD_PARTY}/seqan3/include")

foreach(algo sw nw)
  add_executable(seqan3_${algo} ${algo}.cpp)
  target_link_libraries(seqan3_${algo} PRIVATE baselines_common)
  target_include_directories(seqan3_${algo} PRIVATE "${SEQAN3_INCLUDE}")
  target_compile_features(seqan3_${algo} PRIVATE cxx_std_23)
  # clang + GCC libstdc++ 13 needs explicit sized-deallocation to resolve
  # ::operator delete(void*, size_t) used by seqan3::aligned_allocator.
  target_compile_options(seqan3_${algo} PRIVATE -fsized-deallocation)
endforeach()
