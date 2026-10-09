# Build parasail as a static library (no OpenMP to avoid a libomp dependency on
# macOS clang), then link our driver against it.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(WITH_OPENMP OFF CACHE BOOL "" FORCE)

add_subdirectory("${THIRD_PARTY}/parasail" "${CMAKE_CURRENT_BINARY_DIR}/parasail_lib" EXCLUDE_FROM_ALL)

add_executable(parasail_align align.cpp)
target_link_libraries(parasail_align PRIVATE baselines_common parasail)
target_include_directories(parasail_align PRIVATE "${THIRD_PARTY}/parasail")
