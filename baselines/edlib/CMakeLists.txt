# edlib: single source (edlib.cpp) + header, compiled with our driver.
set(EDLIB_DIR "${THIRD_PARTY}/edlib/edlib")

add_executable(edlib_editdist editdist.cpp "${EDLIB_DIR}/src/edlib.cpp")
target_include_directories(edlib_editdist PRIVATE "${EDLIB_DIR}/include")
target_link_libraries(edlib_editdist PRIVATE baselines_common)
