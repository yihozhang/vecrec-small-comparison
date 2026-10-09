# SSW: compile the C core (ssw.c) + C++ wrapper (ssw_cpp.cpp) + our driver.
# ssw.c includes sse2neon.h automatically when __ARM_NEON is defined.
set(SSW_SRC "${THIRD_PARTY}/ssw/src")

add_executable(ssw_sw sw.cpp "${SSW_SRC}/ssw.c" "${SSW_SRC}/ssw_cpp.cpp")
target_include_directories(ssw_sw PRIVATE "${SSW_SRC}")
target_link_libraries(ssw_sw PRIVATE baselines_common)
