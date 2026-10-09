add_executable(nw_dilated nw_dilated.cpp)
add_executable(nw_dilated_v2 nw_dilated_v2.cpp)
target_link_libraries(nw_dilated_v2 PRIVATE baselines_common)
target_include_directories(nw_dilated_v2 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../traceback"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../nitty_gritty_comparison/handwritten"
)

add_executable(nw_dilated_v3 nw_dilated_v3.cpp)
target_link_libraries(nw_dilated_v3 PRIVATE baselines_common)
target_include_directories(nw_dilated_v3 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../traceback"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../nitty_gritty_comparison/handwritten"
)
target_link_libraries(nw_dilated PRIVATE baselines_common)
target_include_directories(nw_dilated PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../traceback"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../nitty_gritty_comparison/handwritten"
)
