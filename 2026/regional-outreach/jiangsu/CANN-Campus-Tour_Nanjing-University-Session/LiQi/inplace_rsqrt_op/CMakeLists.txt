cmake_minimum_required(VERSION 3.13)
project(inplace_rsqrt_op CXX)

# CANN toolkit path
if(NOT DEFINED CANN_HOME)
    set(CANN_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AscendC include paths
set(AICORE_PATH ${CANN_HOME}/aicpu)
include_directories(
    ${AICORE_PATH}/include/kernel_operator/
    ${AICORE_PATH}/include/kernel_operator/tbe/
)

# Libraries
find_library(TILING_LIB
    NAMES op_api
    PATHS ${AICORE_PATH}/lib64
)

find_library(ACL_LIB
    NAMES acl_opencv
    PATHS ${CANN_HOME}/lib64
)

# Kernel (device-side)
add_library(inplace_rsqrt_kernel OBJECT
    src/inplace_rsqrt_kernel.cpp
)
set_target_properties(inplace_rsqrt_kernel PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(inplace_rsqrt_kernel PRIVATE
    -O2
    --std=c++14
)

# Tiling (host-side)
add_library(inplace_rsqrt_tiling STATIC
    src/inplace_rsqrt_tiling.cpp
)
target_include_directories(inplace_rsqrt_tiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/
    ${AICORE_PATH}/include/
)
target_link_libraries(inplace_rsqrt_tiling PRIVATE ${TILING_LIB})

# Combined shared library
add_library(inplace_rsqrt_op SHARED
    $<TARGET_OBJECTS:inplace_rsqrt_kernel>
    src/inplace_rsqrt_tiling.cpp
)
target_link_libraries(inplace_rsqrt_op PRIVATE
    ${TILING_LIB}
    ${ACL_LIB}
    op_common_utils
)

# Install
install(TARGETS inplace_rsqrt_op DESTINATION lib)
install(FILES src/inplace_rsqrt_tiling.h DESTINATION include)
