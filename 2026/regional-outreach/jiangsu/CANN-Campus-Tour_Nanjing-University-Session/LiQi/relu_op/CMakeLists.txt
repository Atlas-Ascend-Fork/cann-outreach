cmake_minimum_required(VERSION 3.13)
project(relu_op CXX)

# CANN toolkit path (set via environment or -DCANN_HOME)
if(NOT DEFINED CANN_HOME)
    set(CANN_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AscendC include and library paths
set(AICORE_PATH ${CANN_HOME}/aicpu)
include_directories(
    ${AICORE_PATH}/include/kernel_operator/
    ${AICORE_PATH}/include/kernel_operator/tbe/
)

# Tiling library (host-side)
find_library(TILING_LIB
    NAMES op_api
    PATHS ${AICORE_PATH}/lib64
)

find_library(ACL_LIB
    NAMES acl_opencv
    PATHS ${CANN_HOME}/lib64
)

# Kernel (device-side, compiled with nvcc-like aicore compiler)
add_library(relu_kernel OBJECT
    src/relu_op_kernel.cpp
)
set_target_properties(relu_kernel PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(relu_kernel PRIVATE
    -O2
    --std=c++14
)

# Tiling (host-side)
add_library(relu_tiling STATIC
    src/relu_op_tiling.cpp
)
target_include_directories(relu_tiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/
    ${AICORE_PATH}/include/
)
target_link_libraries(relu_tiling PRIVATE ${TILING_LIB})

# Combined library for registration
add_library(relu_op SHARED
    $<TARGET_OBJECTS:relu_kernel>
    src/relu_op_tiling.cpp
)
target_link_libraries(relu_op PRIVATE
    ${TILING_LIB}
    ${ACL_LIB}
    op_common_utils
)

# Install
install(TARGETS relu_op DESTINATION lib)
install(FILES src/relu_op_tiling.h DESTINATION include)
