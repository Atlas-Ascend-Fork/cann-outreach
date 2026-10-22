cmake_minimum_required(VERSION 3.16)
project(add_custom LANGUAGES CXX ASCENDC)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CANN toolkit 路径（可用 -DASCEND_CANN_PACKAGE_PATH=... 覆盖）
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()

# kernel 侧：Ascend C 编译（交叉编译出设备侧库，链接时自动注册 kernel）
ascendc_library(add_custom_kernels SHARED kernel/add_custom.cpp)

# host 侧：tiling/算子注册 + ACL 单算子调用 main
add_executable(main main.cpp host/add_custom.cpp)

target_include_directories(main PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(main PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64/stub
)

target_link_libraries(main PRIVATE
    add_custom_kernels
    ascendcl
    tiling_api
    register
    c_sec
    platform
    ascend_protobuf
)
