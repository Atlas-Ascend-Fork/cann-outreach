# add_custom 算子构建脚本
# 用法（在 CANN 环境下）：
#   bash build.sh   或按 CMake 流程：
#   mkdir build && cd build
#   cmake .. && make

cmake_minimum_required(VERSION 3.16)
project(add_custom C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN 安装路径（可通过 -DASCEND_CANN_PACKAGE_PATH=xxx 指定）
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()

list(APPEND CMAKE_MODULE_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler/cmake)
find_package(cann_kernel REQUIRED)
find_package(cann_host REQUIRED)

# kernel 侧：编译 add_custom.cpp 生成 NPU 二进制
add_npu_kernel(add_custom add_custom.cpp)

# host 侧：编译 main.cpp 并链接 kernel
add_executable(add_custom_host main.cpp)
target_link_libraries(add_custom_host PRIVATE
    ${ASCEND_KERNEL_NAME_add_custom}
    ascendcl
)
