# add_custom 算子编译配置（基于 CANN 样例工程结构）
cmake_minimum_required(VERSION 3.16)
project(add_custom LANGUAGES CXX CCE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN 包路径（按实际环境调整，或在命令行 -DCANN_ROOT=... 传入）
if(NOT DEFINED CANN_ROOT)
    set(CANN_ROOT "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(ASCEND_CANN_PACKAGE_PATH ${CANN_ROOT})

# 源文件
file(GLOB KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 添加算子（生成 add_custom_npu 目标，面向昇腾 AI Core 编译）
add_npu_target(add_custom_npu ${KERNEL_SRC} "")

install(TARGETS add_custom_npu
        LIBRARY DESTINATION lib)
