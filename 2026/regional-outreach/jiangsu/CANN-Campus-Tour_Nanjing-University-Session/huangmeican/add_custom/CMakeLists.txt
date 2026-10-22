cmake_minimum_required(VERSION 3.16)
project(add_custom)

# Ascend C 算子标准 CMake 工程：依赖 CANN 包自带的模板
set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest CACHE STRING "CANN package path")
endif()

# 使用 CANN 提供的 cmake 模块
list(APPEND CMAKE_MODULE_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler/cmake)
include(cmake/ascend)
# 或者直接引入官方 samples 中的模板：
# include(${ASCEND_CANN_PACKAGE_PATH}/compiler/cmake/ascend)

add_kernel_executable(add_custom_npu
    SOURCES
        add_custom.cpp
        main.cpp
)
