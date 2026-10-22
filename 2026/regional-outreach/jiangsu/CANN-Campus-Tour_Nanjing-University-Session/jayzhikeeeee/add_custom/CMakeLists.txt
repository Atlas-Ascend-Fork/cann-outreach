# add_custom 算子工程 CMakeLists
# 用法：bash build.sh [-v Ascend910B1]
cmake_minimum_required(VERSION 3.16)
project(add_custom)

set(CMAKE_CXX_STANDARD 17)
set(RUN_MODE "npu" CACHE STRING "run mode: npu / cpu")
set(SOC_VERSION "Ascend910B1" CACHE STRING "Ascend SoC version")

if(RUN_MODE STREQUAL "cpu")
    # CPU 调试模式（ASCENDC_CPU_DEBUG）：用 gcc 直接编译
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DASCENDC_CPU_DEBUG")
    add_executable(add_custom_npu
        add_custom.cpp
        op_host/add_custom_host.cpp)
else()
    # NPU 模式：使用 CANN 的 ai_cmake 编译核函数
    set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN package path")
    list(APPEND CMAKE_MODULE_PATH ${CANN_ROOT}/compiler/cmake)
    find_package(cann)
    if(NOT cann_FOUND)
        message(FATAL_ERROR "cann package not found, please check ASCEND_HOME_PATH")
    endif()

    ascendc_library(add_custom_kernels STATIC
        add_custom.cpp)

    # 编译宿主侧算子库（用于 PyTorch/框架接入或静态编译）
    add_library(add_custom_host SHARED
        op_host/add_custom_host.cpp)
    target_include_directories(add_custom_host PRIVATE
        ${CANN_ROOT}/include
        op_host)
endif()
