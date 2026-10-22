# add0123 算子工程顶层构建 (参照 CANN msopgen FrameworkLaunch 工程, 简化版)
cmake_minimum_required(VERSION 3.16.0)
project(add0123)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN 包路径, 可通过 -DASCEND_CANN_PACKAGE_PATH=xxx 覆盖
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_CANN_PACKAGE_PATH} CACHE PATH "CANN package path")

# SoC 型号, 可通过 -DSOC_VERSION=xxx 覆盖 (如 Ascend910B1 / Ascend910B3)
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION Ascend910B1)
endif()
set(SOC_VERSION ${SOC_VERSION} CACHE STRING "SoC version")

message(STATUS "ASCEND_CANN_PACKAGE_PATH = ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "SOC_VERSION = ${SOC_VERSION}")

add_subdirectory(op_host)
add_subdirectory(op_kernel)
