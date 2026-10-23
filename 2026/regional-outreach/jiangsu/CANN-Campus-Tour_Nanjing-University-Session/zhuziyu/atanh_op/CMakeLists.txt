# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# CMakeLists.txt — build the AscendC atanh kernel into libatanh_kernel.so
#
# Requires a CANN (Ascend Toolkit / CANN community edition) install with the
# AscendC toolkit and the tikcpp / bisheng compiler on PATH.
#
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make
#
# The exact CMake module names differ across CANN releases.  Where
# find_package(AscendC) is unavailable on your version, the explicit compiler
# command at the bottom of this file shows how to build op_kernel/atanh.cpp
# directly with the CANN ccec (Ascend C) compiler.

cmake_minimum_required(VERSION 3.13)
project(atanh_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# ---------------------------------------------------------------------------
# Locate the AscendC toolkit.
# Prefer the env var ASCEND_HOME; fall back to the default install path.
# ---------------------------------------------------------------------------
if(DEFINED ENV{ASCEND_HOME})
    set(ASCEND_HOME $ENV{ASCEND_HOME})
elseif(DEFINED ENV{ASCEND_AICPU_HOME})
    set(ASCEND_HOME $ENV{ASCEND_AICPU_HOME})
else()
    set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()

message(STATUS "Using ASCEND_HOME = ${ASCEND_HOME}")

# Include dir for kernel_operator.h and the shared tiling header.
set(ATANH_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)

# ---------------------------------------------------------------------------
# Kernel library.
# The AscendC kernel is compiled with the CANN ccec compiler, which is not a
# plain clang++ target, so we drive it with an add_custom_command rather than
# the generic CXX compiler.  The flag set matches the CANN AscendC kernel
# template.
# ---------------------------------------------------------------------------
set(ATANH_KERNEL_SO ${CMAKE_CURRENT_BINARY_DIR}/libatanh_kernel.so)

add_custom_command(
    OUTPUT  ${ATANH_KERNEL_SO}
    COMMAND $ENV{CMAKE_COMMAND} -E env
            ${ASCEND_HOME}/tools/ccec/bin/ccec
            -O2
            --std=c++17
            --allow-multiple-definition
            --fast-math
            -D __CCE_KT_TEST__
            -I ${ATANH_KERNEL_DIR}
            -o ${ATANH_KERNEL_SO}
            ${ATANH_KERNEL_DIR}/atanh.cpp
    DEPENDS ${ATANH_KERNEL_DIR}/atanh.cpp
            ${ATANH_KERNEL_DIR}/atanh_tiling.h
    COMMENT "Compiling AscendC atanh kernel -> libatanh_kernel.so"
)

add_custom_target(atanh_kernel ALL
    DEPENDS ${ATANH_KERNEL_SO}
)

# ---------------------------------------------------------------------------
# Manual fallback (if the ccec path above does not match your toolkit layout):
#
#   ${ASCEND_HOME}/tools/ccec/bin/ccec \
#     -O2 --std=c++17 --allow-multiple-definition --fast-math \
#     -D __CCE_KT_TEST__ \
#     -I op_kernel \
#     -o libatanh_kernel.so \
#     op_kernel/atanh.cpp
#
# Some CANN versions ship the driver as `bisheng-ccec` or require the kernel
# to be a relocatable object (.o) consumed by a larger operator build.  Adjust
# the output suffix accordingly.
# ---------------------------------------------------------------------------
