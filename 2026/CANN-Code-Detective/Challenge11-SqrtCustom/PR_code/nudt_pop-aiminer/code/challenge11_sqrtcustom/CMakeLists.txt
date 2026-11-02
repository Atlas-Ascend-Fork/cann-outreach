cmake_minimum_required(VERSION 3.16.0)
project(sqrt_op_prj)
find_package(ASC REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This example is registered for Ascend910B in op_host/sqrt_def.cpp.
if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ascend910b)
endif()
if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    message(FATAL_ERROR "Sqrt supports ascend910b only; unsupported ASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT}")
endif()
set(package_name sqrt_custom)

 npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
    add_subdirectory(op_host)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
    add_subdirectory(op_kernel)
endif()

message(WARNING "cmake 'make' does NOT build kernel binary by default. Use: bash build.sh --soc=<soc>")
