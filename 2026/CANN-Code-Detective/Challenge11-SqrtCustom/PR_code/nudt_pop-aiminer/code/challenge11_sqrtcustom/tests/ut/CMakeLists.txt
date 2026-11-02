cmake_minimum_required(VERSION 3.16.0)
project(sqrt_ut CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/CannHostLayout.cmake")
sqrt_resolve_cann_host_paths("$ENV{ASCEND_HOME_PATH}" CANN_UT_HOST_PATH CANN_UT_LIB_PATH)
message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")
message(STATUS "CANN UT host headers: ${CANN_UT_HOST_PATH}")
message(STATUS "CANN UT libraries: ${CANN_UT_LIB_PATH}")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildGoogleTest.cmake)

# DTYPE 宏定义（根据算子原型和测试用例生成）
add_definitions(
    -DDTYPE_X=float
    -DDTYPE_Y=float
)

set(UT_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(UT_COMMON_SRCS
    ${UT_COMMON_DIR}/infershape_case_executor.cpp
    ${UT_COMMON_DIR}/infershape_context_faker.cpp
    ${UT_COMMON_DIR}/tiling_case_executor.cpp
    ${UT_COMMON_DIR}/tiling_context_faker.cpp
)

set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    ${CANN_UT_HOST_PATH}/include
    ${CANN_UT_HOST_PATH}/include/base
    ${CANN_UT_HOST_PATH}/include/base/context_builder
    ${CANN_UT_HOST_PATH}/asc/include
    ${CANN_UT_HOST_PATH}/asc/include/tiling
    ${CANN_UT_HOST_PATH}/pkg_inc
    ${CANN_UT_HOST_PATH}/pkg_inc/op_common
    ${CANN_UT_HOST_PATH}/pkg_inc/base
    ${CANN_UT_HOST_PATH}/pkg_inc/exe_graph
    ${CANN_UT_HOST_PATH}/pkg_inc/graph
)

set(UT_COMMON_LIB_DIRS
    ${CANN_UT_LIB_PATH}
    ${CANN_UT_HOST_PATH}/lib64
)
list(REMOVE_DUPLICATES UT_COMMON_LIB_DIRS)

add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -fno-access-control
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

add_subdirectory(op_host)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../op_api")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../op_api" op_api)
endif()
add_subdirectory(op_kernel)
