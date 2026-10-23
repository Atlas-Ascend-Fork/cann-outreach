cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(inplace_rsqrt_fp16 LANGUAGES ASC CXX)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Load the CANN set_env.sh so ASCEND_HOME_PATH is set")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NPU_ARCH "dav-2201" CACHE STRING "Architecture of the target Ascend NPU")

add_library(inplace_rsqrt STATIC inplace_rsqrt.asc)
target_include_directories(inplace_rsqrt PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}" "$ENV{ASCEND_HOME_PATH}/include")
target_compile_options(inplace_rsqrt PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>)

add_executable(inplace_rsqrt_demo main.cpp)
set_target_properties(inplace_rsqrt_demo PROPERTIES LINKER_LANGUAGE ASC)
target_link_directories(inplace_rsqrt_demo PRIVATE "$ENV{ASCEND_HOME_PATH}/lib64")
target_link_libraries(inplace_rsqrt_demo PRIVATE inplace_rsqrt ascendcl)
target_link_options(inplace_rsqrt_demo PRIVATE --npu-arch=${NPU_ARCH})
