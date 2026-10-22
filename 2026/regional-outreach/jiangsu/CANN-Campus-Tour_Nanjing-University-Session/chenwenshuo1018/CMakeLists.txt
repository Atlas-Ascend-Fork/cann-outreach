cmake_minimum_required(VERSION 3.16)
project(inplace_rsqrt CCE)

# 用法: 
#   cmake -B build -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
#   cmake --build build
if(NOT ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()

file(GLOB KERNEL_SRC *.cpp)

add_library(inplace_rsqrt SHARED ${KERNEL_SRC})
target_include_directories(inplace_rsqrt PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)
set_target_properties(inplace_rsqrt PROPERTIES
    LINKER_LANGUAGE CXX
)
target_compile_definitions(inplace_rsqrt PRIVATE __CCE_AICORE__=220)
