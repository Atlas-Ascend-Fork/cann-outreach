cmake_minimum_required(VERSION 3.16.0)
project(leakyrelu_op_prj)
find_package(ASC REQUIRED)
if(NOT DEFINED CANNJUDGE_COMPILE_JOBS)
    set(CANNJUDGE_COMPILE_JOBS 4)
endif()
set(CANN_HOST_ARCH "${CMAKE_SYSTEM_PROCESSOR}-linux")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ascend910b)
endif()
string(TOLOWER "${ASCEND_COMPUTE_UNIT}" ASCEND_COMPUTE_UNIT)
if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    message(FATAL_ERROR "Leakyrelu is registered for ascend910b, got: ${ASCEND_COMPUTE_UNIT}")
endif()
set(package_name leakyrelu_custom)

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


