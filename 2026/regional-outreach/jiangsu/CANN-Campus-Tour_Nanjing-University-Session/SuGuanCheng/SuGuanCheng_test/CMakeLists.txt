# add_custom 算子编译脚本
# 用法：在安装了 CANN Toolkit 的环境下执行
#   source ${ASCEND_HOME}/set_env.sh
#   mkdir build && cd build
#   cmake .. && make
cmake_minimum_required(VERSION 3.16)
project(add_custom LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

message(STATUS "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}")

# ---------- Device 侧：使用 bisheng / aic 编译核函数为 .o ----------
set(KERNEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/add_custom.cpp)

add_custom_command(
    OUTPUT add_custom_npu.o
    COMMAND ${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin/aic
            ${KERNEL_SOURCES}
            -o add_custom_npu.o
            --soc_version=<SOC_VERSION>     # 按实际芯片修改，如 Ascend910B1 / Ascend310B1
            -I${ASCEND_HOME_PATH}/include
    DEPENDS ${KERNEL_SOURCES}
    COMMENT "Compiling add_custom kernel (Ascend C)"
)

add_custom_target(compile_kernel ALL DEPENDS add_custom_npu.o)

# ---------- Host 侧：编译调用封装 ----------
find_path(ACL_INCLUDE acl/acl.h PATHS ${ASCEND_HOME_PATH}/include PATH_SUFFIXES aarch64-linux x86_64-linux)
find_library(ACL_LIBRARY acl_dvpp acl PATHS ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/aarch64-linux/lib64)

add_library(add_custom_host SHARED add_custom_host.cpp)
target_include_directories(add_custom_host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ACL_INCLUDE}
)
target_link_libraries(add_custom_host PRIVATE ${ACL_LIBRARY})
