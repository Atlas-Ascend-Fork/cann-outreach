cmake_minimum_required(VERSION 3.16)
project(add_custom)

set(CMAKE_CXX_STANDARD 17)

# CANN 包路径（按实际安装位置修改）
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")

# ---------- NPU 模式：编译为核函数 .o（需用 CANN 的 cce 编译器，见 build.sh） ----------
# CMake 此处仅作为工程骨架；实际编译命令见 build.sh

# ---------- CPU 调试模式（-DASCEND_CPU_DEBUG，x86 上直接跑核函数） ----------
if(DEFINED ASCEND_CPU_DEBUG)
    add_executable(add_custom_cpu
        add_custom.cpp
        main_cpu.cpp
    )
    target_compile_definitions(add_custom_cpu PRIVATE ASCEND_CPU_DEBUG)
    target_include_directories(add_custom_cpu PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include
        ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/aicpu
        ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/profiling
    )
    target_link_directories(add_custom_cpu PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64
    )
    target_link_libraries(add_custom_cpu PRIVATE
        ascendcl acl_cblas c_sec slog mmpa
    )
endif()
