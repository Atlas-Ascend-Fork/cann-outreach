cmake_minimum_required(VERSION 3.16)
project(add_custom CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 获取当前目录
set(PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(ASCENDC_CPU_DEBUG)
    # ---------------- CPU 侧调试模式 ----------------
    add_executable(add_custom
        main.cpp
        add_custom.cpp
    )
    target_compile_definitions(add_custom PRIVATE ASCENDC_CPU_DEBUG)
    target_include_directories(add_custom PRIVATE
        ${CMAKE_CXX_COMPILER_INTERFACE}
        $ENV{ASCEND_HOME_PATH}/include
        $ENV{ASCEND_HOME_PATH}/include/profiler
        $ENV{ASCEND_HOME_PATH}/include/msprof
    )
    target_link_directories(add_custom PRIVATE
        $ENV{ASCEND_HOME_PATH}/lib64
        $ENV{ASCEND_HOME_PATH}/runtime/lib64/stub
    )
    target_link_libraries(add_custom PRIVATE
        ascendcl
        c_sec
        tiling_stub
        kernel_static
    )
else()
    # ---------------- 昇腾设备侧编译 ----------------
    # 使用bisheng编译aocore（kernel编译）
    add_custom_command(
        OUTPUT ${PROJECT_DIR}/build/add_custom_npu
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_DIR}/build
        COMMAND bash -c "cd ${PROJECT_DIR} && \
            $ENV{ASCEND_HOME_PATH}/compiler/ccec_compiler/bin/ascendc_kernel_compiler \
            --ccec-compile-options='-O2' \
            --op-name=add_custom \
            --op-impl=${PROJECT_DIR}/add_custom.cpp \
            --op-impl-type=tbe \
            --kernel-name=add_custom \
            --output=${PROJECT_DIR}/build/add_custom_npu \
            --so \
            --aicore-type=AI-Core \
            --aicore-num=8 \
            --core-type=MIX"
        DEPENDS ${PROJECT_DIR}/add_custom.cpp
        COMMENT "Compiling add_custom kernel for NPU..."
    )
    add_custom_target(kernel_build ALL
        DEPENDS ${PROJECT_DIR}/build/add_custom_npu
    )

    # host侧执行程序
    add_executable(add_custom_host
        main.cpp
    )
    target_include_directories(add_custom_host PRIVATE
        $ENV{ASCEND_HOME_PATH}/include
        $ENV{ASCEND_HOME_PATH}/include/profiler
        $ENV{ASCEND_HOME_PATH}/include/msprof
    )
    target_link_directories(add_custom_host PRIVATE
        $ENV{ASCEND_HOME_PATH}/lib64
        $ENV{ASCEND_HOME_PATH}/runtime/lib64/stub
    )
    target_link_libraries(add_custom_host PRIVATE
        ascendcl
        c_sec
    )
    add_dependencies(add_custom_host kernel_build)
endif()
