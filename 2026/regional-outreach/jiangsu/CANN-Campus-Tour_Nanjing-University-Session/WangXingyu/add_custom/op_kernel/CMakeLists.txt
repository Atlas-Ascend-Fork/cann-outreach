# add_custom 算子核函数（op_kernel）CMake 构建脚本（参考模板）
# 说明：以下为参考脚手架。实际工程请优先用 msOpGen 按目标 CANN 版本生成 op_kernel/CMakeLists.txt，
#       其中 -DASCEND_HOME / -DASCEND_CUSTOM_OP_PATH 等由 build.sh 传入。
cmake_minimum_required(VERSION 3.10)

project(ADD_CUSTOM_KERNEL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 算子名称（编译产物为 libadd_custom.so / add_custom.so）
set(OP_NAME "add_custom")

# 收集核函数源文件
file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(${OP_NAME} SHARED ${SRC_FILES})

# Ascend C 头文件 / 库路径（由 build.sh 传入 ASCEND_HOME 定位）
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_HOME}/ascendc/include
    ${ASCEND_HOME}/include
)

link_directories(
    ${ASCEND_HOME}/ascendc/lib64
    ${ASCEND_HOME}/lib64
)

# 链接 Ascend C 运行时
target_link_libraries(${OP_NAME} PRIVATE libascendcl.so)

# 输出到统一 build 目录（由 build.sh 指定）
if(DEFINED OP_KERNEL_OUTPUT_DIR)
    set_target_properties(${OP_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${OP_KERNEL_OUTPUT_DIR}
        PREFIX "")
endif()
