# add_custom 算子 Host 侧（op_host：Tiling + 原型注册）CMake 构建脚本（参考模板）
# 说明：以下为参考脚手架。实际工程请优先用 msOpGen 按目标 CANN 版本生成，
#       依赖 CANN 的 ge / op_def_registry / gert 等头文件与库（由 build.sh 传入 ASCEND_HOME 定位）。
cmake_minimum_required(VERSION 3.10)

project(ADD_CUSTOM_HOST LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OP_NAME "add_custom")

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(${OP_NAME} SHARED ${SRC_FILES})

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CUSTOM_OP_PATH}
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/torch/ascend/adapter/impl          # 仅 torch 适配场景需要
    ${ASCEND_HOME}/operator/tiling/include
)

link_directories(
    ${ASCEND_HOME}/lib64
    ${ASCEND_HOME}/operator/tiling/lib
)

target_link_libraries(${OP_NAME} PRIVATE
    libge_common.so
    libopdef.so
    libtiling_base.so
)

if(DEFINED OP_HOST_OUTPUT_DIR)
    set_target_properties(${OP_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${OP_HOST_OUTPUT_DIR}
        PREFIX "")
endif()
