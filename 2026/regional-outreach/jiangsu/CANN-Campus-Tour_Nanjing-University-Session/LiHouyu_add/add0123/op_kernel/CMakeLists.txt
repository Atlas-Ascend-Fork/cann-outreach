# op_kernel 侧: 通过 CANN 提供的 ascendc cmake 模块编译核函数,
# 同时生成主机侧拉起所需产物
find_package(ascendc REQUIRED PATHS ${ASCEND_CANN_PACKAGE_PATH}/compiler)

ascendc_library(add0123_kernels SHARED add0123.cpp)

# kernel 需要 include op_host 下的 tiling 头文件
target_include_directories(add0123_kernels PRIVATE ${CMAKE_SOURCE_DIR}/op_host)

install(TARGETS add0123_kernels
        LIBRARY DESTINATION op_impl/ai_core/tbe/op_tiling/lib)
