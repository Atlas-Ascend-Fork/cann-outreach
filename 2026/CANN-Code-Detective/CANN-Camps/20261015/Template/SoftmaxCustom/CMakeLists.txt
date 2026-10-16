cmake_minimum_required(VERSION 3.16)

# 本题基于 RegBase（reg 矢量编程），仅支持 dav-3510（Ascend 950PR / Ascend 950DT）
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU ARCH, e.g. dav-3510")
set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "运行模式: npu(默认) / sim(NPU仿真)")

find_package(ASC REQUIRED)
project(kernel_samples LANGUAGES ASC CXX)

add_executable(softmax_custom
    softmax_custom.asc
)

# host 侧 golden 计算用到 std::exp / std::max，需要链接 libm
target_link_libraries(softmax_custom PRIVATE m)

# 显式下发 NPU 架构。
# 说明：部分 CANN 版本的 ASC cmake 包不会自动把架构传给 bisheng，
# 缺失时会报 "No valid NPU architecture or soc is specified for ASC"。
target_compile_options(softmax_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
