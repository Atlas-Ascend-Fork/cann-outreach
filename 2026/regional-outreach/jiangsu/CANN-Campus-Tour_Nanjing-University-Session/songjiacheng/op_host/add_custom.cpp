/**
 * add_custom 算子 host 侧实现（op_host）
 * 职责：参数校验（TORCH_CHECK）→ 通过平台 API 获取硬件参数 → 计算 Tiling →
 *       分配输出张量 → 启动核函数
 */
#include <torch/extension.h>
#include <acl/acl.h>
#include "ops.h"
#include "kernel_tiling/kernel_tiling.h"   // platformAscendC 平台 API

// 与核函数共享的 Tiling 结构（内存布局需与 op_kernel 中保持一致）
struct AddCustomTilingData {
    uint32_t totalLength;   // N2 * N1，总元素数
    uint32_t blockLength;   // 每个 core 处理的元素数
    uint32_t tileNum;       // 核内切分块数
    uint32_t tileLength;    // 每块元素数（32 对齐）
};

// Python 入口：z = add_custom(x, y)
torch::Tensor add_custom(const torch::Tensor &x, const torch::Tensor &y)
{
    // ---- 参数校验（系统边界，必须校验）----
    TORCH_CHECK(x.is_npu() && y.is_npu(), "add_custom: inputs must be NPU tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16 && y.scalar_type() == torch::kFloat16,
                "add_custom: only float16 is supported");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "add_custom: inputs must be 2-D [N2, N1]");
    TORCH_CHECK(x.sizes() == y.sizes(), "add_custom: x and y must have the same shape");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "add_custom: inputs must be contiguous");

    const uint32_t totalLength = static_cast<uint32_t>(x.numel());

    // 约束：totalLength 为 32 的整数倍（末尾无标量尾巴处理）
    TORCH_CHECK(totalLength % 32 == 0, "add_custom: total element count must be a multiple of 32, got ",
                totalLength);

    // ---- 平台 API 获取硬件参数 ----
    auto platformAscendC = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t coreNum = platformAscendC->GetCoreNumAiv();          // 可用 AI Vector Core 数
    uint64_t ubSize = 0;
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // ---- Block 级切分 ----
    uint32_t blockDim = totalLength < coreNum ? 1u : coreNum;
    const uint32_t blockLength = (totalLength + blockDim - 1) / blockDim; // 向上取整
    blockDim = (totalLength + blockLength - 1) / blockLength;             // 实际启用核数

    // ---- UB 级切分（双缓冲，6 份 tile buffer，见 design.md UB 分配表）----
    // 可用 tile 字节数 = ubSize / (BUFFER_NUM * 3 队列) ，预留 1/4 给系统
    constexpr uint32_t QUEUE_NUM = 3;   // inX, inY, outZ
    constexpr uint32_t BUFFER_NUM = 2;  // 双缓冲
    const uint64_t tileBytes = ubSize / (QUEUE_NUM * BUFFER_NUM) / 4 * 3;  // 预留 1/4
    uint32_t tileLength = static_cast<uint32_t>(tileBytes / sizeof(uint16_t));
    tileLength = std::min(tileLength, blockLength);
    tileLength = std::max(tileLength / 32 * 32, 32u);                     // 32 对齐
    const uint32_t tileNum = (blockLength + tileLength - 1) / tileLength;

    AddCustomTilingData tilingData{totalLength, blockLength, tileNum, tileLength};

    // ---- 分配输出与 tiling 设备内存 ----
    auto z = torch::empty_like(x);
    void *tilingDevice = nullptr;
    TORCH_CHECK(aclrtMalloc(&tilingDevice, sizeof(AddCustomTilingData),
                            ACL_MEM_MALLOC_HIGHEST_FIRST) == ACL_SUCCESS,
                "add_custom: alloc tiling buffer failed");
    TORCH_CHECK(aclrtMemcpy(tilingDevice, sizeof(AddCustomTilingData), &tilingData,
                            sizeof(AddCustomTilingData), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
                "add_custom: copy tiling H2D failed");

    // ---- 启动核函数（CANN 标准启动宏，链接期由 kernel 编译产物提供符号）----
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    ACLRT_LAUNCH_KERNEL(add_custom, blockDim, stream,
                        const_cast<void *>(x.data_ptr()), const_cast<void *>(y.data_ptr()),
                        z.data_ptr(), tilingDevice);

    aclrtFree(tilingDevice);
    return z;
}
