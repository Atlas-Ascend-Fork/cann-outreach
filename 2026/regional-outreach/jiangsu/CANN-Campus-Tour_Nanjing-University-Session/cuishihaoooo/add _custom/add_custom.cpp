/*
 * \file add_custom.cpp
 * \brief AddCustom —— 逐元素加法算子（Ascend C / AI Core Vector）实现
 *
 * 算子规格
 *   OpType  : AddCustom
 *   输入    : x, 2D [N2, N1], ND, float16 (half)
 *             y, 2D [N2, N1], ND, float16 (half)
 *   输出    : z, 2D [N2, N1], ND, float16 (half)
 *   功能    : z[i] = x[i] + y[i]
 *
 * 实现要点
 *   - 使用 AscendC::Add 高阶矢量指令（内部映射 vadd 向量指令）
 *   - 批量向量化：每个 tile 由一次 Add 调用处理，无标量逐元素循环
 *   - 多核并行：整张量（totalSize = N2*N1）按核均分，各核处理独立分片
 *   - 输入 x、y 只读，计算结果写入 z
 *
 * 编译运行（CANN 9.x / bisheng，dav-2201 请替换为你芯片的架构版本号）:
 *   bisheng -x asc add_custom.cpp -o add_custom --npu-arch=dav-2201
 *   ./add_custom
 */

#include "add_custom.h"

#include "acl/acl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace AscendC;

// ========================== KernelAddCustom 方法实现（Device 侧） ==========================

__aicore__ void KernelAddCustom::Init(__gm__ uint8_t *xGm, __gm__ uint8_t *yGm, __gm__ uint8_t *zGm,
                                      uint32_t totalSize)
{
    // 2D shape [N2, N1] 在 ND 格式下按行连续存放，totalSize 为两个维度的乘积
    // 将 totalSize 按核数均分，余数分给前 remainder 个核（各核分片互不重叠）
    blockNum_ = GetBlockNum();
    blockIdx_ = GetBlockIdx();
    const uint32_t base = totalSize / blockNum_;
    const uint32_t remainder = totalSize % blockNum_;
    blockLen_ = base + (blockIdx_ < remainder ? 1u : 0u);
    inOffset_ = blockIdx_ * base + (blockIdx_ < remainder ? blockIdx_ : remainder);

    // 将本核负责的输入/输出绑定到 Global Memory（元素偏移 inOffset_）
    xGlobal_.SetGlobalBuffer((__gm__ half *)(xGm) + inOffset_, blockLen_);
    yGlobal_.SetGlobalBuffer((__gm__ half *)(yGm) + inOffset_, blockLen_);
    zGlobal_.SetGlobalBuffer((__gm__ half *)(zGm) + inOffset_, blockLen_);

    // 初始化队列缓冲（单位：字节）
    // inQueue_ 深度为 2：同一时刻需同时持有 x、y 两个 tile
    pipe_.InitBuffer(inQueue_, 2, ADD_CUSTOM_TILE_LEN * sizeof(half));
    pipe_.InitBuffer(outQueue_, 1, ADD_CUSTOM_TILE_LEN * sizeof(half));
}

__aicore__ void KernelAddCustom::Process()
{
    uint32_t processed = 0;
    while (processed < blockLen_) {
        uint32_t curLen = ADD_CUSTOM_TILE_LEN;
        if (curLen > blockLen_ - processed) {
            curLen = blockLen_ - processed;  // 处理最后一个尾部 tile
        }
        CopyIn(processed, curLen);
        Compute(curLen);
        CopyOut(processed, curLen);
        processed += curLen;
    }

    PipeBarrier<PIPE_ALL>();  // 确保所有写 GM 完成后再退出 kernel
}

__aicore__ void KernelAddCustom::CopyIn(uint32_t offset, uint32_t len)
{
    // x、y 两个输入 tile 同时搬入 UB VECIN（输入只读，仅从 GM 读入）
    LocalTensor<half> xLocal = inQueue_.AllocTensor<half>();
    DataCopy(xLocal, xGlobal_[offset], len);
    inQueue_.EnQue(xLocal);

    LocalTensor<half> yLocal = inQueue_.AllocTensor<half>();
    DataCopy(yLocal, yGlobal_[offset], len);
    inQueue_.EnQue(yLocal);
}

__aicore__ void KernelAddCustom::Compute(uint32_t len)
{
    LocalTensor<half> xLocal = inQueue_.DeQue<half>();
    LocalTensor<half> yLocal = inQueue_.DeQue<half>();
    LocalTensor<half> zLocal = outQueue_.AllocTensor<half>();
    // zLocal = xLocal + yLocal，len 个元素一批向量计算（vadd），结果写入 zLocal
    Add(zLocal, xLocal, yLocal, 0);
    inQueue_.FreeTensor(xLocal);
    inQueue_.FreeTensor(yLocal);
    outQueue_.EnQue<half>(zLocal);
}

__aicore__ void KernelAddCustom::CopyOut(uint32_t offset, uint32_t len)
{
    LocalTensor<half> zLocal = outQueue_.DeQue<half>();
    DataCopy(zGlobal_[offset], zLocal, len);
    outQueue_.FreeTensor(zLocal);
}

// ========================== Kernel 入口 ==========================

__global__ __vector__ void AddCustom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z,
                                     uint32_t totalSize)
{
    AscendC::InitSocState();

    KernelAddCustom op;
    op.Init(x, y, z, totalSize);
    op.Process();
}

// ========================== Host 侧 half <-> float 转换 ==========================

static inline float half_to_float(half h)
{
    uint32_t x;
    std::memcpy(&x, &h, 2);
    const uint32_t sign = (x >> 15) & 0x1u;
    const uint32_t exp = (x >> 10) & 0x1Fu;
    const uint32_t mant = x & 0x3FFu;
    float val;
    if (exp == 0x1Fu) {
        val = mant ? std::nanf("") : std::numeric_limits<float>::infinity();
    } else if (exp == 0u) {
        val = std::ldexpf(static_cast<float>(mant), -24);  // 非规格化
    } else {
        val = std::ldexpf(static_cast<float>(mant + 0x400u), static_cast<int>(exp) - 25);
    }
    return sign ? -val : val;
}

static inline half float_to_half(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xFFu;
    const uint32_t mant = x & 0x7FFFFFu;
    uint32_t r;
    if (exp == 0xFFu) {
        r = sign | 0x7C00u | (mant ? 0x200u : 0u);  // Inf / NaN
    } else if (exp == 0u) {
        r = sign;  // 下溢按 0 处理（测试数据不会走到这里）
    } else {
        const int32_t e = static_cast<int32_t>(exp) - 112;  // exp - 127 + 15
        if (e >= 0x1F) {
            r = sign | 0x7C00u;  // 上溢按 Inf
        } else if (e <= 0) {
            r = sign;  // 下溢按 0
        } else {
            r = sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13);
        }
    }
    half hh;
    std::memcpy(&hh, &r, 2);
    return hh;
}

// ========================== Host 侧调用与验证（测试入口） ==========================

int main(int32_t /*argc*/, char ** /*argv*/)
{
    // ---- acl 初始化 ----
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::printf("[AddCustom] aclInit failed\n");
        return 1;
    }
    const int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        return 1;
    }

    // ---- 2D shape [N2, N1]，ND ----
    const int64_t N2 = 16, N1 = 128;
    const uint32_t totalSize = static_cast<uint32_t>(N2 * N1);  // 2048
    const size_t byteSize = static_cast<size_t>(totalSize) * sizeof(half);

    // ---- host 输入与 golden（x: [-2, 2], y: [1, 3]，z = x + y 在 [-1, 5] 内） ----
    std::vector<half> hX(totalSize);
    std::vector<half> hY(totalSize);
    std::vector<float> golden(totalSize);
    for (uint32_t i = 0; i < totalSize; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(totalSize);
        const float xv = -2.0f + 4.0f * t;
        const float yv = 1.0f + 2.0f * t;
        hX[i] = float_to_half(xv);
        hY[i] = float_to_half(yv);
        golden[i] = half_to_float(hX[i]) + half_to_float(hY[i]);
    }

    // ---- 设备内存 ----
    uint8_t *dX = nullptr;
    uint8_t *dY = nullptr;
    uint8_t *dZ = nullptr;
    if (aclrtMalloc(reinterpret_cast<void **>(&dX), byteSize, ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS) {
        return 1;
    }
    if (aclrtMalloc(reinterpret_cast<void **>(&dY), byteSize, ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS) {
        return 1;
    }
    if (aclrtMalloc(reinterpret_cast<void **>(&dZ), byteSize, ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS) {
        return 1;
    }
    if (aclrtMemcpy(dX, byteSize, hX.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE) !=
        ACL_SUCCESS) {
        return 1;
    }
    if (aclrtMemcpy(dY, byteSize, hY.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE) !=
        ACL_SUCCESS) {
        return 1;
    }

    // ---- 启动 kernel（8 个核并行） ----
    const uint32_t numBlocks = 8;
    AddCustom<<<numBlocks, nullptr, stream>>>(dX, dY, dZ, totalSize);

    // ---- 拷回结果 ----
    std::vector<half> hZ(totalSize);
    aclrtSynchronizeStream(stream);
    if (aclrtMemcpy(hZ.data(), byteSize, dZ, byteSize, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        return 1;
    }

    // ---- 验证（half 精度容差 2e-2） ----
    float maxErr = 0.0f;
    uint32_t badCount = 0;
    for (uint32_t i = 0; i < totalSize; ++i) {
        const float got = half_to_float(hZ[i]);
        const float diff = std::fabs(got - golden[i]);
        if (diff > maxErr) {
            maxErr = diff;
        }
        if (diff > 2e-2f) {
            ++badCount;
        }
    }
    std::printf("[AddCustom] N2=%ld N1=%ld  totalSize=%u  numBlocks=%u\n", N2, N1, totalSize,
                numBlocks);
    std::printf("[AddCustom] maxAbsErr=%.6f  badCount=%u/%u  =>  %s\n", maxErr, badCount,
                totalSize, badCount == 0 ? "PASS" : "FAIL");

    // ---- 清理 ----
    aclrtFree(dX);
    aclrtFree(dY);
    aclrtFree(dZ);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return badCount == 0 ? 0 : 1;
}
