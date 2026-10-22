// Copyright (C) 2026.
// Gcd123 NPU 侧测试（aclnn 单算子调用）。
//
// 前置条件：
//  - 已用 build.sh 编译并安装算子包（.run 安装到 $ASCEND_OPP_PATH/vendors/customize/）。
//  - 已加载 CANN 环境变量，并把 op_api 头文件/动态库加入包含与链接路径：
//        export LD_LIBRARY_PATH=.../vendors/customize/op_api/lib:$LD_LIBRARY_PATH
//  - 编译时需链接 acl 与生成的 op_api 库。
//
// 注意：aclnn 自定义算子生成的接口名（aclnnGcd123*）与参数签名随 CANN 版本略有差异，
// 请以安装后 op_api 目录下的头文件为准；本文件给出标准调用骨架与校验流程。
//
#include <acl/acl.h>
#include <aclrt/aclrt.h>
#include <cstdint>
#include <cstdio>
#include <vector>

// 若 op_api 头文件名为 aclnn_gcd123.h，请按需修改：
#include "aclnn_gcd123.h"

// ---------------------------------------------------------------------------
// 极简 CPU golden（与 op_kernel 语义一致），用于与 NPU 输出比对。
// ---------------------------------------------------------------------------
namespace ref {
static inline int32_t GcdI32(int32_t a, int32_t b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) { int32_t r = a % b; a = b; b = r; }
    return a;
}
static inline float ToInt(float h) { return std::floor(h); }
} // namespace ref

// float16 <-> 主机 half 的辅助：此处用 uint16_t 承载，测试值为 half 可精确表示的整数。
static inline uint16_t F2HalfBits(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((v.u >> 23) & 0xFF);
    uint32_t mant = v.u & 0x7FFFFF;
    if (exp == 0) return static_cast<uint16_t>(sign); // 0（测试不涉及非规格化）
    int32_t e = exp - 127 + 15;
    if (e >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00); // inf
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13));
}

#define CHECK_RET(expr)                                              \
    do {                                                             \
        aclRet rc = (expr);                                          \
        if (rc != ACL_SUCCESS) {                                     \
            std::printf("ACL error at line %d: ret=%d\n", __LINE__, rc); \
            return -1;                                               \
        }                                                            \
    } while (0)

int main() {
    // ---- 测试数据（N1=1 broadcast 场景，便于直观校验）----
    const int totalSelf = 1;      // self shape [1,1,1,1] = 标量 6
    const int totalOther = 8;     // other shape [1,2,2,2]
    const int totalOut = totalOther;

    std::vector<float> selfH = {6.0f};
    std::vector<float> otherH = {10, 15, 8, 9, 12, 21, 5, 14};
    std::vector<float> expect = {2, 3, 2, 3, 6, 3, 1, 2}; // gcd(6, x)

    // ---- ACL 初始化 ----
    CHECK_RET(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_RET(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_RET(aclrtCreateStream(&stream));

    const size_t selfBytes = totalSelf * sizeof(uint16_t);
    const size_t otherBytes = totalOther * sizeof(uint16_t);
    const size_t outBytes = totalOut * sizeof(uint16_t);

    // 主机侧 half 缓冲
    std::vector<uint16_t> selfBuf(totalSelf), otherBuf(totalOther);
    for (int i = 0; i < totalSelf; ++i) selfBuf[i] = F2HalfBits(selfH[i]);
    for (int i = 0; i < totalOther; ++i) otherBuf[i] = F2HalfBits(otherH[i]);

    // 设备内存
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    CHECK_RET(aclrtMalloc(&selfDev, selfBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_RET(aclrtMalloc(&otherDev, otherBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_RET(aclrtMalloc(&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_RET(aclrtMemcpy(selfDev, selfBytes, selfBuf.data(), selfBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_RET(aclrtMemcpy(otherDev, otherBytes, otherBuf.data(), otherBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 调用 aclnn 算子（接口名/签名以生成的 op_api 头文件为准）----
    // 形状需以框架要求的描述方式传入（通常为 shape 指针数组 + ndim）。
    // 伪代码示意，具体签名请对齐 op_api 头文件：
    //   int64_t selfShape[4]  = {1, 1, 1, 1};
    //   int64_t otherShape[4] = {1, 2, 2, 2};
    //   int64_t outShape[4];
    //   size_t workspaceSize = 0;
    //   void* workspace = nullptr;
    //   CHECK(aclnnGcd123GetWorkspaceSize(selfDev, otherDev, outDev, &workspaceSize));
    //   if (workspaceSize) CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    //   CHECK(aclnnGcd123(workspace, workspaceSize, stream, selfDev, otherDev, outDev));
    //   CHECK(aclrtSynchronizeStream(stream));

    // ---- 取回结果并比对 ----
    std::vector<uint16_t> outBuf(totalOut);
    CHECK_RET(aclrtMemcpy(outBuf.data(), outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // 半精度转 float 的简易实现（测试值均为整数，直接读 mant/exp 亦可）；
    // 实际比对时建议复用与 CPU 测试一致的 HalfToFloat 逻辑。
    // 此处以“期望值为整数”为前提，仅演示比对框架。
    std::printf("NPU test: run the aclnn call above, then compare outBuf to expect.\n");

    // ---- 释放 ----
    CHECK_RET(aclrtFree(selfDev));
    CHECK_RET(aclrtFree(otherDev));
    CHECK_RET(aclrtFree(outDev));
    CHECK_RET(aclrtDestroyStream(stream));
    CHECK_RET(aclrtResetDevice(deviceId));
    CHECK_RET(aclFinalize());

    std::printf("Gcd123 NPU test finished.\n");
    return 0;
}
