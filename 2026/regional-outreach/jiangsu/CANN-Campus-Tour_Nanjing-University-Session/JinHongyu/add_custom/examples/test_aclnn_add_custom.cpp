/*!
 * \file test_aclnn_add_custom.cpp
 * \brief add_custom 算子 aclnn 调用示例（NPU 真机，float16 / ND / 二维）
 *
 * 流程：构造 x/y -> aclnnAddCustomGetWorkspaceSize -> aclnnAddCustom -> 回读 z
 *       -> host 侧按 float16 容差校验 z = x + y
 * 前置：已执行 bash build.sh 并安装算子包（opp/vendors/add_custom）
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_add_custom.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

static uint16_t FloatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (uint32_t)(exp << 10) | mant);
}

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int32_t e = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3ff;
            bits = sign | ((uint32_t)e << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

int CreateAclTensor(const std::vector<uint16_t>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclTensor** tensor)
{
    int64_t elemCount = GetShapeSize(shape);
    int64_t size = elemCount * sizeof(uint16_t);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), aclDataType::ACL_FLOAT16, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 形状 [8, 2048]（对齐）；如需测非对齐形状，可改成 {3, 100} 等
    std::vector<int64_t> shape = {8, 2048};
    int64_t n = GetShapeSize(shape);

    // 生成输入（float32 值域 [0,32)，转 float16）
    std::vector<uint16_t> xHost(n), yHost(n);
    std::vector<float> xF32(n), yF32(n);
    for (int64_t i = 0; i < n; i++) {
        xF32[i] = (float)(i % 1000) / 32.0f;
        yF32[i] = (float)(i % 77) / 8.0f;
        xHost[i] = FloatToHalf(xF32[i]);
        yHost[i] = FloatToHalf(yF32[i]);
    }
    std::vector<uint16_t> zHost(n, 0);

    aclTensor* x = nullptr;
    void* xDeviceAddr = nullptr;
    ret = CreateAclTensor(xHost, shape, &xDeviceAddr, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    ret = CreateAclTensor(yHost, shape, &yDeviceAddr, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* z = nullptr;
    void* zDeviceAddr = nullptr;
    ret = CreateAclTensor(zHost, shape, &zDeviceAddr, &z);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 第一段接口：workspace 大小 + executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddCustomGetWorkspaceSize(x, y, z, &workspaceSize, &executor);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnAddCustomGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 第二段接口：执行
    ret = aclnnAddCustom(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnAddCustom failed. ERROR: %d\n", ret); return ret);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 回读 z 并校验 z = x + y（float16 容差 10 * 2^-10）
    ret = aclrtMemcpy(zHost.data(), (uint32_t)(n * sizeof(uint16_t)), zDeviceAddr,
                      (uint32_t)(n * sizeof(uint16_t)), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return ret);

    double mare = 0.0, mereSum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double o = (double)HalfToFloat(zHost[i]);
        double g = (double)xF32[i] + (double)yF32[i];
        double diff = fabs(o - g);
        double denom = fabs(g);
        if (denom < 1e-6) denom = 1e-6;
        double rel = diff / denom;
        if (rel > mare) mare = rel;
        mereSum += rel;
    }
    double mere = mereSum / (double)n;
    const double thr = pow(2.0, -10);
    bool pass = (mere < thr) && (mare < 10.0 * thr);
    LOG_PRINT("check z = x + y: MERE=%.6f MARE=%.6f (pass if MERE<%.6f && MARE<%.6f) => %s\n",
              mere, mare, thr, 10.0 * thr, pass ? "PASS" : "FAIL");

    // 释放资源
    aclDestroyTensor(x);
    aclrtFree(xDeviceAddr);
    aclDestroyTensor(y);
    aclrtFree(yDeviceAddr);
    aclDestroyTensor(z);
    aclrtFree(zDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    // 注：与集训营模板示例一致，不显式销毁 executor（进程退出时自动回收）

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return pass ? 0 : 1;
}
