#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "acl/acl.h"
#include "aclnn_abs.h"

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

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto elemCount = GetShapeSize(shape);
    int64_t elemSize = sizeof(T);
    switch (dataType) {
        case aclDataType::ACL_FLOAT16:
        case aclDataType::ACL_BF16:
        case aclDataType::ACL_INT16:
        case aclDataType::ACL_UINT16:
            elemSize = 2;
            break;
        case aclDataType::ACL_INT8:
        case aclDataType::ACL_UINT8:
        case aclDataType::ACL_BOOL:
            elemSize = 1;
            break;
        case aclDataType::ACL_INT64:
        case aclDataType::ACL_UINT64:
        case aclDataType::ACL_DOUBLE:
            elemSize = 8;
            break;
        default:
            break;
    }
    auto size = elemCount * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<uint8_t> convBuf(size);
    if (dataType == aclDataType::ACL_FLOAT16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t h = FloatToHalf(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &h, 2);
        }
    } else if (dataType == aclDataType::ACL_BF16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t b = FloatToBFloat16(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &b, 2);
        }
    } else if (dataType == aclDataType::ACL_DOUBLE) {
        for (int64_t i = 0; i < elemCount; i++) {
            double d = static_cast<double>(hostData[i]);
            memcpy(convBuf.data() + i * 8, &d, 8);
        }
    } else {
        auto copySize = std::min((int64_t)(elemCount * sizeof(T)), size);
        memcpy(convBuf.data(), hostData.data(), copySize);
    }
    ret = aclrtMemcpy(*deviceAddr, size, convBuf.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// 生成确定性输入：覆盖 正/负/0/极小负数/极大正数/典型负数，保证能验证 abs 语义
static void FillInputHost(std::vector<float>& x)
{
    for (size_t i = 0; i < x.size(); ++i) {
        switch (i % 6) {
            case 0: x[i] =  static_cast<float>(i + 1); break;
            case 1: x[i] = -static_cast<float>(i + 1); break;
            case 2: x[i] =  0.0f;                       break;
            case 3: x[i] = -1.0e-30f;                   break;
            case 4: x[i] =  1.0e+30f;                   break;
            case 5: x[i] = -3.1415926f;                 break;
        }
    }
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 构造输入 tensor：含负数/0/极值，确保 abs 语义被覆盖
    aclTensor* x = nullptr;
    void* xDeviceAddr = nullptr;
    std::vector<int64_t> xShape = {8, 2048};
    std::vector<float> xHostData(GetShapeSize(xShape));
    FillInputHost(xHostData);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造输出 tensor
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    std::vector<int64_t> yShape = {8, 2048};
    std::vector<float> yHostData(GetShapeSize(yShape), 0.0f);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用 aclnnAbsGetWorkspaceSize 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAbsGetWorkspaceSize(x, y, &workspaceSize, &executor);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnAbsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnAbs 第二段接口
    ret = aclnnAbs(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnAbs failed. ERROR: %d\n", ret); return ret);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // ★ 回读输出并逐元素校验
    const int64_t elemCount = GetShapeSize(yShape);
    const size_t yBytes = static_cast<size_t>(elemCount) * sizeof(float);
    std::vector<float> yOut(elemCount, 0.0f);
    ret = aclrtMemcpy(yOut.data(), yBytes, yDeviceAddr, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return ret);

    int64_t mismatch = 0;
    constexpr float kEps = 1e-6f;
    for (int64_t i = 0; i < elemCount; ++i) {
        float golden = std::fabs(xHostData[i]);
        if (std::fabs(golden - yOut[i]) > kEps) {
            if (mismatch < 10) {
                LOG_PRINT("mismatch at index %ld: golden=%f, out=%f\n", (long)i, golden, yOut[i]);
            }
            ++mismatch;
        }
    }
    if (mismatch != 0) {
        LOG_PRINT("[FAIL] aclnnAbs precision check failed, mismatch=%ld / %ld\n",
                  (long)mismatch, (long)elemCount);
        aclDestroyTensor(x);
        aclrtFree(xDeviceAddr);
        aclDestroyTensor(y);
        aclrtFree(yDeviceAddr);
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
    LOG_PRINT("[PASS] aclnnAbs precision check passed, %ld elements all correct\n", (long)elemCount);

    // 释放资源
    aclDestroyTensor(x);
    aclrtFree(xDeviceAddr);
    
    aclDestroyTensor(y);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
