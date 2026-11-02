#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "aclnn_sqrt.h"

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

// Check the values returned by the operator, independently of ACL return codes.
static bool CheckSqrtOutput(const std::vector<float>& input, const std::vector<float>& output)
{
    if (input.empty() || input.size() != output.size()) {
        LOG_PRINT("Sqrt validation failed: invalid output size\n");
        return false;
    }
    constexpr double ATOL = 1e-7;
    constexpr double RTOL = 1e-5;
    constexpr size_t MAX_ERROR_LOGS = 5;
    size_t mismatches = 0;
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        const double expected = std::sqrt(static_cast<double>(input[i]));
        const double actual = static_cast<double>(output[i]);
        const double absError = std::isfinite(actual)
            ? std::abs(actual - expected) : std::numeric_limits<double>::infinity();
        const double relError = absError / std::max(std::abs(expected), 1e-6);
        maxAbsError = std::max(maxAbsError, absError);
        maxRelError = std::max(maxRelError, relError);
        if (!std::isfinite(actual) || absError > ATOL + RTOL * std::abs(expected)) {
            if (mismatches < MAX_ERROR_LOGS) {
                LOG_PRINT("Mismatch[%zu]: x=%.9g, actual=%.9g, expected=%.9g\n",
                          i, static_cast<double>(input[i]), actual, expected);
            }
            ++mismatches;
        }
    }
    LOG_PRINT("Sqrt validation %s: elements=%zu, mismatches=%zu, max_abs=%.9g, max_rel=%.9g\n",
              mismatches == 0 ? "PASSED" : "FAILED", input.size(), mismatches,
              maxAbsError, maxRelError);
    return mismatches == 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 构造输入 tensor
    aclTensor* x = nullptr;
    void* xDeviceAddr = nullptr;
    std::vector<int64_t> xShape = {8, 2048};
    // Include non-fixed points so identity-copy and all-zero kernels fail validation.
    constexpr std::array<float, 12> inputValues{
        0.0f, 0.25f, 1.0f, 2.0f, 4.0f, 9.0f, 16.0f, 100.0f,
        0.0001f, 65504.0f, 1e-8f, 10000.0f};
    std::vector<float> xHostData(GetShapeSize(xShape));
    for (size_t i = 0; i < xHostData.size(); ++i) {
        xHostData[i] = inputValues[i % inputValues.size()];
    }
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造输出 tensor
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    std::vector<int64_t> yShape = {8, 2048};
    std::vector<float> yHostData(16384, 0);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    

    // 调用 aclnnSqrtGetWorkspaceSize 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSqrtGetWorkspaceSize(x, y, &workspaceSize, &executor);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnSqrtGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnSqrt 第二段接口
    ret = aclnnSqrt(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnSqrt failed. ERROR: %d\n", ret); return ret);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    const size_t outputBytes = yHostData.size() * sizeof(float);
    ret = aclrtMemcpy(yHostData.data(), outputBytes, yDeviceAddr, outputBytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    bool valuesCorrect = false;
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Output readback failed. ERROR: %d\n", ret);
    } else {
        valuesCorrect = CheckSqrtOutput(xHostData, yHostData);
    }

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

    return valuesCorrect ? 0 : 1;
}
