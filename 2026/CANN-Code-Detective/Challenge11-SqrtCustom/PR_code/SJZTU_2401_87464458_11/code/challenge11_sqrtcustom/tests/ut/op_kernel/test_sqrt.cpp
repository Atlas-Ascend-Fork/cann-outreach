/*!
 * \file test_sqrt.cpp
 * \brief Sqrt 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "sqrt_tiling.h"
#include "../../../op_kernel/sqrt.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

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

class SqrtKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "SqrtKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "SqrtKernelTest TearDown" << endl;
    }
};

TEST_F(SqrtKernelTest, test_kernel_run)
{
    constexpr size_t size = 16384;
    constexpr size_t tilingDataSize = sizeof(SqrtTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 16384 * 4;
    constexpr size_t yByteSize = 16384 * 4;
    std::vector<float> xHost(16384, 1);
    std::vector<float> yHost(16384, 0);
    
    
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(x, xHost.data(), xByteSize);
    
    // Keep the all-ones output for the supplied binary comparison script.
    SqrtTilingData* tilingData = reinterpret_cast<SqrtTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = 256;
    
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((sqrt<1>), numBlocks, x, y, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float32_output_sqrt_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SqrtKernelTest, varied_input_multicore_multitile)
{
    constexpr size_t size = 16384;
    constexpr size_t byteSize = size * sizeof(float);
    constexpr uint32_t numBlocks = 8;
    std::vector<float> xHost(size);
    std::vector<float> yHost(size, -1.0f);
    const std::array<float, 5> specialValues = {0.0f, 0.25f, 2.0f, 4.0f, 9.0f};
    for (size_t i = 0; i < size; ++i) {
        xHost[i] = i % 8 < specialValues.size() ? specialValues[i % 8] :
            static_cast<float>((i * 37) % 1009) / 16.0f;
    }

    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(byteSize));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(byteSize));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SqrtTilingData)));
    memcpy(x, xHost.data(), byteSize);
    memcpy(y, yHost.data(), byteSize);

    auto* tilingData = reinterpret_cast<SqrtTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size / numBlocks;
    tilingData->ubFactor = 256;
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((sqrt<1>), numBlocks, x, y, workspace, tiling);
    memcpy(yHost.data(), y, byteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    for (size_t i = 0; i < size; ++i) {
        const float expected = std::sqrt(xHost[i]);
        EXPECT_NEAR(yHost[i], expected, 1e-5f * std::max(1.0f, expected)) << "index=" << i;
    }
}
