/*!
 * \file test_relu.cpp
 * \brief Relu 算子 kernel UT 测试
 *
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "relu_tiling.h"
#include "../../../op_kernel/relu.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

class ReluKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ReluKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ReluKernelTest TearDown" << endl;
    }
};

TEST_F(ReluKernelTest, test_kernel_run)
{
    constexpr size_t size = 16384;
    constexpr size_t tilingDataSize = sizeof(ReluTilingData);
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

    ReluTilingData* tilingData = reinterpret_cast<ReluTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF((relu<float>), numBlocks, x, y, workspace, tiling);

    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float32_output_relu_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
