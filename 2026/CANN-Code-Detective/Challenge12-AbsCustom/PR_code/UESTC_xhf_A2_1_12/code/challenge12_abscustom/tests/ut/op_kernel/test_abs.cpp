/*!
 * \file test_abs.cpp
 * \brief Abs 算子 kernel UT 测试
 *
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "abs_tiling.h"
#include "../../../op_kernel/abs.cpp"

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

class AbsKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "AbsKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "AbsKernelTest TearDown" << endl;
    }
};

TEST_F(AbsKernelTest, test_kernel_run)
{
    constexpr size_t size = 16384;
    constexpr size_t tilingDataSize = sizeof(AbsTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 16384 * 4;
    constexpr size_t yByteSize = 16384 * 4;
    std::vector<float> xHost(size);
    std::vector<float> yHost(size, 0);
    for (size_t i = 0; i < size; ++i) {
        xHost[i] = static_cast<float>(static_cast<int>(i % 33) - 16) / 4.0F;
    }


    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    if (!x || !y || !workspace || !tiling) {
        if (x) AscendC::GmFree(x);
        if (y) AscendC::GmFree(y);
        if (workspace) AscendC::GmFree(workspace);
        if (tiling) AscendC::GmFree(tiling);
        FAIL() << "Unable to allocate simulator buffers";
    }

    memcpy(x, xHost.data(), xByteSize);

    AbsTilingData* tilingData = reinterpret_cast<AbsTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = 2048;

    ICPU_SET_TILING_KEY(ABS_FLOAT32);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF((abs<ABS_FLOAT32>), numBlocks, x, y, workspace, tiling);

    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    for (size_t i = 0; i < size; ++i) {
        ASSERT_FLOAT_EQ(yHost[i], xHost[i] < 0.0F ? -xHost[i] : xHost[i]) << "Index " << i;
    }
    std::ofstream output("float32_output_abs_0.bin", std::ios::binary);
    output.write(reinterpret_cast<const char*>(yHost.data()), yByteSize);
    ASSERT_TRUE(output.good());
}
