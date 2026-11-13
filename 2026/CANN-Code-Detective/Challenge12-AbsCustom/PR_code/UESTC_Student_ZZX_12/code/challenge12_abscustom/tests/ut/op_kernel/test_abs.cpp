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
    constexpr uint32_t numBlocks = 64;

    constexpr size_t xByteSize = 16384 * 4;
    constexpr size_t yByteSize = 16384 * 4;
    std::vector<float> xHost(16384, 0);
    std::vector<float> yHost(16384, 0);
    { std::ifstream input("float32_input_abs_x.bin", std::ios::binary);
      ASSERT_TRUE(input.good());
      input.read(reinterpret_cast<char*>(xHost.data()), xByteSize);
      ASSERT_EQ(input.gcount(), static_cast<std::streamsize>(xByteSize)); }
    
    
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(x, xHost.data(), xByteSize);
    
    AbsTilingData* tilingData = reinterpret_cast<AbsTilingData*>(tiling);
    memset(tiling, 0, tilingDataSize);
    tilingData->totalNum = size;
    tilingData->blockFactor = 256;
    tilingData->ubFactor = 256;
    
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((abs<1>), numBlocks, x, y, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float32_output_abs_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
