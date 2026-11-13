/*!
 * \file test_leakyrelu.cpp
 * \brief Leakyrelu 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "leakyrelu_tiling.h"
#include "../../../op_kernel/leakyrelu.cpp"

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

class LeakyreluKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "LeakyreluKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "LeakyreluKernelTest TearDown" << endl;
    }
};

TEST_F(LeakyreluKernelTest, test_kernel_run)
{
    constexpr size_t size = 16384;
    constexpr size_t tilingDataSize = sizeof(LeakyreluTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 16384 * 4;
    constexpr size_t yByteSize = 16384 * 4;
    std::vector<uint8_t> xHost(xByteSize);
    std::vector<uint8_t> yHost(yByteSize);
    
    
    ASSERT_NE(std::getenv("SAMPLE_DIR"), nullptr);
    const std::string root = std::getenv("SAMPLE_DIR");
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    { std::ifstream f(root + "/input/x.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(x), 65536); ASSERT_EQ(f.gcount(), 65536); }
    
    // tilingData 字段与 op_kernel/leakyrelu_tiling_data.h 的定义一致：
    //   totalLength=总元素数, tileNum=核内分块数, negativeSlope=负半轴斜率(与 samples/case.json 的 0.01 对齐)
    LeakyreluTilingData* tilingData = reinterpret_cast<LeakyreluTilingData*>(tiling);
    tilingData->totalLength = size;
    tilingData->tileNum = 8;
    tilingData->negativeSlope = 0.01f;

    // 本工程按 dtype 选择模板实例（ASCENDC_TPL_SEL_PARAM），kernel 直接实例化 float 版本
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF((leakyrelu<float>), numBlocks, x, y, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs((root + "/output/y.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
