/*!
 * \file test_exp.cpp
 * \brief Exp 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "exp_tiling.h"
#include "../../../op_kernel/exp.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

class ExpKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ExpKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ExpKernelTest TearDown" << endl;
    }
};

TEST_F(ExpKernelTest, test_kernel_run)
{
    constexpr size_t size = 16384;
    constexpr size_t tilingDataSize = sizeof(ExpTilingData);
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
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/exp_tiling_data.h 中的字段定义
    //         参考 op_host/exp_tiling.cpp 中的 tiling 计算逻辑
    ExpTilingData* tilingData = reinterpret_cast<ExpTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = 1024;
    
    // TODO: tilingKey 应与 op_host/exp_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((exp<1>), numBlocks, x, y, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(yHost[i], std::exp(xHost[i]), 1e-4f);
    }
    { std::ofstream _ofs("float32_output_exp_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ExpKernelTest, test_multicore_unaligned_tail)
{
    constexpr size_t size = 2051;
    constexpr size_t guard = 16;
    constexpr size_t bytes = (size + guard) * sizeof(float);
    std::vector<float> input(size + guard, 0.0f);
    std::vector<float> output(size + guard, -123.0f);
    for (size_t i = 0; i < size; ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 41) - 20) / 10.0f;
    }
    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(bytes));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(bytes));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(ExpTilingData)));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);
    memcpy(x, input.data(), bytes);
    memcpy(y, output.data(), bytes);
    auto* data = reinterpret_cast<ExpTilingData*>(tiling);
    data->totalNum = size;
    data->blockFactor = 1024;
    data->ubFactor = 256;
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((exp<1>), 3, x, y, workspace, tiling);
    memcpy(output.data(), y, bytes);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(output[i], std::exp(input[i]), 1e-4f);
    }
    for (size_t i = size; i < size + guard; ++i) {
        EXPECT_EQ(output[i], -123.0f) << "tail write exceeded output boundary";
    }
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
