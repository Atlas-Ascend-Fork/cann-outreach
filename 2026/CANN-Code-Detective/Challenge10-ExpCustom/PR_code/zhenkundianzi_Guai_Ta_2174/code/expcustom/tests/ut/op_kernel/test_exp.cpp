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
#include <cmath>
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

// ================= 原来的测试用例 =================
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
    
    ExpTilingData* tilingData = reinterpret_cast<ExpTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = 8192;
    
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((exp<1>), numBlocks, x, y, workspace, tiling);
    
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float32_output_exp_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    
    // 补上原来缺失的资源释放
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// ================= 新增的尾块多核测试用例 =================
TEST_F(ExpKernelTest, test_kernel_tail)
{
    // 构造非整除的数据量，触发多核偏移和尾块计算
    constexpr size_t totalElements = 10000;
    constexpr size_t blockFactor = 4096;
    constexpr size_t ubFactor = 1024;
    constexpr uint32_t numBlocks = 3; // 3个核

    constexpr size_t xByteSize = totalElements * 4;
    constexpr size_t yByteSize = totalElements * 4;
    
    std::vector<float> xHost(totalElements, 1);
    std::vector<float> yHost(totalElements, 0);
    
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ExpTilingData));
    
    memcpy(x, xHost.data(), xByteSize);
    
    ExpTilingData* tilingData = reinterpret_cast<ExpTilingData*>(tiling);
    tilingData->totalNum    = totalElements;
    tilingData->blockFactor = blockFactor;
    tilingData->ubFactor    = ubFactor;
    
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((exp<1>), numBlocks, x, y, workspace, tiling);
    
    memcpy(yHost.data(), y, yByteSize);
    
    // 校验所有元素计算是否正确，用于确认尾块没有越界、漏算
    for (size_t i = 0; i < totalElements; i++) {
        ASSERT_NEAR(yHost[i], std::exp(1.0f), 1e-5f) << "Mismatch at index " << i;
    }
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
