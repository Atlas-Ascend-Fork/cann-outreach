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
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

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
    // 与 exp_data/gen_data.py 保持同一输入：全 1 张量 (8, 2048)
    std::vector<float> xHost(16384, 1);
    std::vector<float> yHost(16384, 0);
    
    
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(x, xHost.data(), xByteSize);
    
    // tiling 数据与 op_kernel/exp_tiling_data.h 的字段定义保持一致
    // （totalNum、blockFactor、ubFactor），ubFactor 取 2048 避免 UB 溢出
    ExpTilingData* tilingData = reinterpret_cast<ExpTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = 2048;
    
    // tilingKey 与 op_host/exp_tiling.cpp 中 SetTilingKey 设置的值一致
    // （float 输入对应 EXP_TPL_SCH_MODE_1）
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((exp<1>), numBlocks, x, y, workspace, tiling);
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float32_output_exp_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
