/*!
 * \file test_sigmoid.cpp
 * \brief Sigmoid 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "sigmoid_tiling.h"
#include "../../../op_kernel/sigmoid.cpp"

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

class SigmoidKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "SigmoidKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "SigmoidKernelTest TearDown" << endl;
    }
};

TEST_F(SigmoidKernelTest, test_kernel_run)
{
    constexpr size_t size = 16384;
    constexpr size_t tilingDataSize = sizeof(SigmoidTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 16384 * 4;
    constexpr size_t yByteSize = 16384 * 4;
    // 与 sigmoid_data/gen_data.py 保持同一生成公式：[-20, 20] 等间距，覆盖饱和区与过渡区
    std::vector<float> xHost(16384);
    for (size_t i = 0; i < xHost.size(); i++) {
        xHost[i] = static_cast<float>(-20.0 + 40.0 * static_cast<double>(i) / 16383.0);
    }
    std::vector<float> yHost(16384, 0);
    
    
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(x, xHost.data(), xByteSize);
    
    // tiling 数据与 op_host/sigmoid_tiling.cpp 的字段定义保持一致
    // （totalNum、blockFactor、ubFactor），ubFactor 取 2048 避免 UB 溢出
    SigmoidTilingData* tilingData = reinterpret_cast<SigmoidTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = 2048;
    
    // tilingKey 与 op_host/sigmoid_tiling.cpp 中 SetTilingKey 设置的值一致
    // （float 输入对应 SIGMOID_TPL_SCH_MODE_1）
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((sigmoid<1>), numBlocks, x, y, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float32_output_sigmoid_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
