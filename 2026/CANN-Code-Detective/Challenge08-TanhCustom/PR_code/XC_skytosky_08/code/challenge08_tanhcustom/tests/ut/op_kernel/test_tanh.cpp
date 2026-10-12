/*!
 * \file test_tanh.cpp
 * \brief Tanh 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "tanh_tiling.h"
#include "../../../op_kernel/tanh.cpp"

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

class TanhKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "TanhKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "TanhKernelTest TearDown" << endl;
    }
};

TEST_F(TanhKernelTest, test_kernel_run)
{
    constexpr size_t size = 16384;
    constexpr size_t tilingDataSize = sizeof(TanhTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 16384 * 4;
    constexpr size_t yByteSize = 16384 * 4;
    std::vector<float> xHost(16384, 1);
    std::vector<float> yHost(16384, 0);
    
    
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    if (x == nullptr || y == nullptr || workspace == nullptr || tiling == nullptr) {
        ADD_FAILURE() << "GmAlloc failed";
        if (x) AscendC::GmFree(x);
        if (y) AscendC::GmFree(y);
        if (workspace) AscendC::GmFree(workspace);
        if (tiling) AscendC::GmFree(tiling);
        return;
    }
    
    memcpy(x, xHost.data(), xByteSize);
    
    // tilingData 字段与 op_kernel/tanh_tiling_data.h 中的定义一致
    TanhTilingData* tilingData = reinterpret_cast<TanhTilingData*>(tiling);
    tilingData->totalLength = size;
    tilingData->tileNum = 1;
    
    // tilingKey 与 op_host/tanh_tiling.cpp 中 ASCENDC_TPL_SEL_PARAM 的选择一致：
    // 0 -> float32（tanh<float>），1 -> float16（tanh<half>）
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((tanh<float>), numBlocks, x, y, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float32_output_tanh_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
