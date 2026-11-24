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
    // 输入模式与 gen_data.py 保持一致：(i % 16 - 8) * 0.25，覆盖 [-2.0, 1.75]，正负值兼有
    std::vector<float> xHost(16384, 1);
    std::vector<float> yHost(16384, 0);
    for (size_t i = 0; i < 16384; i++) {
        xHost[i] = static_cast<float>((static_cast<int64_t>(i) % 16) - 8) * 0.25f;
    }

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    memcpy(x, xHost.data(), xByteSize);

    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/leakyrelu_tiling_data.h 中的字段定义
    //         参考 op_host/leakyrelu_tiling.cpp 中的 tiling 计算逻辑
    LeakyreluTilingData* tilingData = reinterpret_cast<LeakyreluTilingData*>(tiling);
    tilingData->totalLength = size;
    tilingData->tileNum = 8;        // 与 op_host/leakyrelu_tiling.cpp 中 TILE_NUM 一致
    tilingData->negativeSlope = 0.01f; // 与 gen_data.py 中 golden 计算所用 α 一致

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // 直接实例化 float 版本的模板核函数（与 tiling_key 的 DT_X=float 实例对应）
    ICPU_RUN_KF((leakyrelu<float>), numBlocks, x, y, workspace, tiling);

    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float32_output_leakyrelu_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
