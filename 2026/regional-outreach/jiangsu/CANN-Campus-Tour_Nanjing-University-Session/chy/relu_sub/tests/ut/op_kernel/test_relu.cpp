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

// float16 位模式：1.0 = 0x3C00，-1.0 = 0xBC00
static const uint16_t HALF_ONE = 0x3C00;
static const uint16_t HALF_NEG_ONE = 0xBC00;

TEST_F(ReluKernelTest, test_kernel_run)
{
    constexpr size_t size = 32768;   // 4D ND {8, 4, 2, 512} 总元素数
    constexpr size_t tilingDataSize = sizeof(ReluTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 32768 * 2;
    constexpr size_t yByteSize = 32768 * 2;
    std::vector<uint16_t> xHost(32768);
    for (size_t i = 0; i < 32768; i++) {
        xHost[i] = (i % 2 == 0) ? HALF_ONE : HALF_NEG_ONE;
    }
    std::vector<uint16_t> yHost(32768, 0);


    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    memcpy(x, xHost.data(), xByteSize);

    // tiling 数据与 op_host/relu_tiling.cpp 的赋值保持一致（totalLength、tileNum）
    ReluTilingData* tilingData = reinterpret_cast<ReluTilingData*>(tiling);
    tilingData->totalLength = size;
    tilingData->tileNum = 16;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // 直接实例化 float16 版本的模板核函数（与 tiling_key 的 DT_X=float16 实例对应）
    ICPU_RUN_KF((relu<AscendC::half>), numBlocks, x, y, workspace, tiling);

    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float16_output_relu_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
