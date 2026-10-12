/*!
 * \file test_relu.cpp
 * \brief Relu 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "relu_tiling.h"
#include "../../../op_kernel/relu.cpp"

#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
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

TEST_F(ReluKernelTest, test_kernel_run)
{
    constexpr size_t size = 92160;
    constexpr size_t tilingDataSize = sizeof(ReluTilingData);
    constexpr uint32_t numBlocks = 64;

    constexpr size_t xByteSize = size * sizeof(float);
    constexpr size_t yByteSize = size * sizeof(float);
    std::vector<float> xHost(size);
    std::vector<float> yHost(size, -12345.0f);
    // 与 gen_data.py 保持一致：循环覆盖负数、零、正数及可精确表示的小数。
    for (size_t i = 0; i < size; ++i) {
        xHost[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
    }

    auto freeGm = [](uint8_t* ptr) { AscendC::GmFree(ptr); };
    using GmBuffer = std::unique_ptr<uint8_t, decltype(freeGm)>;
    GmBuffer x(static_cast<uint8_t*>(AscendC::GmAlloc(xByteSize)), freeGm);
    GmBuffer y(static_cast<uint8_t*>(AscendC::GmAlloc(yByteSize)), freeGm);
    GmBuffer workspace(static_cast<uint8_t*>(AscendC::GmAlloc(32)), freeGm);
    GmBuffer tiling(static_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize)), freeGm);
    ASSERT_NE(x.get(), nullptr);
    ASSERT_NE(y.get(), nullptr);
    ASSERT_NE(workspace.get(), nullptr);
    ASSERT_NE(tiling.get(), nullptr);

    memcpy(x.get(), xHost.data(), xByteSize);
    // 未写回的元素必须导致断言失败，包括期望为零的位置。
    memcpy(y.get(), yHost.data(), yByteSize);

    // 多核切分，每核 1440 个元素，使用 1024 元素 UB 验证尾块。
    ReluTilingData* tilingData = reinterpret_cast<ReluTilingData*>(tiling.get());
    tilingData->totalNum = size;
    tilingData->blockFactor = size / numBlocks;
    tilingData->ubFactor = 1024;
    
    // float32 对应调度模式 1。
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((relu<1>), numBlocks, x.get(), y.get(), workspace.get(), tiling.get());
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y.get(), yByteSize);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(yHost[i], xHost[i] > 0.0f ? xHost[i] : 0.0f) << "index=" << i;
    }
    std::ofstream output("float32_output_relu_0.bin", std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output.write(reinterpret_cast<const char*>(yHost.data()), yByteSize);
    output.close();
    ASSERT_TRUE(output.good());
}
