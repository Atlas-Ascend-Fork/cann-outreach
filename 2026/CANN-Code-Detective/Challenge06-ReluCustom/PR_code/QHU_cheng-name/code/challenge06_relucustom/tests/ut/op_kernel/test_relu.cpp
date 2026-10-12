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
#include <cmath>
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

TEST_F(ReluKernelTest, test_kernel_run)
{
    constexpr size_t size = 16384;
    constexpr size_t tilingDataSize = sizeof(ReluTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 16384 * 4;
    constexpr size_t yByteSize = 16384 * 4;
    // 正负混合输入（含 0），确保 max(0,x) 被真实验证（透传型错误 kernel 无法通过）
    std::vector<float> xHost(16384);
    for (size_t i = 0; i < xHost.size(); i++) {
        xHost[i] = (i % 5 == 0) ? -3.5f : ((i % 7 == 0) ? 0.0f : (float)(i % 11) + 0.25f);
    }
    std::vector<float> yHost(16384, 0);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    memcpy(x, xHost.data(), xByteSize);

    // tilingData：totalNum/blockFactor/ubFactor 与 op_host/relu_tiling.cpp 的 tiling 逻辑一致
    // （totalNum=全部、blockFactor=每核、ubFactor=每块；numBlocks=1 时三者相同）
    ReluTilingData* tilingData = reinterpret_cast<ReluTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;

    // tilingKey 与 op_host/relu_tiling.cpp 一致（0=float32 实例）
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF((relu<0>), numBlocks, x, y, workspace, tiling);

    // 断言：逐元素验证 y = max(0, x)（ICPU_RUN_KF 执行无异常即内核返回成功）
    memcpy(yHost.data(), y, yByteSize);
    bool allOk = true;
    for (size_t i = 0; i < xHost.size(); i++) {
        float expect = (xHost[i] > 0.0f) ? xHost[i] : 0.0f;
        if (std::fabs(yHost[i] - expect) > 1e-5f) {
            allOk = false;
            break;
        }
    }
    EXPECT_TRUE(allOk) << "kernel 输出与 max(0,x) 不一致";

    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    { std::ofstream _ofs("float32_output_relu_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 多核切分 + 非整除尾块：总元素数不能被核数整除，且尾核长度按 fp32 折算后不是 32B 的整数倍，
// 同时覆盖 Init 的尾部截断、逐块 currentNum 与 DataCopyPad 精确搬运三条路径
TEST_F(ReluKernelTest, test_kernel_run_tail_not_divisible)
{
    constexpr uint32_t numBlocks = 8;
    constexpr size_t totalNum = 4095;    // 4095 = 7 * 512 + 511，最后一个核不足整块
    constexpr size_t blockFactor = 512;  // CeilDiv(4095, 8)
    constexpr size_t ubFactor = 512;
    constexpr size_t xByteSize = totalNum * sizeof(float);
    constexpr size_t yByteSize = totalNum * sizeof(float);

    std::vector<float> xHost(totalNum);
    for (size_t i = 0; i < totalNum; i++) {
        xHost[i] = (i % 5 == 0) ? -2.25f : ((i % 7 == 0) ? 0.0f : (float)(i % 13) + 0.5f);
    }
    std::vector<float> yHost(totalNum, 0.0f);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ReluTilingData));

    memcpy(x, xHost.data(), xByteSize);
    // 输出预置为 0xFF（按 fp32 解释为 NaN）：任何未被写入的元素都会被判为不一致，
    // 避免"尾部静默丢弃"因显存初值恰为 0 而误判通过
    memset(y, 0xFF, yByteSize);

    ReluTilingData* tilingData = reinterpret_cast<ReluTilingData*>(tiling);
    tilingData->totalNum = totalNum;
    tilingData->blockFactor = blockFactor;
    tilingData->ubFactor = ubFactor;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF((relu<0>), numBlocks, x, y, workspace, tiling);

    memcpy(yHost.data(), y, yByteSize);
    size_t firstBad = static_cast<size_t>(-1);
    for (size_t i = 0; i < totalNum; i++) {
        float expect = (xHost[i] > 0.0f) ? xHost[i] : 0.0f;
        if (std::fabs(yHost[i] - expect) > 1e-5f) {
            firstBad = i;
            break;
        }
    }
    EXPECT_TRUE(firstBad == static_cast<size_t>(-1))
        << "非整除尾块存在未计算/越界元素，首个异常下标: " << firstBad << "（尾核区间 3584~4094）";

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
