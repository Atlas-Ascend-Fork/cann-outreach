/*!
 * \file test_add_custom.cpp
 * \brief add_custom 算子 kernel UT（CPU 仿真，不依赖 NPU）
 *
 * 直接在 CPU 仿真器（tikicpulib）上运行 add_custom kernel，覆盖三类形状：
 *   1) 对齐形状  [8, 2048]  = 16384 元素：8 核 => 每核 2048 元素，tile=1024 => 每核 2 个完整 tile
 *   2) 跨核非对齐 [3, 100]  = 300 元素：8 核 => base=37, rem=4，前 4 核 38 元素、后 4 核 37 元素
 *   3) 核内尾块  [10, 1200] = 12000 元素：8 核 => 每核 1500 元素，tile=1024 => 1024 + 476 尾块
 *
 * 数值精度：输入/输出均为 float16，golden 用 float64 求和后再转 float16，
 * 按社区标准做 MERE/MARE 比对（float16 阈值 2^-10，MARE 上限 10*阈值）。
 */

#include "add_custom_tiling.h"
#include "../../../op_kernel/add_custom.cpp"

#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

// ------------------------- float16 转换工具 -------------------------
static uint16_t FloatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (uint32_t)(exp << 10) | mant);
}

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // 非规格化数
            int32_t e = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3ff;
            bits = sign | ((uint32_t)e << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

// ------------------------- 单形状运行 + 精度比对 -------------------------
static bool RunAndCheck(const char* name, int64_t dim0, int64_t dim1, uint32_t numBlocks, uint32_t tileLength)
{
    int64_t n = dim0 * dim1;

    // 生成输入（float32 值域 [0,32)，避免 float16 溢出），golden 用 float64 求和后转 float16
    std::vector<uint16_t> xH(n), yH(n), goldenH(n);
    for (int64_t i = 0; i < n; i++) {
        float xf = (float)(i % 1000) / 32.0f;
        float yf = (float)(i % 77) / 8.0f;
        double golden = (double)xf + (double)yf;
        xH[i] = FloatToHalf(xf);
        yH[i] = FloatToHalf(yf);
        goldenH[i] = FloatToHalf((float)golden);
    }

    size_t byteSize = (size_t)n * sizeof(uint16_t);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(byteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(byteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(byteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AddCustomTilingData));

    memcpy(x, xH.data(), byteSize);
    memcpy(y, yH.data(), byteSize);

    // 与 op_host/add_custom_tiling.cpp 的赋值保持一致（totalLength、tileLength）
    AddCustomTilingData tilingData;
    tilingData.totalLength = (uint32_t)n;
    tilingData.tileLength = tileLength;
    memcpy(tiling, &tilingData, sizeof(AddCustomTilingData));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // 直接实例化 float16 版本的模板核函数（与 tiling_key 的 DT_X=float16 实例对应）
    ICPU_RUN_KF((add_custom<__half>), numBlocks, x, y, z, workspace, tiling);

    std::vector<uint16_t> zH(n);
    memcpy(zH.data(), z, byteSize);

    // 精度比对：MERE / MARE
    double mare = 0.0, mereSum = 0.0, maxAbs = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double o = (double)HalfToFloat(zH[i]);
        double g = (double)HalfToFloat(goldenH[i]);
        double diff = fabs(o - g);
        if (diff > maxAbs) maxAbs = diff;
        double denom = fabs(g);
        if (denom < 1e-6) denom = 1e-6;
        double rel = diff / denom;
        if (rel > mare) mare = rel;
        mereSum += rel;
    }
    double mere = mereSum / (double)n;

    const double thr = pow(2.0, -10); // float16 社区标准阈值
    bool pass = (mere < thr) && (mare < 10.0 * thr);

    cout << "[add_custom] " << name << " shape=[" << dim0 << "," << dim1 << "] n=" << n
         << " maxAbs=" << maxAbs << " MERE=" << mere << " MARE=" << mare
         << " (pass if MERE<"<< thr << " && MARE<"<< (10.0 * thr) << ") => "
         << (pass ? "PASS" : "FAIL") << endl;

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    return pass;
}

class AddCustomKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "AddCustomKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "AddCustomKernelTest TearDown" << endl; }
};

// 对齐形状：16384 元素 / 8 核 = 2048/核，tile=1024 => 每核 2 个完整 tile
TEST_F(AddCustomKernelTest, aligned_8x2048)
{
    bool pass = RunAndCheck("aligned", 8, 2048, 8, 1024);
    EXPECT_TRUE(pass);
}

// 跨核非对齐：300 元素 / 8 核 => base=37, rem=4，前 4 核 38 元素、后 4 核 37 元素
TEST_F(AddCustomKernelTest, unaligned_cross_core_3x100)
{
    bool pass = RunAndCheck("unaligned_cross_core", 3, 100, 8, 1024);
    EXPECT_TRUE(pass);
}

// 核内尾块：12000 元素 / 8 核 = 1500/核，tile=1024 => 1024 + 476 尾块
TEST_F(AddCustomKernelTest, unaligned_tail_10x1200)
{
    bool pass = RunAndCheck("unaligned_tail", 10, 1200, 8, 1024);
    EXPECT_TRUE(pass);
}
