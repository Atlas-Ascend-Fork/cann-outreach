/*!
 * \file test_abs.cpp
 * \brief Abs 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 * 输入从 gen_data.py 生成的 float32_input_abs_x.bin 读取，保证与 golden 同源
 */

#include "abs_tiling.h"
#include "../../../op_kernel/abs.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
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

// 从 bin 读取 float32 输入，长度必须与 out.size() 完全一致
static bool ReadBinFloat32(const char* path, std::vector<float>& out)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.good()) return false;
    std::streamsize n = ifs.tellg();
    if (n <= 0) return false;
    if (n != static_cast<std::streamsize>(out.size() * sizeof(float))) return false;
    ifs.seekg(0);
    ifs.read(reinterpret_cast<char*>(out.data()), n);
    return ifs.good();
}

class AbsKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "AbsKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "AbsKernelTest TearDown" << endl;
    }
};

TEST_F(AbsKernelTest, test_kernel_run)
{
    constexpr size_t size = 16384;
    constexpr size_t tilingDataSize = sizeof(AbsTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = size * sizeof(float);
    constexpr size_t yByteSize = size * sizeof(float);

    // ★ 输入从 gen_data.py 生成的 bin 读取，与 golden 严格同源
    std::vector<float> xHost(size);
    {
        const char* candidates[] = {
            "float32_input_abs_x.bin",
            "../../op_kernel/abs_data/float32_input_abs_x.bin",
            "../../../op_kernel/abs_data/float32_input_abs_x.bin",
            "../op_kernel/abs_data/float32_input_abs_x.bin",
            "op_kernel/abs_data/float32_input_abs_x.bin",
        };
        bool ok = false;
        std::string usedPath;
        for (const char* p : candidates) {
            if (ReadBinFloat32(p, xHost)) { ok = true; usedPath = p; break; }
        }
        ASSERT_TRUE(ok)
            << "找不到 float32_input_abs_x.bin（或长度不是 " << size
            << " 个 float32），请先运行 gen_data.py";
        cout << "[INFO] Loaded input bin: " << usedPath << endl;
    }

    // golden 同源派生（UT 内自检用；外部 compare 用 gen_data.py 的 golden bin）
    std::vector<float> yGolden(size);
    for (size_t i = 0; i < size; ++i) {
        yGolden[i] = std::fabs(xHost[i]);
    }

    int negCount = 0;
    for (float v : xHost) { if (v < 0.0f) ++negCount; }
    ASSERT_GT(negCount, 0) << "输入 bin 必须包含负数，否则无法验证 abs 语义";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    memcpy(x, xHost.data(), xByteSize);
    memset(y, 0, yByteSize);

    AbsTilingData* tilingData = reinterpret_cast<AbsTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = 4096;

    ICPU_SET_TILING_KEY(1);                        // 对应 abs<1> → float 分支
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((abs<1>), numBlocks, x, y, workspace, tiling);

    // kernel 输出
    std::vector<float> yKernel(size);
    memcpy(yKernel.data(), y, yByteSize);

    // 逐元素比对 kernel 输出 vs golden
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(yGolden[i], yKernel[i]) << "mismatch at index " << i;
    }

    // 保存输出供 compare_data.py 比对
    {
        std::ofstream _ofs("float32_output_abs_0.bin", std::ios::binary);
        _ofs.write(reinterpret_cast<const char*>(yKernel.data()), yByteSize);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
