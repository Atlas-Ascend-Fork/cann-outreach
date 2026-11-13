/*!
 * \file test_leakyrelu.cpp
 * \brief Leakyrelu 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "leakyrelu_tiling.h"
#include "../../../op_kernel/leakyrelu.cpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"

namespace {

struct KernelCase {
    const char* inputFile;
    const char* outputFile;
    int64_t elements;
    int64_t blockFactor;
    uint32_t blocks;
    float slope;
};

class LeakyreluKernelTest : public testing::TestWithParam<KernelCase> {};

TEST_P(LeakyreluKernelTest, RunsSample)
{
    const auto& test = GetParam();
    const char* sampleDir = std::getenv("SAMPLE_DIR");
    ASSERT_NE(sampleDir, nullptr);
    const std::string root(sampleDir);
    const size_t bytes = test.elements * sizeof(float);
    constexpr size_t guardBytes = 32;
    constexpr uint8_t guardValue = 0xA5;
    std::vector<uint8_t> input(bytes);
    std::ifstream in(root + "/input/" + test.inputFile, std::ios::binary);
    ASSERT_TRUE(in.read(reinterpret_cast<char*>(input.data()), bytes));
    ASSERT_EQ(in.peek(), std::char_traits<char>::eof());

    const auto freeGm = [](uint8_t* ptr) { AscendC::GmFree(ptr); };
    const auto allocate = [freeGm](size_t size) {
        return std::unique_ptr<uint8_t, decltype(freeGm)>{
            static_cast<uint8_t*>(AscendC::GmAlloc(size)), freeGm};
    };
    auto x = allocate(bytes + guardBytes);
    auto y = allocate(bytes + guardBytes);
    auto workspace = allocate(32);
    auto tiling = allocate(sizeof(LeakyreluTilingData));
    ASSERT_TRUE(x && y && workspace && tiling);
    std::memcpy(x.get(), input.data(), bytes);
    std::memset(x.get() + bytes, 0, guardBytes);
    std::memset(y.get(), guardValue, bytes + guardBytes);

    const LeakyreluTilingData tilingData{
        test.elements, test.blockFactor, test.blockFactor, test.slope, 0
    };
    std::memcpy(tiling.get(), &tilingData, sizeof(tilingData));

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((leakyrelu<1>), test.blocks, x.get(), y.get(), workspace.get(), tiling.get());

    EXPECT_TRUE(std::all_of(y.get() + bytes, y.get() + bytes + guardBytes,
                           [](uint8_t value) { return value == guardValue; }));
    std::ofstream out(root + "/output/" + test.outputFile, std::ios::binary);
    ASSERT_TRUE(out.write(reinterpret_cast<const char*>(y.get()), bytes));
}

INSTANTIATE_TEST_SUITE_P(
    Samples, LeakyreluKernelTest,
    testing::Values(
        KernelCase{"x.bin", "y.bin", 16384, 1024, 16, 0.01f},
        KernelCase{"x_tail.bin", "y_tail.bin", 1031, 520, 2, 0.125f}));

} // namespace
