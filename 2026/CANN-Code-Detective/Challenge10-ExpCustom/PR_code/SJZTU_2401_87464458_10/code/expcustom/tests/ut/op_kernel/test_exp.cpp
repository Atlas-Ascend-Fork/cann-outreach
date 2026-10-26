/*!
 * \file test_exp.cpp
 * \brief Exp kernel tests using the host tiling fields and eight blocks.
 */

#include "exp_tiling.h"
#include "../../../op_kernel/exp.cpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"

namespace {

constexpr size_t ELEMENT_COUNT = 8 * 2048;
constexpr uint32_t BLOCK_COUNT = 8;
constexpr uint32_t TILE_NUM = 16;

struct GmDeleter {
    void operator()(uint8_t* address) const
    {
        if (address != nullptr) {
            AscendC::GmFree(address);
        }
    }
};

using GmBuffer = std::unique_ptr<uint8_t, GmDeleter>;

class ExpKernelTest : public testing::Test {
protected:
    static void RunAndCheck(const std::vector<float>& input, std::vector<float>& output)
    {
        ASSERT_EQ(input.size(), ELEMENT_COUNT);
        const size_t byteSize = input.size() * sizeof(float);
        output.assign(input.size(), std::numeric_limits<float>::quiet_NaN());

        GmBuffer x(static_cast<uint8_t*>(AscendC::GmAlloc(byteSize)));
        GmBuffer y(static_cast<uint8_t*>(AscendC::GmAlloc(byteSize)));
        GmBuffer workspace(static_cast<uint8_t*>(AscendC::GmAlloc(32)));
        GmBuffer tiling(static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(ExpTilingData))));
        ASSERT_NE(x.get(), nullptr);
        ASSERT_NE(y.get(), nullptr);
        ASSERT_NE(workspace.get(), nullptr);
        ASSERT_NE(tiling.get(), nullptr);
        std::memcpy(x.get(), input.data(), byteSize);
        // A missed tile remains NaN and fails the finite-output assertion below.
        std::memcpy(y.get(), output.data(), byteSize);

        auto* tilingData = reinterpret_cast<ExpTilingData*>(tiling.get());
        tilingData->totalLength = static_cast<uint32_t>(input.size());
        tilingData->tileNum = TILE_NUM;

        // The supplied code_add reference uses the same dtype declaration: float -> 0.
        ICPU_SET_TILING_KEY(0);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        uint8_t* xGm = x.get();
        uint8_t* yGm = y.get();
        uint8_t* workspaceGm = workspace.get();
        uint8_t* tilingGm = tiling.get();
        ICPU_RUN_KF((exp<float>), BLOCK_COUNT, xGm, yGm, workspaceGm, tilingGm);

        std::memcpy(output.data(), y.get(), byteSize);
        for (size_t i = 0; i < input.size(); ++i) {
            const float expected = static_cast<float>(std::exp(static_cast<double>(input[i])));
            const float tolerance = 1.0e-6f + 1.0e-4f * std::abs(expected);
            ASSERT_TRUE(std::isfinite(output[i])) << "index=" << i;
            ASSERT_NEAR(output[i], expected, tolerance) << "index=" << i << ", input=" << input[i];
        }
    }
};

TEST_F(ExpKernelTest, test_kernel_run)
{
    // gen_data.py produces float32 ones with shape (8, 2048). Keep its output fixture.
    const std::vector<float> input(ELEMENT_COUNT, 1.0f);
    std::vector<float> output;
    ASSERT_NO_FATAL_FAILURE(RunAndCheck(input, output));

    std::ofstream outputFile("float32_output_exp_0.bin", std::ios::binary);
    ASSERT_TRUE(outputFile.is_open());
    outputFile.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
    ASSERT_TRUE(outputFile.good());
}

TEST_F(ExpKernelTest, test_kernel_varied_finite)
{
    std::vector<float> input(ELEMENT_COUNT);
    for (size_t i = 0; i < input.size(); ++i) {
        // Repeat finite values in [-4, 4] with different values across tile/block boundaries.
        input[i] = static_cast<float>(static_cast<int32_t>((i * 37) % 1025) - 512) / 128.0f;
    }
    std::vector<float> output;
    ASSERT_NO_FATAL_FAILURE(RunAndCheck(input, output));
    // Do not write another output bin: compare_data.py expects the original single fixture.
}

} // namespace
