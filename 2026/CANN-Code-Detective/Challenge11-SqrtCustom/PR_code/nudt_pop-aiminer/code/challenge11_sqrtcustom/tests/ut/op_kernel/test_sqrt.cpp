#include "sqrt_tiling.h"
#include "../../../op_kernel/sqrt.cpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"

namespace {
constexpr size_t INPUT_ELEMENTS = 8 * 2048;
constexpr size_t GUARD_BYTES = 32;

// Consume the exact file used to produce the external NumPy golden.
template <typename T>
std::vector<T> LoadInput(const char* filename)
{
    const char* directory = std::getenv("SQRT_UT_DATA_DIR");
    const std::string path = directory == nullptr
        ? std::string(filename)
        : std::string(directory) + "/" + filename;
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto expectedBytes = static_cast<std::streamoff>(INPUT_ELEMENTS * sizeof(T));
    if (!stream || stream.tellg() != expectedBytes) {
        throw std::runtime_error("Missing or invalid Sqrt input file: " + path + "; run gen_data.py first");
    }
    std::vector<T> input(INPUT_ELEMENTS);
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(input.data()), expectedBytes)) {
        throw std::runtime_error("Failed to read Sqrt input file: " + path);
    }
    return input;
}

struct GmDeleter {
    void operator()(uint8_t* ptr) const
    {
        if (ptr != nullptr) {
            AscendC::GmFree(ptr);
        }
    }
};
using GmBuffer = std::unique_ptr<uint8_t, GmDeleter>;

template <int DATA_TYPE>
void RunCase(size_t size, int64_t blockFactor, int64_t ubFactor, uint32_t blocks,
             const char* outputFile = nullptr)
{
    using T = std::conditional_t<DATA_TYPE == SQRT_FLOAT16, half, float>;
    const char* filename = DATA_TYPE == SQRT_FLOAT16
        ? "float16_input_sqrt_x.bin" : "float32_input_sqrt_x.bin";
    const auto input = LoadInput<T>(filename);
    ASSERT_LE(size, input.size());
    const size_t bytes = size * sizeof(T);
    GmBuffer x(static_cast<uint8_t*>(AscendC::GmAlloc(bytes + GUARD_BYTES)));
    GmBuffer y(static_cast<uint8_t*>(AscendC::GmAlloc(bytes + GUARD_BYTES)));
    GmBuffer workspace(static_cast<uint8_t*>(AscendC::GmAlloc(GUARD_BYTES)));
    GmBuffer tiling(static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SqrtTilingData))));
    ASSERT_NE(x.get(), nullptr);
    ASSERT_NE(y.get(), nullptr);
    ASSERT_NE(workspace.get(), nullptr);
    ASSERT_NE(tiling.get(), nullptr);
    std::memcpy(x.get(), input.data(), bytes);
    std::memset(y.get(), 0xCD, bytes + GUARD_BYTES);
    const SqrtTilingData data{static_cast<int64_t>(size), blockFactor, ubFactor};
    std::memcpy(tiling.get(), &data, sizeof(data));

    ICPU_SET_TILING_KEY(DATA_TYPE);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((sqrt<DATA_TYPE>), blocks, x.get(), y.get(), workspace.get(), tiling.get());

    std::vector<T> output(size);
    std::memcpy(output.data(), y.get(), bytes);
    constexpr double relativeTolerance = DATA_TYPE == SQRT_FLOAT16 ? 0x1p-10 : 1e-5;
    for (size_t i = 0; i < size; ++i) {
        const double expected = std::sqrt(static_cast<double>(input[i]));
        EXPECT_TRUE(std::isfinite(static_cast<double>(output[i]))) << "index=" << i;
        EXPECT_NEAR(static_cast<double>(output[i]), expected, 1e-7 + relativeTolerance * expected)
            << "index=" << i << ", input=" << static_cast<double>(input[i]);
    }
    for (size_t i = 0; i < GUARD_BYTES; ++i) {
        EXPECT_EQ(y.get()[bytes + i], 0xCD) << "Output guard byte " << i;
    }
    if (outputFile != nullptr) {
        std::ofstream stream(outputFile, std::ios::binary);
        ASSERT_TRUE(stream.good());
        stream.write(reinterpret_cast<const char*>(output.data()), bytes);
        EXPECT_TRUE(stream.good());
    }
}
} // namespace

TEST(SqrtKernelTest, activity_shape)
{
    RunCase<SQRT_FLOAT32>(INPUT_ELEMENTS, 256, 128, 64, "float32_output_sqrt_0.bin");
}

TEST(SqrtKernelTest, multiple_tiles)
{
    RunCase<SQRT_FLOAT32>(INPUT_ELEMENTS, 4096, 80, 4);
}

TEST(SqrtKernelTest, unaligned_tail)
{
    RunCase<SQRT_FLOAT32>(1003, 256, 80, 4);
}

TEST(SqrtKernelTest, one_element_tail)
{
    RunCase<SQRT_FLOAT32>(17, 16, 16, 2);
}

TEST(SqrtKernelTest, fp16_activity_shape)
{
    RunCase<SQRT_FLOAT16>(INPUT_ELEMENTS, 256, 128, 64, "float16_output_sqrt_0.bin");
}

TEST(SqrtKernelTest, fp16_multiple_tiles)
{
    RunCase<SQRT_FLOAT16>(INPUT_ELEMENTS, 4096, 80, 4);
}

TEST(SqrtKernelTest, fp16_unaligned_tail)
{
    RunCase<SQRT_FLOAT16>(1003, 256, 80, 4);
}

TEST(SqrtKernelTest, fp16_one_element_tail)
{
    RunCase<SQRT_FLOAT16>(17, 16, 16, 2);
}
