#include <cstring>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "leakyrelu_tiling_data.h"

namespace LeakyreluUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "Leakyrelu";

struct LeakyreluTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::string socVersion;
    uint64_t expectTilingKey;
    float negativeSlope;
    LeakyreluTilingData expected;
    size_t expectBlockNum;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

static LeakyreluTestParam testCases[] = {
    {"leakyrelu_float", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND,
     "Ascend910B", 1UL, 0.01F, {16384, 256, 256, 0.01F}, 64, 64, 262144, 4096},
    {"leakyrelu_half_custom_slope", {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B",
     0UL, 0.125F, {16384, 256, 256, 0.125F}, 64, 64, 262144, 4096},
};

class LeakyreluTilingTest : public testing::TestWithParam<LeakyreluTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "LeakyreluTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "LeakyreluTilingTest TearDown." << std::endl;
    }
};

struct LeakyreluCompileInfo {} compileInfo;

static void TestOneParamCase(const LeakyreluTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.emplace_back("negative_slope", Ops::Math::AnyValue::CreateFrom<float>(param.negativeSlope));
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, info));
    EXPECT_EQ(info.tilingKey, param.expectTilingKey);
    EXPECT_EQ(info.blockNum, param.expectBlockNum);
    EXPECT_EQ(info.workspaceSizes, (std::vector<int64_t>{0}));
    ASSERT_EQ(info.tilingDataSize, sizeof(LeakyreluTilingData));
    ASSERT_NE(info.tilingData.get(), nullptr);
    LeakyreluTilingData actual;
    std::memcpy(&actual, info.tilingData.get(), sizeof(actual));
    EXPECT_EQ(actual.totalNum, param.expected.totalNum);
    EXPECT_EQ(actual.blockFactor, param.expected.blockFactor);
    EXPECT_EQ(actual.ubFactor, param.expected.ubFactor);
    EXPECT_FLOAT_EQ(actual.negativeSlope, param.expected.negativeSlope);
}

TEST_P(LeakyreluTilingTest, tiling_test)
{
    const LeakyreluTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    LeakyreluTilingTests,
    LeakyreluTilingTest,
    testing::ValuesIn(testCases));

}
