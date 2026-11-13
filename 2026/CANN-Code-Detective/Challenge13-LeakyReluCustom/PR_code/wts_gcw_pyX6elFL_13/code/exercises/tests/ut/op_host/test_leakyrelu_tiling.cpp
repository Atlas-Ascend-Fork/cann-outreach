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
    float negativeSlope;
    uint64_t expectTilingKey;
};

static LeakyreluTestParam testCases[] = {
    {"float", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND,
     0.01f, 1UL},
    {"half", {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND,
     -0.25f, 0UL},
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
    SCOPED_TRACE(param.caseName);
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr(
        "negative_slope", Ops::Math::AnyValue::CreateFrom<float>(param.negativeSlope)));
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        64,
        262144,
        4096);
    TilingInfo result;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, result));
    EXPECT_EQ(result.tilingKey, param.expectTilingKey);
    EXPECT_EQ(result.workspaceSizes, (std::vector<int64_t>{0}));
    EXPECT_EQ(result.blockNum, 16U);
    ASSERT_EQ(result.tilingDataSize, sizeof(LeakyreluTilingData));
    LeakyreluTilingData data;
    std::memcpy(&data, result.tilingData.get(), sizeof(data));
    EXPECT_EQ(data.totalNum, 16384);
    EXPECT_EQ(data.blockFactor, 1024);
    EXPECT_EQ(data.ubFactor, 1024);
    EXPECT_FLOAT_EQ(data.negativeSlope, param.negativeSlope);
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
