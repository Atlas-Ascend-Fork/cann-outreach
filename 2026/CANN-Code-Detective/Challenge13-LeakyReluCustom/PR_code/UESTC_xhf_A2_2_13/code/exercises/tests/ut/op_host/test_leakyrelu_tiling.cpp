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
    float slope;
    int64_t tilingKey;
    size_t blockNum;
    LeakyreluTilingData expected;
};

static LeakyreluTestParam testCases[] = {
    {"aligned_f32", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT,
     ge::FORMAT_ND, 0.01f, 1, 16, {16384, 1024, 1024, 0.01f, 0}},
    {"tail_f32", {1, 1031}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1031}, ge::DT_FLOAT,
     ge::FORMAT_ND, 0.125f, 1, 2, {1031, 520, 520, 0.125f, 0}},
    {"tail_f16", {1, 1031}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 1031}, ge::DT_FLOAT16,
     ge::FORMAT_ND, 0.125f, 0, 2, {1031, 528, 528, 0.125f, 0}},
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
    attrs_.push_back(gert::TilingContextPara::OpAttr("negative_slope", Ops::Math::AnyValue::CreateFrom<float>(param.slope)));
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo, 64, 262144, 4096);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, info)) << param.caseName;
    EXPECT_EQ(info.tilingKey, param.tilingKey);
    EXPECT_EQ(info.blockNum, param.blockNum);
    EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>{0});
    ASSERT_EQ(info.tilingDataSize, sizeof(LeakyreluTilingData));
    LeakyreluTilingData actual;
    std::memcpy(&actual, info.tilingData.get(), sizeof(actual));
    EXPECT_EQ(actual.totalNum, param.expected.totalNum);
    EXPECT_EQ(actual.blockFactor, param.expected.blockFactor);
    EXPECT_EQ(actual.ubFactor, param.expected.ubFactor);
    EXPECT_FLOAT_EQ(actual.negativeSlope, param.expected.negativeSlope);
    EXPECT_EQ(actual.reserved, param.expected.reserved);
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
