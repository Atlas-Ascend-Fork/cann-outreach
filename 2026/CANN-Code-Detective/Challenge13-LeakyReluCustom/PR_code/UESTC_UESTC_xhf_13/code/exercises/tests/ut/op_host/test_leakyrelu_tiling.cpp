#include <iostream>
#include <cstring>
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
    ge::graphStatus status;
    uint64_t expectTilingKey;
    int64_t expectTotal;
    int64_t expectBlock;
    int64_t expectUb;
    float slope;
    std::vector<int64_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新：
//   expectTilingKey:  参考 op_kernel/leakyrelu_tiling_key.h 和 op_host/leakyrelu_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/leakyrelu_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/leakyrelu_tiling.cpp 中 GetWorkspaceSize 的逻辑
static LeakyreluTestParam testCases[] = {
    {"leakyrelu_0", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND,
        "Ascend910B", ge::GRAPH_SUCCESS, 1UL, 16384, 256, 256, 0.01f, {0}, 64, 262144, 4096},
    {"leakyrelu_half_tail", {17}, ge::DT_FLOAT16, ge::FORMAT_ND, {17}, ge::DT_FLOAT16, ge::FORMAT_ND,
        "Ascend910B", ge::GRAPH_SUCCESS, 0UL, 17, 16, 16, 0.2f, {0}, 64, 262144, 4096},
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
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    TilingInfo info;
    ASSERT_EQ(ExecuteTiling(tilingContextPara, info), param.status == ge::GRAPH_SUCCESS);
    ASSERT_NE(info.tilingData, nullptr);
    ASSERT_GE(info.tilingDataSize, sizeof(LeakyreluTilingData));
    LeakyreluTilingData data;
    std::memcpy(&data, info.tilingData.get(), sizeof(data));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(param.expectTilingKey));
    EXPECT_EQ(info.workspaceSizes, param.expectWorkspaces);
    EXPECT_EQ(data.totalNum, param.expectTotal);
    EXPECT_EQ(data.blockFactor, param.expectBlock);
    EXPECT_EQ(data.ubFactor, param.expectUb);
    EXPECT_FLOAT_EQ(data.negativeSlope, param.slope);
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
