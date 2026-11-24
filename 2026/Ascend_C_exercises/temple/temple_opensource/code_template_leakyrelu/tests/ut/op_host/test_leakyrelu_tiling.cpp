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
    float negativeSlope; // 属性：负半轴斜率 α
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// tilingData 内存布局（按 int64 逐字比对）：{uint32 totalLength, uint32 tileNum, float negativeSlope}
// word0 = totalLength | (tileNum << 32) = 16384 | (8 << 32) = 34359754752
static LeakyreluTestParam testCases[] = {
    {"leakyrelu_0", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, 0.01f, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "34359754752 ", {0}, 64, 262144, 4096},
    {"leakyrelu_1", {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND, 0.125f, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "34359754752 ", {0}, 64, 262144, 4096},
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
    // 注入属性 negative_slope（本系列首个带属性算子，Tiling 内经 GetAttrPointer<float>(0) 读取）
    std::vector<gert::TilingContextPara::OpAttr> attrs_ = {
        gert::TilingContextPara::OpAttr("negative_slope", Ops::Math::AnyValue::CreateFrom<float>(param.negativeSlope))
    };

    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
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
