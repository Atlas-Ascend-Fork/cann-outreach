#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "sqrt_tiling_data.h"

namespace SqrtUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "Sqrt";

struct SqrtTestParam {
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
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// Tiling fields are totalNum, blockFactor and ubFactor (int64_t).
static SqrtTestParam testCases[] = {
    {"sqrt_0", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "16384 256 256 ", {0}, 64, 262144, 4096},
    {"sqrt_float16", {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "16384 256 256 ", {0}, 64, 262144, 4096},
};

class SqrtTilingTest : public testing::TestWithParam<SqrtTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "SqrtTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "SqrtTilingTest TearDown." << std::endl;
    }
};

struct SqrtCompileInfo {} compileInfo;

static void TestOneParamCase(const SqrtTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

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

TEST_P(SqrtTilingTest, tiling_test)
{
    const SqrtTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    SqrtTilingTests,
    SqrtTilingTest,
    testing::ValuesIn(testCases));

}
