#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "exp_tiling_data.h"

namespace ExpUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "Exp";

struct ExpTestParam {
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

// 期望值对应 op_host/exp_tiling.cpp 的实现：
//   tilingData = {totalLength=16384, tileNum=1}，workspace = {0}
//   raw tiling 数据按 int64 序列化：16384 | (1 << 32) = 4294983680
//   expectTilingKey：ASCENDC_TPL_SEL_PARAM 按 dtype 值编码，DT_X=float(ge::DT_FLOAT=0) → 0
static ExpTestParam testCases[] = {
    {"exp_0", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "4294983680 ", {0}, 64, 262144, 4096},
};

class ExpTilingTest : public testing::TestWithParam<ExpTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "ExpTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "ExpTilingTest TearDown." << std::endl;
    }
};

struct ExpCompileInfo {} compileInfo;

static void TestOneParamCase(const ExpTestParam &param)
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

TEST_P(ExpTilingTest, tiling_test)
{
    const ExpTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    ExpTilingTests,
    ExpTilingTest,
    testing::ValuesIn(testCases));

}
