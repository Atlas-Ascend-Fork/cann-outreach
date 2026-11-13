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
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// 期望值说明（case: shape={8,2048}, dtype=DT_FLOAT, maxAIVNum=64, ubSize=262144）：
//   totalNum     = 8 * 2048 = 16384
//   blockDim     = min(64, CeilDiv(16384, 1024)) = 16
//   blockFactor  = CeilDiv(16384, 16) = 1024
//   ubFactor     = min(FloorAlign(262144/(2*4*2), 32/4), blockFactor) = 1024
//   tilingData 展开：int64 x 4（sizeof(LeakyreluTilingData)=32）
//       token0 = totalNum      = 16384
//       token1 = blockFactor   = 1024
//       token2 = ubFactor      = 1024
//       token3 = negativeSlope 位模式 | padding
//                0.01f = 0x3C23D70A，小端 int64 = 0x000000003C23D70A = 1008981770
//   expectTilingKey = GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_1)（DT_FLOAT → MODE_1）
//   expectWorkspaces = { WS_SYS_SIZE } = { 0 }
//
// 注：token3 的稳定性依赖 host tiling 在 GetTilingData 之后 memset 清 padding；
//     若未清零，padding 为堆上残留值，该期望值可能随机失配。
static LeakyreluTestParam testCases[] = {
    {"leakyrelu_0", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND,
     "Ascend910B", ge::GRAPH_SUCCESS, 1UL,
     "16384 1024 1024 1008981770 ",
     {0}, 64, 262144, 4096},
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
    attrs_.push_back(gert::TilingContextPara::OpAttr("negative_slope", Ops::Math::AnyValue::CreateFrom<float>(0.01)));
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
