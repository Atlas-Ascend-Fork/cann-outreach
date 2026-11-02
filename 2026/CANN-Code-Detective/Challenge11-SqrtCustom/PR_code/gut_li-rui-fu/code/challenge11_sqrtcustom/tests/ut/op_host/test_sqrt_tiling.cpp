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

/* TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新：
//   expectTilingKey:  参考 op_kernel/sqrt_tiling_key.h 和 op_host/sqrt_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/sqrt_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/sqrt_tiling.cpp 中 GetWorkspaceSize 的逻辑

static SqrtTestParam testCases[] = {
    {"sqrt_0", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};*/

// 期望值来源（与 op_host/sqrt_tiling.cpp 的赋值逐项对应）：
//   tilingKey   : fp32 走 SQRT_TPL_SCH_MODE_1，直接由 tiling_key.h 编码得出，避免手写魔数
//   tilingData  : totalNum   = 8*2048 = 16384
//                 blockFactor = CeilAlign(CeilDiv(16384, coreLimit=8), align=8) = 2048
//                 ubFactor    = min(UB 容量切分 16384, align(blockFactor)) = 2048   （本用例 ubSize=262144）
//   顺序即 SqrtTilingData 的字段顺序，按 int64 拼接并补空格（见 tiling_case_executor.cpp to_string）
//   workspace   : GetWorkspaceSize 写的是 WS_SYS_SIZE（当前为 0），故期望 {0}
static const uint64_t TILING_KEY_FP32 = GET_TPL_TILING_KEY(SQRT_TPL_SCH_MODE_1);

static SqrtTestParam testCases[] = {
    {"sqrt_0", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B",
     ge::GRAPH_SUCCESS, TILING_KEY_FP32, "16384 2048 2048 ", {0}, 64, 262144, 4096},
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
