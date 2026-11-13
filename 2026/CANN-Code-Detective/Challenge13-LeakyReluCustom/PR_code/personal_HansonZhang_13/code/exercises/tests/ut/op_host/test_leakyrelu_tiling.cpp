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

// TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新：
//   expectTilingKey:  参考 op_kernel/leakyrelu_tiling_key.h 和 op_host/leakyrelu_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/leakyrelu_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/leakyrelu_tiling.cpp 中 GetWorkspaceSize 的逻辑
//
// ★ 本期（第 13 期）按上面 TODO 的要求更新了 expectTilingData：
//   本实现在该用例（shape {8,2048}、DT_FLOAT、Ascend910B、maxAIVNum=64、ubSize=262144）
//   下写入的 tiling 数据是：
//     totalNum=16384、blockFactor=256、ubFactor=256、negativeSlope=0.01f
//   而 common/tiling_case_executor.cpp 按 int64 数组序列化整个 tiling buffer，
//   故期望字符串为 "16384 256 256 1008981770 "
//   （1008981770 = 0.01f 的 IEEE754 位模式 0x3C23D70A；
//    LeakyreluTilingData 末尾的 uint32_t reserved 显式写 0，保证该序列化结果确定）。
static LeakyreluTestParam testCases[] = {
    {"leakyrelu_0", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "16384 256 256 1008981770 ", {0}, 64, 262144, 4096},
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
