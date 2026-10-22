/*!
 * \file test_add_custom_tiling.cpp
 * \brief add_custom 算子 host 侧 tiling UT
 *
 * 期望值对应 op_host/add_custom_tiling.cpp 的实现：
 *   tilingData = {totalLength, tileLength=1024}，workspace = {0}
 *   raw tiling 数据按 int64 序列化：totalLength | (tileLength << 32)
 *   tilingKey：模板仅 float16 一个 SEL 实例 -> 0
 */
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "add_custom_tiling_data.h"

namespace AddCustomUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "AddCustom";

struct AddCustomTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::initializer_list<int64_t> zShape;
    ge::DataType zDtype;
    ge::Format zFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// tileLength 固定 1024：
//   [8, 2048]  => totalLength = 16384 => 16384 + (1024 << 32) = 4398046527488
//   [3, 100]   => totalLength = 300   => 300   + (1024 << 32) = 4398046511404
static AddCustomTestParam testCases[] = {
    // 对齐形状
    {"add_custom_aligned_8x2048", {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "4398046527488 ", {0}, 64, 262144, 4096},
    // 非对齐形状（跨核余数）
    {"add_custom_unaligned_3x100", {3, 100}, ge::DT_FLOAT16, ge::FORMAT_ND, {3, 100}, ge::DT_FLOAT16, ge::FORMAT_ND, {3, 100}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "4398046511404 ", {0}, 64, 262144, 4096},
};

class AddCustomTilingTest : public testing::TestWithParam<AddCustomTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "AddCustomTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "AddCustomTilingTest TearDown." << std::endl;
    }
};

struct AddCustomCompileInfo {} compileInfo;

static void TestOneParamCase(const AddCustomTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    gert::StorageShape zShape = {param.zShape, param.zShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat}, {yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{zShape, param.zDtype, param.zFormat}});
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

TEST_P(AddCustomTilingTest, tiling_test)
{
    const AddCustomTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    AddCustomTilingTests,
    AddCustomTilingTest,
    testing::ValuesIn(testCases));

} // namespace AddCustomUT
