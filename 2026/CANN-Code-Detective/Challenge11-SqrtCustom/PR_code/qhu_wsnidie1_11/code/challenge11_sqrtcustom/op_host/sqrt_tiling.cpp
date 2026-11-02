/*!
 * \file sqrt_tiling.cpp
 * \brief Sqrt 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sqrt_tiling_data.h"
#include "../op_kernel/sqrt_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_DIM = 8;      // 启动核数
constexpr int64_t UB_ELEM_NUM = 8192;  // 单次 UB 循环元素数
constexpr int64_t ALIGN_NUM = 64;      // 32B 对齐的元素数

static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
{
    // 按输入 dtype 选择模板实例：float16 -> MODE_0，float32 -> MODE_1
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtype = inputDesc->GetDataType();
    uint32_t schMode = (dtype == ge::DT_FLOAT16) ? static_cast<uint32_t>(SQRT_TPL_SCH_MODE_0)
                                                 : static_cast<uint32_t>(SQRT_TPL_SCH_MODE_1);
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 输入 shape 逐维相乘得到总元素数
    const gert::StorageShape* xShape = context->GetInputShape(0);
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        totalNum *= xShape->GetStorageShape().GetDim(i);
    }

    // 多核切分：每核处理量按 32B 对齐；UB 内再按 ubFactor 分块
    SqrtTilingData* tiling = context->GetTilingData<SqrtTilingData>();
    int64_t blockFactor = (totalNum + BLOCK_DIM - 1) / BLOCK_DIM;
    blockFactor = (blockFactor + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
    int64_t ubFactor = blockFactor < UB_ELEM_NUM ? blockFactor : UB_ELEM_NUM;
    ubFactor = (ubFactor + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 配置启动核数
    context->SetBlockDim(BLOCK_DIM);

    // 配置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSqrt([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SqrtCompileInfo {};

IMPL_OP_OPTILING(Sqrt).Tiling(SqrtTilingFunc).TilingParse<SqrtCompileInfo>(TilingParseForSqrt);

} // namespace optiling
