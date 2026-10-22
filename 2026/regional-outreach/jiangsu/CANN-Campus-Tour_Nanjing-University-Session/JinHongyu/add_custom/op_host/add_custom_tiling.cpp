/*!
 * \file add_custom_tiling.cpp
 * \brief add_custom 算子 Tiling 实现
 *
 * tiling 数据只传两个标量：
 *   - totalLength：总元素数（各维相乘）
 *   - tileLength ：单核内每块元素数（固定值）
 * 多核切分（余数分摊）与单核内分块、尾块处理均在 kernel 内完成，
 * 因此任意形状（含非对齐形状）都能被完整、正确地处理。
 */

#include "register/op_def_registry.h"

#include "../op_kernel/add_custom_tiling_data.h"
#include "../op_kernel/add_custom_tiling_key.h"

namespace optiling {

// 单核内每块元素数：float16 下每块 2KB，双缓冲 x/y/z 共 12KB，远小于 910B 单核 UB(256KB)
const uint32_t TILE_LENGTH = 1024;
// 启动核数（与 kernel 多核切分逻辑配套，kernel 侧用 GetBlockNum 自适应）
const uint32_t BLOCK_DIM = 8;

static ge::graphStatus AddCustomTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（本算子仅 float16 一个实例）
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 输入 shape 逐维相乘得到总元素数
    AddCustomTilingData* tiling = context->GetTilingData<AddCustomTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    for (size_t i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= static_cast<uint32_t>(xShape->GetStorageShape().GetDim(i));
    }
    tiling->totalLength = totalLength;
    tiling->tileLength = TILE_LENGTH;

    // 配置启动核数
    context->SetBlockDim(BLOCK_DIM);

    // 配置 workspace 大小（本算子无需 workspace）
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAddCustom([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AddCustomCompileInfo {};

IMPL_OP_OPTILING(AddCustom).Tiling(AddCustomTilingFunc).TilingParse<AddCustomCompileInfo>(TilingParseForAddCustom);

} // namespace optiling
