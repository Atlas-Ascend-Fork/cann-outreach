/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include <cstdint>
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

const uint32_t TILE_NUM = 8;
const uint32_t BLOCK_DIM = 8;
const uint32_t BUFFER_NUM = 2;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // 获取输入数据类型
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 获取 TilingData 和输入 Shape
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);

    if (tiling == nullptr || xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 计算输入张量元素总数
    const auto& shape = xShape->GetStorageShape();
    uint64_t totalLength = 1;

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        int64_t dim = shape.GetDim(i);

        if (dim < 0) {
            return ge::GRAPH_FAILED;
        }

        if (dim == 0) {
            totalLength = 0;
            break;
        }

        if (totalLength > UINT32_MAX / static_cast<uint64_t>(dim)) {
            return ge::GRAPH_FAILED;
        }

        totalLength *= static_cast<uint64_t>(dim);
    }

    // 当前分块方案要求每个计算核分得的数据可以整除
    // tileNum * BUFFER_NUM
    if (totalLength == 0 || totalLength % BLOCK_DIM != 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t blockLength = static_cast<uint32_t>(totalLength / BLOCK_DIM);

    if (blockLength % (TILE_NUM * BUFFER_NUM) != 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t tileLength = blockLength / TILE_NUM / BUFFER_NUM;

    // 设置 TilingData
    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->tileNum = TILE_NUM;

    // 设置计算核数量
    context->SetBlockDim(BLOCK_DIM);

    // 根据输入类型获取元素字节数
    uint32_t typeSize = static_cast<uint32_t>(
        inputDesc->GetDataType() == ge::DT_FLOAT
            ? sizeof(float)
            : sizeof(uint16_t)
    );

    // 获取 Tanh 临时缓冲区大小
    std::vector<int64_t> tileDims = {
        static_cast<int64_t>(tileLength)
    };

    ge::Shape tileShape(tileDims);

    uint32_t maxTmpSize = 0;
    uint32_t minTmpSize = 0;

    AscendC::GetTanhMaxMinTmpSize(
        tileShape,
        typeSize,
        false,
        maxTmpSize,
        minTmpSize
    );

    // 采用最小必要临时缓冲区
    tiling->tmpBufferSize = minTmpSize;

    // 设置 workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);

    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh)
    .Tiling(TanhTilingFunc)
    .TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling