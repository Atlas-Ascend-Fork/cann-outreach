/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t TILE_LENGTH = 1024;
constexpr uint32_t BUFFER_NUM = 2;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 获取输入数据类型
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 获取输入 Shape 和 TilingData
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);

    if (tiling == nullptr || xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 计算总元素数量
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

        if (totalLength >
            std::numeric_limits<uint32_t>::max() /
            static_cast<uint64_t>(dim)) {
            return ge::GRAPH_FAILED;
        }

        totalLength *= static_cast<uint64_t>(dim);
    }

    // 根据输入规模确定计算核数量
    uint32_t blockDim = 1;

    if (totalLength > 0) {
        blockDim = totalLength < BLOCK_DIM
                       ? static_cast<uint32_t>(totalLength)
                       : BLOCK_DIM;
    }

    // 计算最大单核数据量
    uint64_t maxCoreLength =
        (totalLength + blockDim - 1) / blockDim;

    // 每次循环处理两个 1024 元素的数据块
    uint64_t pairLength = TILE_LENGTH * BUFFER_NUM;

    uint64_t tileNum = (maxCoreLength + pairLength - 1) / pairLength;

    // 设置 TilingData
    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->tileNum = static_cast<uint32_t>(tileNum);

    // 设置计算核数量
    context->SetBlockDim(blockDim);

    // 设置 workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);

    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling