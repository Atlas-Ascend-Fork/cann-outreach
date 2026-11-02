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

// Ops::Base utilities used implicitly via math_util.h

constexpr uint32_t WS_SYS_SIZE = 0U;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    SqrtTilingData* tiling = context->GetTilingData<SqrtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 计算总元素数
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    // 获取数据类型大小
    auto desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    int64_t typeSize = (desc->GetDataType() == ge::DT_FLOAT16 || desc->GetDataType() == ge::DT_BF16) ? 2 : 4;

    // 32B 对齐的元素数
    int64_t alignNum = 32 / typeSize;

    // 计算可用 UB 大小（预留部分空间给系统）
    int64_t ubAvailable = static_cast<int64_t>(ubSize) - 8 * 1024;
    OP_CHECK_IF(ubAvailable <= 0, OP_LOGE(context, "ubAvailable is invalid"), return ge::GRAPH_FAILED);

    // 单次 UB 循环可处理的最大元素数（double buffer: 输入+输出各 2 份）
    int64_t ubFactor = ubAvailable / (typeSize * 2 * 2);
    ubFactor = ubFactor / alignNum * alignNum;
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "ubFactor is invalid"), return ge::GRAPH_FAILED);

    int64_t blockDim = 0;
    int64_t blockFactor = 0;

    // 获取 shape 最后一维（连续维度）
    int64_t lastDim = shape.GetDim(shape.GetDimNum() - 1);
    int64_t firstDims = totalNum / lastDim;  // 前面所有维度的乘积

    if (typeSize == 2) {
        // fp16/bf16 优化策略
        if (lastDim >= 2048 && firstDims >= 2 && firstDims <= coreNum) {
            // 最后一维足够大且有多行，按行分核：每核处理一行
            blockDim = firstDims;
            blockFactor = lastDim;
            // ubFactor 不能超过 UB 容量限制，避免单行过长时 InitBuffer 失败
            ubFactor = (ubFactor < blockFactor) ? ubFactor : blockFactor;
        } else if (totalNum <= 32768) {
            // 小数据量：减少核数，每核至少 2048 元素，但核数不少于 2
            int64_t minPerCore = 2048;
            blockDim = (totalNum + minPerCore - 1) / minPerCore;
            if (blockDim > coreNum) blockDim = coreNum;
            if (blockDim < 2) blockDim = 2;  // 至少 2 核避免单核串行
            blockFactor = (totalNum + blockDim - 1) / blockDim;
            blockFactor = (blockFactor + alignNum - 1) / alignNum * alignNum;
            blockDim = (totalNum + blockFactor - 1) / blockFactor;
            if (ubFactor > blockFactor) {
                ubFactor = blockFactor;
            }
        } else {
            // 大数据量 fp16：满核并行
            blockDim = coreNum;
            blockFactor = (totalNum + blockDim - 1) / blockDim;
            blockFactor = (blockFactor + alignNum - 1) / alignNum * alignNum;
            blockDim = (totalNum + blockFactor - 1) / blockFactor;
            if (ubFactor > blockFactor) {
                ubFactor = blockFactor;
            }
        }
    } else {
        // fp32 路径：满核并行
        blockDim = coreNum;
        if (blockDim > totalNum / alignNum) {
            blockDim = (totalNum + alignNum - 1) / alignNum;
            if (blockDim < 1) blockDim = 1;
        }
        blockFactor = (totalNum + blockDim - 1) / blockDim;
        blockFactor = (blockFactor + alignNum - 1) / alignNum * alignNum;
        blockDim = (totalNum + blockFactor - 1) / blockFactor;
        if (ubFactor > blockFactor) {
            ubFactor = blockFactor;
        }
    }

    OP_CHECK_IF(blockDim <= 0, OP_LOGE(context, "blockDim is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockFactor <= 0, OP_LOGE(context, "blockFactor is invalid"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(SQRT_CUSTOM_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(SQRT_CUSTOM_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSqrt([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SqrtCompileInfo {};

IMPL_OP_OPTILING(Sqrt).Tiling(SqrtTilingFunc).TilingParse<SqrtCompileInfo>(TilingParseForSqrt);

} // namespace optiling
