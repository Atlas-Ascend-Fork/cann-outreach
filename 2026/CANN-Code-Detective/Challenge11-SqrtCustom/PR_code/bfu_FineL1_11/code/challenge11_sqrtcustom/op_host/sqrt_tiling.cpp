/*!
 * \file sqrt_tiling.cpp
 * \brief Sqrt 算子 Tiling 实现
 */

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sqrt_tiling_data.h"
#include "../op_kernel/sqrt_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t DATA_BLOCK_BYTES = 32;
// 本题总元素数为 16384。以每核约 256 个元素为目标，优先利用可用 AIV 核。
constexpr int64_t TARGET_ELEMS_PER_BLOCK = 256;

static const gert::Shape g_vec_1_shape = {1};

static inline gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(context, "coreNum must be positive"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "ubSize is zero"),
        return ge::GRAPH_FAILED);

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
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const gert::Shape normalizedShape =
        EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum = normalizedShape.GetShapeSize();

    const auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    int64_t typeSize = 0;
    uint64_t tilingKey = 0;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        typeSize = 2;
        tilingKey = GET_TPL_TILING_KEY(SQRT_TPL_SCH_MODE_0);
    } else if (inputDesc->GetDataType() == ge::DT_FLOAT) {
        typeSize = 4;
        tilingKey = GET_TPL_TILING_KEY(SQRT_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "Sqrt supports only float16 and float32");
        return ge::GRAPH_FAILED;
    }

    const int64_t alignElems = DATA_BLOCK_BYTES / typeSize;

    int64_t blockDim = 1;
    if (totalNum > 0) {
        const int64_t neededBlocks =
            (totalNum + TARGET_ELEMS_PER_BLOCK - 1) /
            TARGET_ELEMS_PER_BLOCK;
        blockDim = std::max<int64_t>(
            1, std::min<int64_t>(coreNum, neededBlocks));
    }

    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    if (totalNum > 0) {
        const int64_t elemsPerBlock =
            (totalNum + blockDim - 1) / blockDim;

        // 以 32-byte DataBlock 对齐，减少不对齐搬运。
        blockFactor =
            ((elemsPerBlock + alignElems - 1) / alignElems) * alignElems;

        // 两个队列分别存放输入和输出。最多使用约 1/2 UB，
        // 为队列及运行时开销预留空间。
        int64_t maxUbElems =
            static_cast<int64_t>(ubSize) / (4 * typeSize);
        maxUbElems = (maxUbElems / alignElems) * alignElems;
        if (maxUbElems < alignElems) {
            maxUbElems = alignElems;
        }
        ubFactor = std::min<int64_t>(blockFactor, maxUbElems);
    }

    SqrtTilingData* tiling = context->GetTilingData<SqrtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSqrt(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SqrtCompileInfo {};

IMPL_OP_OPTILING(Sqrt)
    .Tiling(SqrtTilingFunc)
    .TilingParse<SqrtCompileInfo>(TilingParseForSqrt);

} // namespace optiling
