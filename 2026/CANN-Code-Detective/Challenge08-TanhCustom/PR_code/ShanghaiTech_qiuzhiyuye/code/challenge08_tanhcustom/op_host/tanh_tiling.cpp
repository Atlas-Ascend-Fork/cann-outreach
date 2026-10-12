/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
// Tanh 的 Exp 路径需要一定的向量计算量；512 个元素/核在题目默认的
// 16384 元素场景下可以提高并行度，同时避免小张量产生过多 block 调度开销。
constexpr int64_t MIN_SPLIT_THRESHOLD = 512;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

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

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // CANN 9.0 中 TilingContext::GetInputShape 返回 StorageShape 指针，
    // 使用 auto 保持与当前 Ascend C host API 的实际类型一致。
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Tanh only supports float32 and float16"), return ge::GRAPH_FAILED);

    // Tanh 是逐元素算子，因此将连续 tensor 视作一维数组切分。
    const gert::Shape shape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim < 0 ||
                        (dim > 0 && totalNum > std::numeric_limits<int64_t>::max() / dim),
                    OP_LOGE(context, "Invalid input shape"), return ge::GRAPH_FAILED);
        totalNum *= dim;
    }

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

    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < sizeof(TanhTilingData),
        OP_LOGE(context, "Tiling buffer is too small"), return ge::GRAPH_FAILED);

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? TYPE_SIZE / 2 : TYPE_SIZE;
    constexpr int64_t DMA_BLOCK_BYTES = 32;
    constexpr int64_t VECTOR_BYTES = 256;
    // 输入/输出各两份 UB，剩余空间预留给运行时和向量指令。
    constexpr uint64_t UB_RESERVED_BYTES = 1024;
    constexpr int64_t BUFFER_COUNT = 4;
    const int64_t alignNum = DMA_BLOCK_BYTES / typeSize;
    const int64_t vectorNum = VECTOR_BYTES / typeSize;

    OP_CHECK_IF(ubSize <= UB_RESERVED_BYTES ||
                    (ubSize - UB_RESERVED_BYTES) < static_cast<uint64_t>(BUFFER_COUNT * typeSize),
                OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);

    const uint64_t ubCapacity = (ubSize - UB_RESERVED_BYTES) /
                                static_cast<uint64_t>(BUFFER_COUNT * typeSize);
    const int64_t maxUbNum = FloorAlign(
        static_cast<int64_t>(std::min<uint64_t>(ubCapacity, 255ULL * vectorNum)), vectorNum);
    OP_CHECK_IF(maxUbNum <= 0, OP_LOGE(context, "UB tile size is zero"), return ge::GRAPH_FAILED);

    const int64_t usefulCores = std::max<int64_t>(1, CeilDiv(totalNum, MIN_SPLIT_THRESHOLD));
    const int64_t usedCores = std::min(coreNum, usefulCores);
    const int64_t blockFactor = totalNum == 0
                                    ? alignNum
                                    : CeilAlign(CeilDiv(totalNum, usedCores), alignNum);
    const int64_t blockDim = totalNum == 0 ? 1 : CeilDiv(totalNum, blockFactor);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = std::min(blockFactor, maxUbNum);

    rawTilingData->SetDataSize(sizeof(TanhTilingData));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
