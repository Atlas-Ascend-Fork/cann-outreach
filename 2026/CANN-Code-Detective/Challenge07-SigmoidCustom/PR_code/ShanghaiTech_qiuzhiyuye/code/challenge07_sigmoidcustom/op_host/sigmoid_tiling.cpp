/*!
 * \file sigmoid_tiling.cpp
 * \brief Sigmoid 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sigmoid_tiling_data.h"
#include "../op_kernel/sigmoid_tiling_key.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
// Sigmoid 的 Exp 比简单逐元素算子更依赖并行度；512 个元素/核在小张量上
// 可以控制 block 开销，在题目默认的 16384 元素上仍能充分使用 AIV。
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

static ge::graphStatus SigmoidTilingFunc(gert::TilingContext* context)
{
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Sigmoid only supports float32 and float16"), return ge::GRAPH_FAILED);

    // 算子只做逐元素计算，因此把连续 tensor 视作一维数组切分。
    // 标量在 tiling 阶段临时按 1 个元素处理，不改变对外 shape。
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
    OP_CHECK_IF(rawTilingData->GetCapacity() < sizeof(SigmoidTilingData),
        OP_LOGE(context, "Tiling buffer is too small"), return ge::GRAPH_FAILED);

    SigmoidTilingData* tiling = context->GetTilingData<SigmoidTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? TYPE_SIZE / 2 : TYPE_SIZE;
    constexpr int64_t DMA_BLOCK_BYTES = 32;
    constexpr int64_t VECTOR_BYTES = 256;
    // 4 份 UB：输入/输出各 2 份。其余空间留给运行时和向量指令使用。
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

    // 小张量不盲目启动全部核；对本题 16384 个元素，默认使用最多 32 个 AIV，
    // 每个 block 至少有约 512 个元素，能兼顾 Exp 的并行度与 block 调度开销。
    const int64_t usefulCores = std::max<int64_t>(1, CeilDiv(totalNum, MIN_SPLIT_THRESHOLD));
    const int64_t usedCores = std::min(coreNum, usefulCores);
    const int64_t blockFactor = totalNum == 0
                                    ? alignNum
                                    : CeilAlign(CeilDiv(totalNum, usedCores), alignNum);
    const int64_t blockDim = totalNum == 0 ? 1 : CeilDiv(totalNum, blockFactor);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = std::min(blockFactor, maxUbNum);

    rawTilingData->SetDataSize(sizeof(SigmoidTilingData));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(SIGMOID_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(SIGMOID_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSigmoid([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SigmoidCompileInfo {};

IMPL_OP_OPTILING(Sigmoid).Tiling(SigmoidTilingFunc).TilingParse<SigmoidCompileInfo>(TilingParseForSigmoid);

} // namespace optiling


