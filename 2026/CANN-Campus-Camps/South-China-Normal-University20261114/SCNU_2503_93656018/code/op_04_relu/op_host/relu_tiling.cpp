/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(
    const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto platform =
        platform_ascendc::PlatformAscendC(platformInfoPtr);

    coreNum = platform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum == 0,
        OP_LOGE(context, "coreNum is 0"),
        return ge::GRAPH_FAILED);

    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "ubSize is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);

    workspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum)
            != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    auto* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputShape = context->GetInputShape(0);
    auto inputDesc = context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const auto dtype = inputDesc->GetDataType();

    OP_CHECK_IF(
        dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT,
        OP_LOGE(context, "Only float16 and float32 are supported"),
        return ge::GRAPH_FAILED);

    const int64_t totalNum =
        EnsureNotScalar(
            inputShape->GetStorageShape()).GetShapeSize();

    OP_CHECK_IF(
        totalNum < 0,
        OP_LOGE(context, "Invalid input shape"),
        return ge::GRAPH_FAILED);

    // 小样例保留上一版的分块策略。
    const int64_t splitThreshold =
        totalNum <= 16384
            ? int64_t{4096}
            : MIN_SPLIT_THRESHOLD;

    int64_t blockDim =
        totalNum == 0
            ? 1
            : CeilDiv(totalNum, splitThreshold);

    // 本轮唯一的调优因素：大输入使用约 75% 的 AIV 核。
    const int64_t activeCoreLimit =
        totalNum >= (int64_t{1} << 20)
            ? std::max(int64_t{1}, coreNum * 3 / 4)
            : coreNum;

    blockDim = std::min(blockDim, activeCoreLimit);

    // 保证两种 dtype 的每核起点满足 32 字节对齐。
    const int64_t blockFactor =
        totalNum == 0
            ? 16
            : CeilAlign(
                CeilDiv(totalNum, blockDim),
                int64_t{16});

    blockDim =
        totalNum == 0
            ? 1
            : CeilDiv(totalNum, blockFactor);

    const int64_t elementBytes =
        dtype == ge::DT_FLOAT16 ? 2 : 4;

    OP_CHECK_IF(
        ubSize <= 8192,
        OP_LOGE(context, "Insufficient UB"),
        return ge::GRAPH_FAILED);

    // 保留上一版的 UB 容量策略，预留 8 KiB。
    const int64_t singleLimit = FloorAlign(
        static_cast<int64_t>(
            (ubSize - 8192) / elementBytes),
        int64_t{64});

    const int64_t doubleLimit = FloorAlign(
        static_cast<int64_t>(
            (ubSize - 8192) / (2 * elementBytes)),
        int64_t{64});

    OP_CHECK_IF(
        doubleLimit == 0,
        OP_LOGE(context, "Insufficient UB"),
        return ge::GRAPH_FAILED);

    const int64_t alignedBlock =
        CeilAlign(blockFactor, int64_t{64});

    int64_t tileLength;

    if (alignedBlock <= singleLimit) {
        // 每核数据能一次装下，Kernel 只分配 ping。
        tileLength = alignedBlock;
    } else {
        // 装不下时按双缓冲容量分块。
        const int64_t tileNum =
            CeilDiv(blockFactor, doubleLimit);

        tileLength = CeilAlign(
            CeilDiv(blockFactor, tileNum),
            int64_t{64});
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = tileLength;

    context->GetRawTilingData()->SetDataSize(
        sizeof(ReluTilingData));

    context->SetBlockDim(
        static_cast<uint32_t>(blockDim));

    const uint64_t tilingKey =
        dtype == ge::DT_FLOAT16
            ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
            : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);

    context->SetTilingKey(tilingKey);

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