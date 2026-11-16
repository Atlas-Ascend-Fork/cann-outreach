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
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int32_t KERNEL_BUFFER_NUM = 2;        // 与 kernel 中双缓冲块数保持一致
constexpr int64_t RESERVE_UB_BYTES = 32 * 1024; // UB 系统预留空间
constexpr int64_t DATA_COPY_ALIGN_BYTES = 32;   // 数据搬移 32 字节对齐

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

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
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

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    // 输入 shape 逐维相乘得到总元素数
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    int64_t totalNum = 1;
    if (inputShape != nullptr) {
        const gert::Shape& storageShape = inputShape->GetStorageShape();
        for (size_t i = 0; i < storageShape.GetDimNum(); i++) {
            totalNum *= storageShape.GetDim(i);
        }
    }

    // 按元素字节数得到 32B 对齐元素数（float/half 分别为 8 和 16）
    int64_t dataSize =
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) ? 2 : 4;
    int64_t alignUnit = DATA_COPY_ALIGN_BYTES / dataSize;

    // 配置启动核数量：每核至少处理 alignUnit 个元素
    int64_t maxBlockDim = totalNum / alignUnit;
    if (maxBlockDim < 1) {
        maxBlockDim = 1;
    }
    int64_t blockDim = std::min<int64_t>(coreNum, maxBlockDim);
    if (blockDim <= 0) {
        blockDim = 1;
    }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 每个核处理的基础元素数，向下对齐到 alignUnit，
    // 未被整除的余量统一交给 0 号核处理，保证所有核的拷贝长度均 32B 对齐
    int64_t blockFactor = (totalNum / blockDim) / alignUnit * alignUnit;
    if (blockFactor <= 0) {
        blockFactor = alignUnit;
    }
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;

    // 每次 UB 循环处理的元素数：按双缓冲（2 队列 × 2 块）计算单个 buffer 容量并向下对齐
    int64_t usableUbBytes = static_cast<int64_t>(ubSize) - RESERVE_UB_BYTES;
    int64_t ubFactor = usableUbBytes / (2 * KERNEL_BUFFER_NUM * dataSize);
    ubFactor = (ubFactor / alignUnit) * alignUnit;
    if (ubFactor <= 0) {
        ubFactor = alignUnit;
    }
    tiling->ubFactor = ubFactor;

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
