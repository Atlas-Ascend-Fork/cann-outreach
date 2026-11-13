/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

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
    // TODO: 实现 Tiling 逻辑
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

    // 计算输入总元素数（GetInputShape 返回 StorageShape，需再取 GetShape）
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    const gert::Shape inShape = EnsureNotScalar(inputStorageShape->GetShape());
    int64_t totalNum = inShape.GetShapeSize();

    // 输入 dtype：决定元素字节数（用于 UB 容量与对齐计算）
    auto inputDesc = context->GetInputDesc(0);
    int64_t typeSize = sizeof(float);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        typeSize = sizeof(uint16_t);
    }

    // 使用核数：min(平台AIV核数, 32, totalNum/4096)。
    // 实测核数过多（>32）时 GM/L2 带宽竞争与核启动开销超过并行收益，反而变慢，
    // 因此上限限制为 32；小 shape 再按每核至少 4096 个元素减少核数
    constexpr int64_t MIN_PER_CORE = 4096;
    constexpr int64_t MAX_BLOCK_DIM = 32;
    int64_t blockDim = (coreNum < MAX_BLOCK_DIM) ? coreNum : MAX_BLOCK_DIM;
    if (totalNum / MIN_PER_CORE < blockDim) {
        blockDim = totalNum / MIN_PER_CORE;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }
    // 每核元素数向下对齐到 32B（保证 DataCopy 对齐），最后一个核处理剩余元素
    int64_t alignNum = 32 / typeSize;
    int64_t blockFactor = (totalNum / blockDim / alignNum) * alignNum;
    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }

    // 每次 UB 循环处理的元素数（tileNum = blockFactor / ubFactor）：
    // 实验：单缓冲换取更大单块（fp32:64KB/块），大 shape 循环数从 16 降到 8；
    // 队列占用 = 2*BUFFER_NUM*ubFactor*typeSize = 128KB，处于 UB 容量安全范围内
    constexpr int64_t BUFFER_NUM = 1;  // 与 kernel 单缓冲一致
    constexpr int64_t UB_FACTOR_CAP = 28672;  // 单块元素上限
    int64_t maxUbFactor = static_cast<int64_t>(ubSize) / (2 * BUFFER_NUM * typeSize);
    int64_t ubFactor = std::min(blockFactor, std::min(maxUbFactor, UB_FACTOR_CAP));

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(blockDim);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
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