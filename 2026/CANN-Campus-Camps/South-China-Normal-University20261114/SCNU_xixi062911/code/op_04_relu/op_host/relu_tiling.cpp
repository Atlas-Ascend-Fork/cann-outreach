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
constexpr int64_t TYPE_SIZE = 4;      // float 每个元素的字节数
constexpr int64_t HALF_TYPE_SIZE = 2; // float16 每个元素的字节数

// ---- 以下 UB 预算参数必须与 op_kernel/relu.h 中的同名常量保持一致 ----
// 输入/输出队列各自的 buffer 块数（双缓冲流水）
constexpr int64_t BUFFER_NUM = 2;
// 输入队列 + 输出队列
constexpr int64_t UB_QUEUE_NUM = 2;
// UB 数据块大小（字节），搬运/对齐的最小粒度
constexpr int64_t UB_BLOCK_SIZE = 32;
// UB 预留（TPipe 管理与对齐余量）
constexpr int64_t UB_RESERVED_BYTES = 8192;
// 单次搬运/计算的数据量上限（字节）：把单条指令控制在 32KB，
// 既保证 DMA 搬出效率，又满足 Maxs 单次 calCount 上限（fp32 16320 / fp16 32640 个元素）
constexpr int64_t MAX_TILE_BYTES = 32768;
// 每个核至少处理的元素数：小 shape 时用它限制参与计算的核数，
// 避免出现「几十个 block 每个只搬 1KB」，把 block 级固定开销放大
constexpr int64_t MIN_ELEMS_PER_CORE = 2048;

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
    // Relu 是纯逐元素计算，不需要额外 workspace
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
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

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 1. 输入 dtype 决定字节数，同时决定 tilingKey（float16/bf16 -> half 分支，float -> float 分支）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    bool isHalfType = (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16);
    int64_t typeSize = isHalfType ? HALF_TYPE_SIZE : TYPE_SIZE;

    // 2. 总元素数：按输入 shape 逐维相乘（0 维标量视为 1 个元素）
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t totalNum = EnsureNotScalar(xShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be greater than 0"), return ge::GRAPH_FAILED);

    // 3. 一个 UB block（32B）对应的元素个数
    int64_t alignNum = UB_BLOCK_SIZE / typeSize;
    OP_CHECK_IF(alignNum <= 0, OP_LOGE(context, "alignNum is 0"), return ge::GRAPH_FAILED);

    // 4. 多核切分：数据量小时按 MIN_ELEMS_PER_CORE 限制核数，大 shape 用满平台 AIV 核数
    int64_t coreNumBySize = CeilDiv(totalNum, MIN_ELEMS_PER_CORE);
    if (coreNumBySize < 1) {
        coreNumBySize = 1;
    }
    if (coreNumBySize > coreNum) {
        coreNumBySize = coreNum;
    }
    // 每核负责 blockFactor 个元素（32B 对齐），最后一个核不足 blockFactor 时由 kernel 按实际长度截断
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNumBySize), alignNum);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    if (usedCoreNum > coreNumBySize) {
        usedCoreNum = coreNumBySize;
    }

    // 5. 单次 UB 循环处理的数据量：UB 需同时容纳 输入队列 + 输出队列（各 BUFFER_NUM 块）
    //    Relu 只需要一条矢量指令，没有 Cast/临时 buffer 的额外开销，UB 全部给数据用
    int64_t perElemUbBytes = UB_QUEUE_NUM * BUFFER_NUM * typeSize;
    OP_CHECK_IF(perElemUbBytes <= 0, OP_LOGE(context, "perElemUbBytes is 0"), return ge::GRAPH_FAILED);
    int64_t ubAvailable = static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES;
    int64_t ubFactor = 0;
    if (ubAvailable > 0) {
        ubFactor = FloorAlign(FloorDiv(ubAvailable, perElemUbBytes), alignNum);
    }
    // 限制单块数据量，保证单条搬运/矢量指令规模可控
    int64_t maxTileElems = FloorAlign(MAX_TILE_BYTES / typeSize, alignNum);
    if (maxTileElems < alignNum) {
        maxTileElems = alignNum;
    }
    if (ubFactor > maxTileElems) {
        ubFactor = maxTileElems;
    }
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }
    // 单核数据量小于 UB 容量时一次搬完（kernel 走单缓冲直通路径）
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    // 6. 写回 TilingData
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 7. 配置启动核数：全部分块一次性下发，各核并行搬运/计算
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 8. 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey = isHalfType ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
                                    : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
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
