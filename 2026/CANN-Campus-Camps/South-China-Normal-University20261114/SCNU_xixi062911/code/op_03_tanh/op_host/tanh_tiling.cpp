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

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// ---- 以下 UB 预算参数必须与 op_kernel/tanh.h 中的同名常量保持一致 ----
// 多轮循环时输入/输出队列的缓冲块数（双缓冲）
constexpr int64_t BUFFER_NUM = 2;
// 输入队列 + 输出队列
constexpr int64_t UB_QUEUE_NUM = 2;
// Tanh 临时空间：按元素占用的字节数（half 需 2 个 float 缓冲、float 需 1 个，取 8B 覆盖两种情况）
constexpr int64_t TMP_BYTES_PER_ELEM = 8;
// UB 预留（框架与对齐余量）
constexpr int64_t UB_RESERVED_BYTES = 8192;
// UB 数据块大小（字节），搬运/对齐的最小粒度
constexpr int64_t UB_BLOCK_SIZE = 32;

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

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
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

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 输入 dtype 决定字节数，同时决定 tilingKey（float16/bf16 -> half 分支，float -> float 分支）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    int64_t typeSize = TYPE_SIZE;
    bool isHalfType = false;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        typeSize = 2;
        isHalfType = true;
    }

    // 总元素数：按输入 shape 逐维相乘（0 维标量视为 1 个元素）
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t totalNum = EnsureNotScalar(xShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is 0"), return ge::GRAPH_FAILED);

    // 一个 UB block（32B）对应的元素个数
    int64_t alignNum = UB_BLOCK_SIZE / typeSize;
    OP_CHECK_IF(alignNum <= 0, OP_LOGE(context, "alignNum is 0"), return ge::GRAPH_FAILED);

    // 多核切分：每核负责 blockFactor 个元素（按 32B 对齐），尾核按剩余数据量截断
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignNum);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }

    // 单次 UB 循环处理的数据量：
    //   UB 需同时容纳 输入队列 + 输出队列（各 BUFFER_NUM 块） + 按元素计数的 Tanh 临时空间 + 预留
    int64_t perElemUbBytes = UB_QUEUE_NUM * BUFFER_NUM * typeSize + TMP_BYTES_PER_ELEM;
    OP_CHECK_IF(perElemUbBytes <= 0, OP_LOGE(context, "perElemUbBytes is 0"), return ge::GRAPH_FAILED);
    int64_t ubAvailable = static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES;
    int64_t ubFactor = 0;
    if (ubAvailable > 0) {
        ubFactor = FloorAlign(FloorDiv(ubAvailable, perElemUbBytes), alignNum);
    }
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }
    // 单核数据量小于 UB 容量时，一次搬完（kernel 会走单缓冲直通路径）
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    uint64_t tilingKey = isHalfType ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
                                    : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
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
