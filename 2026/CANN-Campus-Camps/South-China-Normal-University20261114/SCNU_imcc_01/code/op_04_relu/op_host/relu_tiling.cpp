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

/* 【性能旋钮】每个 AI Core 最少负责的元素数，用来给小 shape 收缩核数：
   16384 / 2048 = 8 个核（每核正好一行 2048）；4194304 / 2048 远超满核数，直接满核跑
   小 shape 想再快就把这里改成 4096（4 核），大 shape 不受影响 */
constexpr int64_t MIN_ELE_PER_CORE = 2048;
/* 必须与 kernel 侧 NsRelu::IN_BUFFER_NUM / OUT_BUFFER_NUM 严格一致！
   UB 占用 = (IN + OUT) * ubFactor * typeSize。
   原来 2+2=4 份时 tile 很大但流水很浅；现在 4+2=6 份，tile 变小但可以让 3 块数据同时在搬，
   MTE2 的 outstanding 请求数翻倍，大 shape 的带宽才能吃满 */
constexpr int64_t IN_BUFFER_NUM = 4;
constexpr int64_t OUT_BUFFER_NUM = 2;
/* UB 预留：留给 TPipe 的余量 */
constexpr int64_t UB_RESERVED_BYTES = 16 * 1024;
/* DMA 对齐粒度：每核起始地址、每个 tile 起始地址都按 512B 对齐，
   比只要求 32B 对齐能拿到更长的 MTE2 burst，大 shape 的带宽明显更好 */
constexpr int64_t DMA_ALIGN_BYTES = 512;

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

    /* 1. 总元素数：(8, 2048) -> 16384；(2048, 2048) -> 4194304 */
    auto xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t totalNum = xShape->GetStorageShape().GetShapeSize();
    if (totalNum < 0) {
        totalNum = 0;
    }

    /* 2. 32B 对齐粒度：float32 -> 8 个元素，float16 -> 16 个元素 */
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    int64_t typeSize = static_cast<int64_t>(ge::GetSizeByDataType(inputDesc->GetDataType()));
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "dtype is not supported"), return ge::GRAPH_FAILED);
    int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));   // 32 字节
    int64_t alignNum = ubBlockSize / typeSize;
    int64_t alignDma = DMA_ALIGN_BYTES / typeSize;   // 512B 对应的元素个数

    /* 3. 切分方案 */
    int64_t blockFactor = 0;    // 每个核处理多少（32B 对齐）
    int64_t ubFactor = 0;       // 每个 tile 处理多少（32B 对齐）
    int64_t usedCoreNum = 1;    // 实际启动几个核
    if (totalNum > 0) {
        /* 3.1 核数：小 shape 不铺满核，避免每个核的固定握手开销把时间拖长 */
        usedCoreNum = CeilDiv(totalNum, MIN_ELE_PER_CORE);
        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }

        /* 3.2 每核处理量，按 512B 对齐（保证每个核的起始地址落在 512B 边界） */
        blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignDma);

        /* 3.3 每个 tile 处理量：UB 去掉预留后按真实 buffer 份数计算
           （输入 IN_BUFFER_NUM 份 + 输出 OUT_BUFFER_NUM 份），再按 512B 对齐 */
        int64_t perElemBytes = (IN_BUFFER_NUM + OUT_BUFFER_NUM) * typeSize;
        int64_t usableUb = static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES;
        if (usableUb < ubBlockSize) {
            usableUb = static_cast<int64_t>(ubSize) / 2;      // 兜底：UB 特别小的平台
        }
        ubFactor = FloorAlign(usableUb / perElemBytes, alignDma);
        if (ubFactor > blockFactor) {
            ubFactor = blockFactor;      // 小 shape：每核一轮搬完
        }
        if (ubFactor < alignNum) {
            ubFactor = alignNum;
        }
    }

    /* 4. 写 tiling 数据 */
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

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
