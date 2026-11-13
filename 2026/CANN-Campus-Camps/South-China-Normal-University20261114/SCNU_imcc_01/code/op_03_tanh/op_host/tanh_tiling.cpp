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

/* 【性能旋钮】每个 AI Core 至少负责的元素数。16384 / 2048 = 8 个核。
   调大 -> 核更少、每核块更大；调小 -> 核更多、每核块更小 */
constexpr int64_t MIN_ELE_PER_CORE = 2048;

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

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    /* 1. 输入总元素数：(8, 2048) -> 16384 */
    auto xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t totalNum = xShape->GetStorageShape().GetShapeSize();

    if (totalNum < 0) {
        totalNum = 0;
    }

    /* 2. 按 dtype 算 32B 对齐需要几个元素：float32 -> 8，float16 -> 16 */
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    int64_t typeSize = static_cast<int64_t>(ge::GetSizeByDataType(inputDesc->GetDataType()));
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "dtype is not supported"), return ge::GRAPH_FAILED);
    int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));   // 32 字节
    int64_t alignNum = ubBlockSize / typeSize;

    /* 3. 切分方案：先按“每核至少 MIN_ELE_PER_CORE 个元素”收缩核数，再算每核块大小
       16384 元素 -> 8 个核，每核正好 2048 个元素（一核一行，天然 32B 对齐） */
    int64_t blockFactor = 0;    // 每个核处理多少（32B 对齐）
    int64_t ubFactor = 0;       // 一次 UB 循环处理多少（32B 对齐）
    int64_t usedCoreNum = 1;    // 实际用几个核
    if (totalNum > 0) {
        /* 关键：不要一上来就按满核数切，小 shape 下核越多固定开销越大 */
        usedCoreNum = CeilDiv(totalNum, MIN_ELE_PER_CORE);
        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }

        blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);

        /* 每个元素占用的 UB 字节：输入 2 份 + 输出 2 份 + float 临时 2 份 */
        int64_t perElemBytes = 4 * typeSize + 2 * TYPE_SIZE;
        ubFactor = static_cast<int64_t>(ubSize) / perElemBytes;
        ubFactor = FloorAlign(ubFactor, alignNum);
        if (ubFactor > blockFactor) {
            ubFactor = blockFactor;
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