/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;

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

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    // 1. 获取平台信息
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取 Workspace 大小（Exp 无需额外 workspace）
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 3. 获取输入 shape 和数据类型
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    // ★ 只接受 float32 / float16：bf16 与 fp16 的位布局不同，
    //   若把 bf16 当 fp16 处理会得到完全错误的数值，故显式拒绝（exp_def.cpp 亦只声明这两种）。
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "Exp only supports float32/float16"),
        return ge::GRAPH_FAILED);
    int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;

    // 4. 设置 Tiling 数据
    tiling->totalNum = totalNum;

    // 关键：blockFactor 与 ubFactor 都按 32 字节对齐，
    // 保证 DataCopy 的 DMA 粒度（32 字节）要求。
    const int64_t unit = 32 / typeSize; // 一个 32 字节对齐单元包含的元素数
    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    blockFactor = CeilAlign(blockFactor, unit); // 每个核处理的数据量是 32 字节的倍数
    if (blockFactor <= 0) {
        blockFactor = unit;
    }

    // ubFactor = 单个 tile 的元素个数。
    // ★ UB 预算必须与 kernel 侧 InitBuffer 的实际申请严格一致（见 op_kernel/exp.h）：
    //     float32: in(T) + out(T)                    = 2 * typeSize 字节/元素
    //     float16: in(T) + out(T) + floatBuf(float32) = 2 * typeSize + 4 字节/元素
    //   并预留少量余量，不要把 UB 吃满（与 Challenge06 已合入实现同一思路）。
    constexpr int64_t UB_RESERVED_BYTES = 1024;
    const int64_t ubBytesPerElem = 2 * typeSize + ((typeSize == 2) ? 4 : 0);
    int64_t maxTileElems = (static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES) / ubBytesPerElem;
    maxTileElems = (maxTileElems / unit) * unit;
    if (maxTileElems < unit) {
        maxTileElems = unit;
    }
    int64_t ubFactor = blockFactor;
    if (ubFactor > maxTileElems) {
        ubFactor = maxTileElems;
    }

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    // 显式声明写入长度，不依赖 GetTilingData<T>() 的隐式 SetDataSize
    // （与第八期已合入实现 tanh_tiling.cpp 的写法一致）。
    context->GetRawTilingData()->SetDataSize(sizeof(ExpTilingData));

    // 设置核数
    context->SetBlockDim(coreNum);

    // 5. 选择 tilingKey：0 -> float16，1 -> float32
    uint64_t tilingKey;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
