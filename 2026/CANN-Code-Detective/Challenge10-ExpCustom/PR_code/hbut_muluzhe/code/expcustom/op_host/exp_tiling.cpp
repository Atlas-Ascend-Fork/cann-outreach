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
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t UB_ALIGN_BYTES = 32;       // DataCopy 32B 对齐粒度
constexpr int64_t UB_BUFFER_NUM = 4;         // UB 预留份数：输入/输出各一份，余量给对齐与流水
constexpr int64_t MIN_ELEMS_PER_CORE = 2048; // 每核最少处理元素数，小 shape 避免过度分核

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    // 平台信息：AIV 核数、UB 大小
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    // 输入 dtype：决定模板实例与元素字节数
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    int64_t typeSize = ge::GetSizeByDataType(dataType);
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "unsupported data type"), return ge::GRAPH_FAILED);

    // 输入 shape 逐维相乘得到总元素数
    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape& xShape = xStorageShape->GetStorageShape();
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        totalNum *= xShape.GetDim(i);
    }

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 多核均分：每个核处理的元素数（尾核不足时在 kernel 内截断）
    // blockFactor 按 32B 向上对齐，保证每个核的 GM 起始地址满足 DataCopy 对齐要求
    int64_t alignElems = UB_ALIGN_BYTES / typeSize;
    int64_t blockNum = (totalNum > 0) ? CeilDiv(totalNum, MIN_ELEMS_PER_CORE) : 1;
    if (blockNum > coreNum) {
        blockNum = coreNum;
    }
    if (blockNum < 1) {
        blockNum = 1;
    }
    int64_t blockFactor = (totalNum > 0) ? CeilDiv(totalNum, blockNum) : 1;
    blockFactor = CeilAlign(blockFactor, alignElems);

    // 单次 UB 循环元素数：输入/输出各占一份 buffer，并预留对齐与流水余量
    int64_t maxUbElems = (static_cast<int64_t>(ubSize) / UB_BUFFER_NUM) / typeSize;
    if (maxUbElems > alignElems) {
        maxUbElems = FloorAlign(maxUbElems, alignElems);
    }
    int64_t ubFactor = (maxUbElems < blockFactor) ? maxUbElems : blockFactor;
    ubFactor = FloorAlign(ubFactor, alignElems);
    if (ubFactor <= 0) {
        ubFactor = alignElems;
    }

    // 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 实际启动核数：数据量不足时不需要全部核参与
    int64_t blockDim = CeilDiv(totalNum, blockFactor);
    if (blockDim < 1) {
        blockDim = 1;
    }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 按输入 dtype 选择模板实例：float -> schMode 1，float16 -> schMode 0
    uint64_t tilingKey = (dataType == ge::DT_FLOAT) ? GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1)
                                                    : GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);

    // 配置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
