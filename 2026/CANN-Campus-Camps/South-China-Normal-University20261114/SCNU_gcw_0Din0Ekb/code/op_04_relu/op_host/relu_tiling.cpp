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
constexpr int64_t MIN_ELEM_PER_CORE = 2048; // 每核最少处理的元素数：数据量小时少开核，省掉多余核的启动与同步开销

constexpr int64_t BUFFER_NUM = 2;          // 队列 buffer 块数，需与 op_kernel/relu.h 中 BUFFER_NUM 一致
constexpr int64_t UB_BLOCK_SIZE = 32;      // UB 最小访问粒度（字节）
constexpr int64_t UB_RESERVED_SIZE = 2048; // UB 预留空间，防止队列刚好占满整块 UB

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
    // 1. 平台信息：UB 容量与可用 AI Core 数量
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

    // 2. 输入 dtype 决定单元素字节数以及 kernel 模板分支（tilingKey）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    bool isHalf = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    int64_t typeSize = isHalf ? 2 : TYPE_SIZE;

    uint64_t tilingKey;
    if (isHalf) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);

    // 3. 输入总元素数（标量按 1 个元素处理）
    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    int64_t totalNum = EnsureNotScalar(xShapePtr->GetStorageShape()).GetShapeSize();

    if (totalNum <= 0) {
        // 空 tensor：只启动一个核，kernel 内直接跳过，不做任何切分
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 1;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    // 4. 多核切分：数据量很小时不拆分，避免多核启动开销；
    //    同时保证每个核至少有 MIN_ELEM_PER_CORE 个元素，数据量不大时不开满核，减少核启动与同步开销；
    //    blockFactor 向上取整并按 UB 块粒度对齐，尾核在 kernel 内按 totalNum 截断，保证不越界
    int64_t ubBlockElem = GetUbBlockSize(context) / typeSize; // 32B 对齐所需的元素数：fp32 为 8，fp16 为 16
    if (ubBlockElem < 1) {
        ubBlockElem = 1;
    }
    int64_t coreLimit = (totalNum < MIN_SPLIT_THRESHOLD) ? 1 : CeilDiv(totalNum, MIN_ELEM_PER_CORE);
    if (coreLimit > coreNum) {
        coreLimit = coreNum;
    }
    if (coreLimit < 1) {
        coreLimit = 1;
    }
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreLimit), ubBlockElem);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);

    // 5. UB 切分：每次搬入搬出的 UB 占用 = 输入队列 + 输出队列（均双缓冲）= BUFFER_NUM * 2 * typeSize 字节/元素；
    //    Relu 没有中间临时缓存，双缓冲即可让 DataCopy 与矢量计算重叠
    int64_t bytesPerElem = BUFFER_NUM * (typeSize + typeSize);
    int64_t ubAvailSize = static_cast<int64_t>(ubSize) - UB_RESERVED_SIZE;
    int64_t ubCap = (ubAvailSize > 0) ? (ubAvailSize / bytesPerElem) : 1;
    if (ubCap < 1) {
        ubCap = 1;
    }
    int64_t ubFactor = (ubCap > blockFactor) ? blockFactor : ubCap;
    if (ubFactor < blockFactor) {
        // 一次搬不完时，按“均分”重新切：避免最后一块只剩零头，各块长度尽量一致
        int64_t loopCount = CeilDiv(blockFactor, ubFactor);
        ubFactor = CeilAlign(CeilDiv(blockFactor, loopCount), ubBlockElem);
        if (ubFactor > ubCap) {
            ubFactor = FloorAlign(ubCap, ubBlockElem);
        }
    }
    ubFactor = FloorAlign(ubFactor, ubBlockElem);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    if (ubFactor < 1) {
        ubFactor = 1;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
