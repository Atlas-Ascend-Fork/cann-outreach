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
constexpr int64_t UB_BLOCK_BYTES = 32;      // GM <-> UB 搬运的对齐粒度
constexpr int64_t UB_BUFFER_NUM = 4;        // 输入 2 + 输出 2，ReLU 无需计算临时空间
constexpr int64_t UB_RESERVE_BYTES = 8192;  // UB 预留
constexpr int64_t MAX_UB_ELEMS = 4096;      // 单次 UB 循环元素上限

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static inline int64_t DivUpLocal(int64_t value, int64_t factor)
{
    return (value + factor - 1) / factor;
}

static inline int64_t AlignUpLocal(int64_t value, int64_t factor)
{
    return DivUpLocal(value, factor) * factor;
}

static inline int64_t AlignDownLocal(int64_t value, int64_t factor)
{
    return (value / factor) * factor;
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

    // 1. 逐维相乘得到输入元素总数
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape shape = EnsureNotScalar(xShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalNum *= shape.GetDim(i);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    // 2. 依据 dtype 求元素字节数与对齐粒度（float -> 8 元素，float16 -> 16 元素）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : TYPE_SIZE;
    int64_t alignNum = UB_BLOCK_BYTES / typeSize;
    if (alignNum < 1) {
        alignNum = 1;
    }

    // 3. 单次 UB 循环元素数：按 UB 容量留出余量后向下对齐，并限制上限
    int64_t usableUb = static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES;
    if (usableUb < 0) {
        usableUb = 0;
    }
    int64_t maxElems = usableUb / (UB_BUFFER_NUM * typeSize);
    if (maxElems > MAX_UB_ELEMS) {
        maxElems = MAX_UB_ELEMS;
    }
    int64_t ubFactor = AlignDownLocal(maxElems, alignNum);
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    // 4. 多核切分：每核元素数取 ubFactor 的整数倍，并去掉尾部空核
    int64_t minPerCore = (MIN_SPLIT_THRESHOLD > ubFactor) ? MIN_SPLIT_THRESHOLD : ubFactor;
    int64_t needCore = DivUpLocal(totalNum, minPerCore);
    int64_t blockDim = (coreNum < needCore) ? coreNum : needCore;
    if (blockDim < 1) {
        blockDim = 1;
    }
    int64_t blockFactor = AlignUpLocal(DivUpLocal(totalNum, blockDim), ubFactor);
    blockDim = DivUpLocal(totalNum, blockFactor);
    if (blockDim < 1) {
        blockDim = 1;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 5. 依据 dtype 选择 kernel 实例：float16 -> schMode 0 (half)，float32 -> schMode 1 (float)
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
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
