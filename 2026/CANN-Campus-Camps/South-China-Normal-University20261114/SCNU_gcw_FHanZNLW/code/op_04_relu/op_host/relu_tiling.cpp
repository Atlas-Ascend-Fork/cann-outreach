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
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;          // 按最大 4 字节类型估算 UB 空间
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t ALIGN_ELEMS = 32;       // 32 个元素对齐，保证 32B 访问

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

    // 计算输入总元素数；标量输入按 1 个元素处理
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    auto inputShape = EnsureNotScalar(inputStorageShape->GetStorageShape());
    int64_t totalNum = inputShape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum <= 0"), return ge::GRAPH_FAILED);
    tiling->totalNum = totalNum;

    // 分核：数据量较小时少用一些核，保证每个核至少处理 MIN_SPLIT_THRESHOLD 个元素
    int64_t blockDim = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }

    // 每个核处理的元素数按 32 个元素对齐，保证 GM 起始地址满足 32B 对齐
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), ALIGN_ELEMS);
    // 对齐后实际需要的核数可能变少，重新计算
    blockDim = CeilDiv(totalNum, blockFactor);
    tiling->blockFactor = blockFactor;

    // 每核内每次 UB 循环处理的元素数：
    // UB 同时容纳输入/输出两个队列，且每个队列使用 DoubleBuffer（BUFFER_NUM=2）
    int64_t ubFactor = static_cast<int64_t>(ubSize) / (2 * 2 * TYPE_SIZE);
    ubFactor = FloorAlign(ubFactor, ALIGN_ELEMS);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "ubFactor <= 0"), return ge::GRAPH_FAILED);
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 根据输入 dtype 选择 tilingKey：fp16/bf16 -> half 实例，其余 -> float 实例
    //
    //    GET_TPL_TILING_KEY 的入参必须是 relu_tiling_key.h 里 ASCENDC_TPL_DATATYPE_DECL
    //    声明的 C_DT_* 值（fp16 -> key 0，fp32 -> key 1），
    //    绝不能传 ge::DataType：ge::DT_FLOAT 的枚举值是 0，会命中 fp16 实例，
    //    float32 数据被当成 half 处理 —— 这正是之前 Wrong Answer(37.73%) 的根因。
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    if (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(C_DT_FLOAT16);
    } else {
        tilingKey = GET_TPL_TILING_KEY(C_DT_FLOAT);
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