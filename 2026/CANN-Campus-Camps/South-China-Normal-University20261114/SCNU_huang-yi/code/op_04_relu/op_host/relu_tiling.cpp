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
constexpr int32_t TILING_BUF_NUM = 2;  // 与 kernel 侧双缓冲保持一致

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

    // 1. 取输入 shape，计算总元素个数 totalNum
    //    链路：GetInputShape(0) -> const gert::StorageShape*；
    //          StorageShape::GetStorageShape() 返回 const gert::Shape&；
    //          再由 gert::Shape 的 GetDimNum()/GetDim() 取维度。
    const gert::StorageShape* storageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Shape& inShape = storageShape->GetStorageShape();
    int64_t totalNum = 1;
    for (size_t d = 0; d < inShape.GetDimNum(); ++d) {
        totalNum *= inShape.GetDim(d);
    }

    // 取输入描述符（仅用于读取 dtype）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    // 2. 按数据类型确定单元素字节数
    int64_t typeSize = 4;  // 默认 float32
    if (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
        inputDesc->GetDataType() == ge::DT_BF16) {
        typeSize = 2;
    }

    // 3. 多核切分：尽量用满 AIV 核，并保证 blockFactor 为整数（找 <= coreNum 的最大约数）
    int64_t blockDim = coreNum;
    if (blockDim > totalNum) {
        blockDim = totalNum;
    }
    while (blockDim > 1 && totalNum % blockDim != 0) {
        --blockDim;
    }
    int64_t blockFactor = totalNum / blockDim;

    // 4. UB 切分（双缓冲）：2 个队列(input+output) × BUFFER_NUM(2) 块 buffer。
    //    取能容纳的最大 2 的幂作为 ubFactor，保证循环无尾巴且 burst 对齐。
    //    乘以 3/4 作为安全余量，避免 UB 占用恰好 100% 边界。
    int64_t queueNum = 2;
    int64_t ubBudget = ((int64_t)ubSize * 3 / 4) / (TILING_BUF_NUM * typeSize * queueNum);
    int64_t ubFactor = 1;
    while (ubFactor * 2 <= ubBudget) {
        ubFactor *= 2;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;  // blockFactor 本身为 2 的幂
    }
    if (ubFactor < 1) {
        ubFactor = 1;
    }

    // TODO: 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim((uint32_t)blockDim);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
        inputDesc->GetDataType() == ge::DT_BF16) {
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
