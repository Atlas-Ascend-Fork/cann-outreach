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

// kernel 侧 UB 占用：inputQueueX / outputQueueY 各 BUFFER_NUM(2) 块 = 4 块，
// 外加 exp(x)、exp(-x)、-x、分子、分母 5 个 TBuf，共 9 块 ubFactor * sizeof(T)。
constexpr int64_t TILING_BUF_NUM = 2;
constexpr int64_t TMP_BUF_NUM = 5;
constexpr int64_t TOTAL_BUF_BLOCKS = TILING_BUF_NUM * 2 + TMP_BUF_NUM; // = 9

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
    // 1. 平台信息
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

    // 2. 取输入 shape 并计算总元素数 totalNum
    //    链路：GetInputShape(0) -> const gert::StorageShape*；
    //          StorageShape::GetStorageShape() -> const gert::Shape&；
    //          gert::Shape 的 GetDimNum()/GetDim() 取各维。
    const gert::StorageShape* storageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Shape inShape = EnsureNotScalar(storageShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t d = 0; d < inShape.GetDimNum(); ++d) {
        totalNum *= inShape.GetDim(d);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is 0"), return ge::GRAPH_FAILED);

    // 3. 按 dtype 确定单元素字节数，并选取 tilingKey
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = TYPE_SIZE;   // 默认 float32
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        typeSize = 2;
    }

    // 4. 多核切分：取 <= 物理 AIV 核数且能整除 totalNum 的最大核数，
    //    保证 blockFactor 为整数、各核无重叠无遗漏。
    int64_t blockDim = coreNum;
    if (blockDim > totalNum) {
        blockDim = totalNum;
    }
    while (blockDim > 1 && (totalNum % blockDim) != 0) {
        --blockDim;
    }
    int64_t blockFactor = totalNum / blockDim;

    // 5. UB 切分：9 块 buffer，取能容纳的最大 2 的幂作为 ubFactor，
    //    乘以 3/4 作为安全余量，避免 UB 占用贴满边界。
    int64_t ubBudget = ((int64_t)ubSize * 3 / 4) / (TOTAL_BUF_BLOCKS * typeSize);
    int64_t ubFactor = 1;
    while (ubFactor * 2 <= ubBudget) {
        ubFactor *= 2;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    if (ubFactor < 1) {
        ubFactor = 1;
    }

    // TODO: 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim((uint32_t)blockDim);

    // 根据输入 dtype 选择 tilingKey（fp16/bf16 -> half 实例，其余 -> float 实例）
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
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
