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

    // 1. 获取输入 shape 并计算总元素数
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& inShape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = inShape.GetShapeSize();
    if (totalNum <= 0) {
        totalNum = 1;
    }

    // 2. 获取数据类型大小
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    bool isFp16 = (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
                   inputDesc->GetDataType() == ge::DT_BF16);
    int64_t typeSize = isFp16 ? 2 : TYPE_SIZE;

    // 3. 32 字节对齐元素数
    int64_t alignNum = 32 / typeSize;
    if (alignNum <= 0) {
        alignNum = 1;
    }

    // 4. 配置启动核数量与每个核处理的元素数
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignNum);
    int64_t blockDim = CeilDiv(totalNum, blockFactor);

    // 5. 计算每次 UB 循环处理的元素数 ubFactor (启用 DoubleBuffer，即 BUFFER_NUM=2)
    constexpr int64_t BUFFER_NUM_TILING = 2;
    int64_t ubFactor = ubSize / (BUFFER_NUM_TILING * 2 * typeSize);
    ubFactor = FloorAlign(ubFactor, alignNum);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    if (ubFactor <= 0) {
        ubFactor = alignNum;
    }

    // 6. 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 7. 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (isFp16) {
        tilingKey = static_cast<uint64_t>(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        tilingKey = static_cast<uint64_t>(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
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
