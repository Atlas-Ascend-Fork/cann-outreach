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

    // ------------------------------------------------------------------
    // 1. 输入总元素数
    // ------------------------------------------------------------------
    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    gert::Shape xShape = EnsureNotScalar(xShapePtr->GetStorageShape());
    int64_t totalNum = xShape.GetShapeSize();
    if (totalNum < 0) {
        totalNum = 0;
    }

    // ------------------------------------------------------------------
    // 2. 输入 dtype：fp16/bf16 走 mode0（half 实例），fp32 走 mode1（float 实例）
    // ------------------------------------------------------------------
    ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    bool isFp16 = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    int64_t typeSize = isFp16 ? 2 : 4;

    // ------------------------------------------------------------------
    // 3. 切分参数
    //    blockFactor：每个核处理的元素数（向上取整，尾差由最后一个核承担）
    //    ubFactor   ：每次 UB 循环处理的元素数，需 32B 对齐
    // ------------------------------------------------------------------
    int64_t blockDim = coreNum;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;

    if (totalNum > 0) {
        // 每元素占用 UB 字节数：
        //   输入双 buffer + 输出双 buffer = 4*typeSize
        //   3 份 fp32 临时缓存 = 3*4
        //   fp16 再多 1 份 fp32 升精度缓存 = 4
        int64_t ubBytesPerElem = 4 * typeSize + 3 * TYPE_SIZE + (isFp16 ? TYPE_SIZE : 0);
        // 32B 对齐的元素个数：fp32 为 8，fp16 为 16
        int64_t alignElem = (TYPE_SIZE * 8) / typeSize;

        ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize), ubBytesPerElem), alignElem);
        if (ubFactor < alignElem) {
            ubFactor = alignElem;
        }

        blockFactor = CeilDiv(totalNum, blockDim);

        // 数据量很小的时候按 blockFactor 收缩核数，避免大量空核
        if (blockFactor < MIN_SPLIT_THRESHOLD) {
            int64_t usedCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
            blockDim = usedCoreNum > coreNum ? coreNum : usedCoreNum;
            if (blockDim < 1) {
                blockDim = 1;
            }
            blockFactor = CeilDiv(totalNum, blockDim);
        }
    } else {
        blockDim = 1;
        blockFactor = 0;
        ubFactor = 0;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // ------------------------------------------------------------------
    // 4. 根据输入 dtype 选择 tilingKey（与 kernel 的 schMode 模板参数对应）
    // ------------------------------------------------------------------
    uint64_t tilingKey;
    if (isFp16) {
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
