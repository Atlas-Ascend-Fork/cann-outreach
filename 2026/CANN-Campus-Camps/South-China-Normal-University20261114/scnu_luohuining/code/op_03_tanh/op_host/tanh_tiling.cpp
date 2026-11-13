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
    // TODO: 实现 Tiling 逻辑
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

    // 1. 总元素数：shape 各维相乘（写法同 code_add代码参考/op_host/add_tiling.cpp:24-27）
    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    gert::Shape xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

    int64_t totalNum = 1;
    int64_t dimNum = static_cast<int64_t>(xShape.GetDimNum());
    for (int64_t i = 0; i < dimNum; i++) {
        totalNum *= xShape.GetDim(i);
    }

    // 2. 单元素字节数：fp16/bf16 为 2，其余为 4（题目要求支持 Float16）
    auto xInputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInputDesc);
    int64_t typeSize = (xInputDesc->GetDataType() == ge::DT_FLOAT16 ||
                        xInputDesc->GetDataType() == ge::DT_BF16) ? 2 : 4;

    // 32Byte 对齐：fp32 一行 8 个元素，fp16 一行 16 个元素
    constexpr int64_t UB_BLOCK_BYTES = 32;
    int64_t elemPerBlock = UB_BLOCK_BYTES / typeSize;

    // 3. 三级切分（默认值用于 totalNum == 0，避免除零）
    int64_t blockFactor = 1;
    int64_t ubFactor = 1;
    int64_t usedCoreNum = 1;
    if (totalNum > 0) {
        // 核间切分：向上取整保证尾部有归属，再按 32Byte 对齐
        blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), elemPerBlock);
        usedCoreNum = CeilDiv(totalNum, blockFactor);
        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }

        // 核内 UB 切分：2 个队列各 BUFFER_NUM(2) 块 + 3 个 float 临时块，共 7 块同时驻留
        constexpr int64_t BLOCK_PER_CORE = 2 * 2 + 3;
        int64_t ubBytesPerBlock = static_cast<int64_t>(ubSize) / BLOCK_PER_CORE;
        ubFactor = (ubBytesPerBlock / UB_BLOCK_BYTES) * UB_BLOCK_BYTES / typeSize;
        if (ubFactor > blockFactor) {
            ubFactor = blockFactor;
        }
        if (ubFactor < elemPerBlock) {
            ubFactor = elemPerBlock;
        }
    }

    // TODO: 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
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
