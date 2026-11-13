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

constexpr uint32_t WS_SYS_SIZE = 0U;

// 512Byte 对齐：源地址与块长落在 512B 边界时 DMA 吞吐最高
// 依据：cannbot-skills/.../memory_coalescing.md "ensure source alignment" + 08.04 课后题 A
constexpr int64_t UB_ALIGN_BYTES = 512;

// 单次搬运粒度下限：不足 16KB 时少用核反而更快（08.04 §4.1）
constexpr int64_t BYTES_PER_CORE_MIN = 16 * 1024;

// 单缓冲：把 UB 全给“单次搬运量”，把每核串行轮数从 11 砍到 5
// 必须与 op_kernel/relu.h 的 BUFFER_NUM 一致
constexpr int64_t BUFFER_NUM_TILING = 1;

static const gert::Shape g_vec_1_shape = {1};

// 0 维张量按 1 个元素处理，避免 totalNum 被算成 0
static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

// 平台约束：可用 AIV（Vector）核数、单核 UB 字节数
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

// 纯逐元素算子不需要用户 workspace，只占 1 个槽位置 0
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
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

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 1. 总元素数：(8,2048) 与 (2048,2048) 同一条路径
    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    gert::Shape xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

    int64_t totalNum = 1;
    int64_t dimNum = static_cast<int64_t>(xShape.GetDimNum());
    for (int64_t i = 0; i < dimNum; i++) {
        totalNum *= xShape.GetDim(i);
    }

    // 2. 单元素字节数：fp16/bf16 为 2，其余为 4
    auto xInputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInputDesc);
    int64_t typeSize = (xInputDesc->GetDataType() == ge::DT_FLOAT16 ||
                        xInputDesc->GetDataType() == ge::DT_BF16) ? 2 : 4;

    // 512B 对齐的元素粒度：fp32 = 128、fp16 = 256
    int64_t elemPerAlign = UB_ALIGN_BYTES / typeSize;
    if (elemPerAlign < 1) {
        elemPerAlign = 1;
    }

    // 3. 核数：先按 16KB 粒度反推上限，再受平台核数约束
    int64_t coreNumByBytes = totalNum * typeSize / BYTES_PER_CORE_MIN;
    if (coreNumByBytes < 1) {
        coreNumByBytes = 1;
    }
    if (coreNum > coreNumByBytes) {
        coreNum = coreNumByBytes;
    }

    // 4. 核间切分：blockFactor 512B 对齐，保证每核 GM 起点对齐
    int64_t blockFactor = 1;
    int64_t ubFactor = 1;
    int64_t usedCoreNum = 1;
    if (totalNum > 0) {
        blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), elemPerAlign);
        usedCoreNum = CeilDiv(totalNum, blockFactor);
        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }

        // 5. 核内 UB 切分：单缓冲，2 队列 × 1 块 = 2 块 → 单块约 96KB
        //    目标：把每核 11 轮串行压到 5 轮（08.04 §4.1 大块搬运）
        constexpr int64_t BLOCK_PER_CORE = 2 * BUFFER_NUM_TILING;
        int64_t ubBytesPerBlock = static_cast<int64_t>(ubSize) / BLOCK_PER_CORE;
        ubBytesPerBlock = (ubBytesPerBlock / UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
        // 留 512B 余量，避免 UB 100% 占满导致 InitBuffer 失败
        if (ubBytesPerBlock * BLOCK_PER_CORE >= static_cast<int64_t>(ubSize)) {
            ubBytesPerBlock -= UB_ALIGN_BYTES;
        }
        ubFactor = ubBytesPerBlock / static_cast<int64_t>(sizeof(float));

        if (ubFactor > blockFactor) {
            ubFactor = blockFactor;
        }
        if (ubFactor < elemPerAlign) {
            ubFactor = elemPerAlign;
        }
    }

    // 6. 每核分块数（模板 TilingData 缺该字段）
    int64_t tileNum = 1;
    if (blockFactor > 0 && ubFactor > 0) {
        tileNum = CeilDiv(blockFactor, ubFactor);
        if (tileNum < 1) {
            tileNum = 1;
        }
    }

    // TODO: 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->tileNum = tileNum;

    // 配置启动核数量，kernel 侧 GetBlockNum() 读的就是它
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey（float16 → MODE_0、float32 → MODE_1）
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
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

// 全工程唯一的 tiling 注册点；relu_def.cpp 里不得再出现
IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling