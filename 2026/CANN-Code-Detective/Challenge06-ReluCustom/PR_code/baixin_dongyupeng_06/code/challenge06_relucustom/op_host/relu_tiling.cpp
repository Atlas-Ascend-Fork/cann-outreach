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

// tiling 常量（DESIGN.md §2.4.1）
constexpr int64_t MIN_TILING_BYTES_PER_CORE = 4096;  // 每核最少 4KB 数据
constexpr int64_t ELEM_ALIGN_FACTOR = 512;           // 多核元素对齐
constexpr int64_t ALIGN_256B = 256;                  // UB 256B 对齐
constexpr int64_t BUFFER_NUM = 2;                    // in/out 各 2 buffer
constexpr int64_t UB_RESERVED = 8 * 1024;            // UB 预留字节

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

    // 根据输入 dtype 选择 tilingKey（保留现有分发：FLOAT16/BF16 → MODE_0(half)，否则 → MODE_1(float)）
    // 空 shape 分支与非空分支共用单一 SetTilingKey 出口，MODE_0/MODE_1 分发现状不失效
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE(context, "GetInputDesc(0) is nullptr"),
                return ge::GRAPH_FAILED);
    uint64_t tilingKey;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }

    // 1) 元素总数（ND 展平）与对齐因子
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_IF(inputShape == nullptr, OP_LOGE(context, "GetInputShape(0) is nullptr"),
                return ge::GRAPH_FAILED);
    const gert::Shape& originShape = inputShape->GetOriginShape();
    int64_t totalNum = 1;
    for (size_t i = 0; i < originShape.GetDimNum(); i++) {
        totalNum *= originShape.GetDim(i);
    }
    ge::DataType inputDtype = inputDesc->GetDataType();
    int64_t elemBytes = (inputDtype == ge::DT_FLOAT16) ? 2 : 4;  // 当前注册仅 FLOAT/FLOAT16
    int64_t alignFactor = ALIGN_256B / elemBytes;                // 64(fp32)/128(fp16)

    int64_t blockFactor;
    int64_t ubFactor;
    if (totalNum == 0) {
        // 空 shape 防护（DESIGN §2.3/§2.4.1）：
        // CeilDiv(0,·)=0、CeilAlign(0,512)=0 会让 blockFactor=0 → blockNum 整数除零（SIGFPE）。
        // 独立分支：blockFactor 显式给 1，ubFactor 给一档 256B 对齐长度，SetBlockDim(1)；
        // Kernel 端 Init 算得 blockLength_=0，Process 空转返回。
        blockFactor = 1;
        ubFactor = alignFactor;
        context->SetBlockDim(1);
    } else {
        // 2) 多核切分（每核至少 4KB 数据才开核）
        int64_t usedCoreNum = std::min((totalNum * elemBytes + MIN_TILING_BYTES_PER_CORE - 1)
                                           / MIN_TILING_BYTES_PER_CORE,
                                       coreNum);
        usedCoreNum = std::max((int64_t)1, usedCoreNum);
        // totalNum>0 → blockFactor ≥ 512，后续除法永不除零
        blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), ELEM_ALIGN_FACTOR);
        int64_t blockNum = (totalNum + blockFactor - 1) / blockFactor;  // blockNum ≤ coreNumAvail 恒成立
        context->SetBlockDim(blockNum);

        // 3) UB 切分（256B 对齐）
        int64_t maxElem = (int64_t)((ubSize - UB_RESERVED) / (BUFFER_NUM * 2 * elemBytes));
        ubFactor = (maxElem / alignFactor) * alignFactor;
    }

    // 写入 tiling 数据（三字段结构体保持不变）
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = std::max(ubFactor, alignFactor);

    context->SetTilingKey(tilingKey);  // 单一出口，空 / 非空分支均写入
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
