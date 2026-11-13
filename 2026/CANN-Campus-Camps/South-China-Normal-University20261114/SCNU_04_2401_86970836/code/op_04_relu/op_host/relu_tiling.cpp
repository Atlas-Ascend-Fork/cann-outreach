/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现（性能优化版）
 *
 * 优化点：
 *   1. 按 512B 对齐（910B 实测最优 GM 搬运粒度），提升 DMA 带宽；
 *   2. 小 shape 减少启动核数：以 64KB/核为目标，避免几十个核各搬 1~2KB
 *      导致块调度和 DMA 启动开销占主导；
 *   3. 大 shape 使用全部 AIV 核，并按 UB 上限选择每核 tile 数，
 *      使 blockFactor 是 ubFactor 的整数倍，避免每核出现极小尾块；
 *   4. 输入/输出各 1 块，共 2 块 UB，按 80% UB 预算分配；这样单块 tile 可放大约 2 倍，
 *      减少 DMA 次数和循环开销，冷/热数据下 4M 元素都能稳定接近带宽上限；
 *   5. 尾核由 kernel 根据 totalNum 裁剪，Tiling 侧直接设置实际启动核数。
 */

#include <algorithm>
#include <cstdint>

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

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE_FP32 = 4;
constexpr int64_t TYPE_SIZE_FP16 = 2;

// 与 op_kernel/relu.h 保持一致：输入/输出各 1 块，共 2 块 UB。
// UB 占用减半后，ubFactor 可放大约 2 倍，减少循环次数与 DMA 启动开销。
constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t TENSOR_NUM = BUFFER_NUM * 2; // in 1 块 + out 1 块

// 使用 80% UB，给系统预留空间
constexpr int64_t UB_RATIO_NUM = 8;
constexpr int64_t UB_RATIO_DEN = 10;

// 910B 实测最优 GM 搬运粒度：512B 对齐比 32B 对齐带宽更高
constexpr int64_t BANDWIDTH_ALIGN_BYTES = 512;

// 小 shape 下每核至少处理 64KB，避免启动过多核导致调度/DMA 启动开销
constexpr int64_t TARGET_BYTES_PER_BLOCK = 64 * 1024;

// 单块元素数上限，避免极端 UB 配置下分配过大的 tile；
// 单缓冲下 fp16 的 ubFactor 可达 ~39K，因此上限提高到 65536。
constexpr int64_t MAX_UB_FACTOR = 65536;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
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

    // 1. 总元素数量
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t totalNum = xShape->GetStorageShape().GetShapeSize();

    // 2. dtype -> typeSize / tilingKey
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();

    int64_t typeSize = TYPE_SIZE_FP32;
    uint32_t schMode = RELU_TPL_SCH_MODE_1; // float32
    if (dtype == ge::DT_FLOAT16) {
        typeSize = TYPE_SIZE_FP16;
        schMode = RELU_TPL_SCH_MODE_0;      // float16
    } else if (dtype != ge::DT_FLOAT) {
        OP_LOGE(context, "Relu only supports float32/float16");
        return ge::GRAPH_FAILED;
    }

    // 空 Tensor 保护
    if (totalNum <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, schMode);
        return ge::GRAPH_SUCCESS;
    }

    // 3. 512B 对齐对应的元素数（fp32=128, fp16=256）
    const int64_t alignElems = BANDWIDTH_ALIGN_BYTES / typeSize;

    // 4. 先根据数据量决定启动核数：
    //    小 shape 用少量核，让每核搬运更大的连续块；
    //    大 shape 用满所有 AIV 核。
    const int64_t totalBytes = totalNum * typeSize;
    int64_t desiredCoreNum = CeilDiv(totalBytes, TARGET_BYTES_PER_BLOCK);
    if (desiredCoreNum < 1) {
        desiredCoreNum = 1;
    }
    if (desiredCoreNum > coreNum) {
        desiredCoreNum = coreNum;
    }
    if (desiredCoreNum > totalNum) {
        desiredCoreNum = totalNum;
    }

    // 5. 每核理想长度：按 512B 对齐，先不绑定 ubFactor
    const int64_t rawBlockFactor = CeilAlign(CeilDiv(totalNum, desiredCoreNum), alignElems);

    // 6. UB 单块上限：输入/输出各 1 块，共 2 块，使用 80% UB
    const int64_t usableUb = static_cast<int64_t>(ubSize) * UB_RATIO_NUM / UB_RATIO_DEN;
    int64_t maxUbFactor = FloorAlign(FloorDiv(usableUb, TENSOR_NUM * typeSize), alignElems);
    if (maxUbFactor > MAX_UB_FACTOR) {
        maxUbFactor = MAX_UB_FACTOR;
    }
    if (maxUbFactor < alignElems) {
        maxUbFactor = alignElems;
    }

    // 7. 选择每核 tile 数：ubFactor <= maxUbFactor，
    //    同时 blockFactor = tileCount * ubFactor，保证每个核内部都是完整 tile，
    //    避免 blockFactor 与 ubFactor 不整除时每个核都多出一个极小尾块。
    int64_t tileCount = CeilDiv(rawBlockFactor, maxUbFactor);
    if (tileCount < 1) {
        tileCount = 1;
    }
    int64_t ubFactor = CeilAlign(CeilDiv(rawBlockFactor, tileCount), alignElems);
    if (ubFactor < alignElems) {
        ubFactor = alignElems;
    }
    int64_t blockFactor = tileCount * ubFactor;

    // 8. 反推实际启动核数
    int64_t realCoreNum = CeilDiv(totalNum, blockFactor);
    if (realCoreNum < 1) {
        realCoreNum = 1;
    }
    if (realCoreNum > desiredCoreNum) {
        realCoreNum = desiredCoreNum;
    }

    // 9. 写入 TilingData
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 10. 配置启动核数量与 tilingKey
    context->SetBlockDim(static_cast<uint32_t>(realCoreNum));
    ASCENDC_TPL_SEL_PARAM(context, schMode);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
