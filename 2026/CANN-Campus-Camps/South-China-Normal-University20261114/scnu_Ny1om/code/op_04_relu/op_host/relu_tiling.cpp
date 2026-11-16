/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 *
 * 默认切分策略与原实现完全一致（保持已验证的性能行为）：
 *   ubFactor  ：UB 能放下的最大 tile（输入/输出各 double buffer），按 32B 对齐，不超过 totalNum；
 *   分核      ：每个核至少处理一个完整 UB tile（needCoreNum = ceil(totalNum/ubFactor)），
 *               小 shape（如 8*2048）只启动 1~2 个核，避免大量核做小搬运的固定开销；
 *   blockFactor：ceil(totalNum/usedCoreNum) 再按 32B 元素数对齐。
 *
 * 实验旋钮（环境变量，默认关闭；未设置时输出与原实现逐字节一致）：
 *   RELU_MIN_BYTES_PER_CORE : 每核最少处理的字节数。设置后 usedCoreNum =
 *                             min(coreNum, ceil(totalBytes / 该值))，用于扫"小 shape 用几个核最快"。
 *                             N 越小核越多（如 8192 -> 64KB 数据用 8 核）。
 *   RELU_DMA_ALIGN_BYTES    : blockFactor / ubFactor 的对齐字节数（默认 32）。可试 512：
 *                             910B 上实测 GM<->UB 搬运带宽与起始地址对齐的关系约为
 *                             512B 对齐 ~100%、256B ~90%、仅 32B 对齐 ~70%。
 * 两个旋钮只在首次 tiling 时读一次环境变量并缓存；扫出最优组合后把对应常量写死、
 * 删掉旋钮即可作为最终提交。扫参方式见 bench/sweep.sh 与 OPTIMIZATION.md。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

#include <cstdlib>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;

constexpr int64_t BUFFER_NUM = 2;           // 输入/输出 Queue 的 double buffer 深度
constexpr int64_t UB_RESERVED_SIZE = 1024;  // UB 预留空间，避免 buffer 对齐取整后越界

// ---- 实验旋钮（默认关闭，未设置时与原实现行为一致）----

// RELU_MIN_BYTES_PER_CORE：每核最少处理字节数；<=0 表示用默认策略（每核至少一个 UB tile）
static int64_t GetTuningMinBytesPerCore()
{
    static int64_t val = []() -> int64_t {
        const char* s = std::getenv("RELU_MIN_BYTES_PER_CORE");
        return (s != nullptr && s[0] != '\0') ? std::atoll(s) : 0;
    }();
    return val;
}

// RELU_DMA_ALIGN_BYTES：对齐粒度字节数；<=0 或不是 32B 整数倍时回退为 32
static int64_t GetTuningDmaAlignBytes()
{
    static int64_t val = []() -> int64_t {
        const char* s = std::getenv("RELU_DMA_ALIGN_BYTES");
        return (s != nullptr && s[0] != '\0') ? std::atoll(s) : 0;
    }();
    return val;
}

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

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = EnsureNotScalar(inputShapePtr->GetOriginShape());
    int64_t totalNum = inputShape.GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    // 根据输入 dtype 选择 tilingKey 并确定单元素占用字节数
    ge::DataType dtype = inputDesc->GetDataType();
    uint64_t tilingKey;
    int64_t typeSize;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
        typeSize = 2;
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
        typeSize = TYPE_SIZE;
    }
    context->SetTilingKey(tilingKey);

    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    // UB 对齐元素个数：一个 block（32B）能容纳的元素数
    int64_t alignNum = static_cast<int64_t>(GetUbBlockSize(context)) / typeSize;
    if (alignNum <= 0) {
        alignNum = 1;
    }

    // 实验旋钮：对齐粒度。默认 32B（= alignNum，与原实现一致），可设 512 等
    int64_t tuneAlignBytes = GetTuningDmaAlignBytes();
    if (tuneAlignBytes < static_cast<int64_t>(GetUbBlockSize(context)) ||
        (tuneAlignBytes % static_cast<int64_t>(GetUbBlockSize(context))) != 0) {
        tuneAlignBytes = static_cast<int64_t>(GetUbBlockSize(context));
    }
    int64_t alignElem = tuneAlignBytes / typeSize;
    if (alignElem < alignNum) {
        alignElem = alignNum;
    }

    // 单个 UB tile 尽量取大：输入 double buffer + 输出 double buffer
    int64_t bytesPerElem = typeSize * BUFFER_NUM * 2;
    int64_t availUb = static_cast<int64_t>(ubSize) > UB_RESERVED_SIZE ?
                          static_cast<int64_t>(ubSize) - UB_RESERVED_SIZE : static_cast<int64_t>(ubSize);
    int64_t ubFactor = FloorAlign(availUb / bytesPerElem, alignElem);
    if (ubFactor < alignElem) {
        ubFactor = alignElem;
    }
    if (ubFactor > totalNum) {
        ubFactor = totalNum;
    }

    // 默认策略：让每个核至少处理一个完整的 UB tile。
    // 小 shape（如 8*2048）只启动 1~2 个核，避免大量核做几百字节搬运带来的固定开销
    int64_t needCoreNum = CeilDiv(totalNum, ubFactor);
    int64_t usedCoreNum = needCoreNum < coreNum ? needCoreNum : coreNum;

    // 实验旋钮：每核最少字节数（覆盖上面的默认策略，用于扫核数）
    int64_t tuneMinBytes = GetTuningMinBytesPerCore();
    if (tuneMinBytes > 0) {
        int64_t minElems = CeilDiv(tuneMinBytes, typeSize);
        int64_t byWorkCoreNum = CeilDiv(totalNum, minElems);
        usedCoreNum = byWorkCoreNum < coreNum ? byWorkCoreNum : coreNum;
    }

    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignElem);
    int64_t realCoreNum = CeilDiv(totalNum, blockFactor);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(realCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
