/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 *
 * 切分策略（性能相关）：
 *  1. 多核切分：totalNum 在 AIV 核间尽量均分，每个核的起始偏移按 32B 对齐，
 *     使 kernel 侧 GM->UB / UB->GM 的常规 tile 都能走对齐的 DataCopy 整块搬运；
 *  2. UB 切分：输入、输出队列各 2 份 buffer（双缓冲），据此计算 ubFactor 上限，
 *     再与单核数据量 blockFactor 取小。单核数据可一次搬完时核内只有一趟 DMA，
 *     数据量大时按 ubFactor 循环，由双缓冲流水掩盖搬运延迟；
 *  3. 不申请 workspace，避免每次调用额外的 device 侧内存申请。
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
constexpr int64_t UB_BLOCK_SIZE = 32;        // UB 与 GM 间搬运的基本块长度（字节）
constexpr int64_t UB_RESERVED_BYTES = 1024;  // UB 预留量，避免平台上报值与实际可用值有细微差异时越界
constexpr int64_t QUEUE_BUFFER_NUM = 2;      // kernel 侧输入/输出队列的 buffer 个数，需与 relu.h 的 BUFFER_NUM 一致

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

    // 根据输入 dtype 选择计算类型与 tilingKey（kernel 侧据此实例化 float / half 版本）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    int64_t dtypeSize = 0;
    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT) {
        dtypeSize = static_cast<int64_t>(sizeof(float));
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    } else if (dtype == ge::DT_FLOAT16) {
        dtypeSize = static_cast<int64_t>(sizeof(uint16_t));
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        OP_LOGE(context, "Relu only supports float32 and float16 input");
        return ge::GRAPH_FAILED;
    }

    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const int64_t totalNum = EnsureNotScalar(inputShapePtr->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "input shape is unknown"), return ge::GRAPH_FAILED);

    // 空 Tensor 不启动实际计算，但仍返回成功并给出合法的 blockDim
    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        context->SetTilingKey(tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    // 1. 多核切分：每核 blockFactor 个元素，并按 32B 对齐，保证每个核的 GM 起始地址满足搬运对齐要求
    const int64_t blockAlign = UB_BLOCK_SIZE / dtypeSize;
    const int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), blockAlign);
    const int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);

    // 2. UB 切分：输入队列 BUFFER_NUM 份 + 输出队列 BUFFER_NUM 份，每元素共 2 * BUFFER_NUM * dtypeSize 字节
    const int64_t bytesPerElem = 2 * QUEUE_BUFFER_NUM * dtypeSize;
    int64_t ubSizeUsable = static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES;
    if (ubSizeUsable < UB_BLOCK_SIZE) {
        ubSizeUsable = static_cast<int64_t>(ubSize);
    }
    int64_t ubFactorMax = FloorAlign(ubSizeUsable / bytesPerElem, blockAlign);
    if (ubFactorMax < blockAlign) {
        ubFactorMax = blockAlign;
    }
    // 单核数据量不超过 UB 上限时只切一块，核内一趟搬入 + 一次计算 + 一趟搬出
    const int64_t ubFactor = ubFactorMax < blockFactor ? ubFactorMax : blockFactor;

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
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
