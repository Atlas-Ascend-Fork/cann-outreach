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

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BUFFER_NUM = 2;      // 队列 buffer 块数
constexpr int64_t TEMP_NUM_FP32 = 4;   // float32 输入的中间 buffer 数：输入转换 + 3 个计算中间量
constexpr int64_t TEMP_NUM_FP16 = 5;   // float16 输入的中间 buffer 数：输入转换 + 输出转换 + 3 个计算中间量
constexpr int64_t ALIGN_NUM = 16;      // 每核处理元素数按 32 字节（float16）向上对齐
constexpr int64_t MAX_BLOCK_NUM = 8;   // 启动核数上限
constexpr double UB_USAGE_RATIO = 0.9; // UB 使用率，预留余量避免申请失败

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

// 逐元素算子：一次 UB 处理所需的字节数
static int64_t GetUbBytePerElem(int64_t elemSize, int64_t tempNum)
{
    // 入/出队列各 BUFFER_NUM 份，中间量按 float32 计算
    return BUFFER_NUM * 2 * elemSize + tempNum * static_cast<int64_t>(sizeof(float));
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

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType inputDtype = inputDesc->GetDataType();
    const bool isFp16 = (inputDtype == ge::DT_FLOAT16 || inputDtype == ge::DT_BF16);
    const int64_t elemSize = isFp16 ? 2 : static_cast<int64_t>(sizeof(float));
    const int64_t tempNum = isFp16 ? TEMP_NUM_FP16 : TEMP_NUM_FP32;

    // 输入元素总数
    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape xShape = EnsureNotScalar(xStorageShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        totalNum *= static_cast<int64_t>(xShape.GetDim(i));
    }

    // 启动核数：以向量核数为基础，上限取 MAX_BLOCK_NUM，且不超元素总数
    int64_t blockDim = coreNum;
    if (blockDim > MAX_BLOCK_NUM) {
        blockDim = MAX_BLOCK_NUM;
    }
    if (blockDim > totalNum) {
        blockDim = totalNum;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }

    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    if (totalNum > 0) {
        // 每核处理元素数，按 32 字节对齐后反推实际核数，保证各核起始地址对齐
        blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), ALIGN_NUM);
        blockDim = CeilDiv(totalNum, blockFactor);

        // 单次 UB 处理元素数：受 UB 容量限制，并对齐到 32 字节
        const int64_t alignNum = 32 / elemSize;
        const int64_t ubBytePerElem = GetUbBytePerElem(elemSize, tempNum);
        int64_t ubElemNum = static_cast<int64_t>(static_cast<double>(ubSize) * UB_USAGE_RATIO) / ubBytePerElem;
        ubElemNum = (ubElemNum / alignNum) * alignNum;
        if (ubElemNum < alignNum) {
            ubElemNum = alignNum;
        }
        ubFactor = (blockFactor < ubElemNum) ? blockFactor : ubElemNum;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(blockDim);

    // 根据输入 dtype 选择 tilingKey：fp16/bf16 走高精度转换分支，fp32 走直算分支
    uint64_t tilingKey = isFp16 ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0) : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
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
