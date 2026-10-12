/*!
 * \file sigmoid_tiling.cpp
 * \brief Sigmoid 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ascendc/host_api/tiling/template_argument.h"
#include <cstdint>
#include <algorithm>
#include <limits>

// 平台 opbuild（ascend_all_ops）阶段无法通过相对路径包含 ../op_kernel 下的头文件，
// 且模板引用的 op_common/* 头在本机 CANN 的 CMake include 路径下不存在，
// 故在此内联与 op_kernel/sigmoid_tiling_data.h、op_kernel/sigmoid_tiling_key.h 等价的定义，
// 两侧必须保持一致。
struct SigmoidTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 1;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
};

#define SIGMOID_TPL_SCH_MODE_0 0
#define SIGMOID_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    Sigmoid,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, SIGMOID_TPL_SCH_MODE_0, SIGMOID_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SIGMOID_TPL_SCH_MODE_0, SIGMOID_TPL_SCH_MODE_1)));

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
// 小数据量逐元素算子为固定开销（任务下发/核间同步）主导，实测带宽远低于硬件峰值；
// 提高单核最小切分门槛减少 block 数，以核数开销换单核并行度（评测 shape 8x2048 时 16 核→8 核）
constexpr int64_t MIN_SPLIT_THRESHOLD = 2048;
constexpr int64_t BUFFER_NUM = 2;

static inline int64_t SigmoidCeilDiv(int64_t x, int64_t y)
{
    return x / y + (x % y != 0 ? 1 : 0);
}

static inline int64_t SigmoidCeilAlign(int64_t x, int64_t align)
{
    return SigmoidCeilDiv(x, align) * align;
}

static inline int64_t SigmoidFloorAlign(int64_t x, int64_t align)
{
    return x / align * align;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SigmoidTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    if (GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SigmoidTilingData* tiling = context->GetTilingData<SigmoidTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType inputDtype = inputDesc->GetDataType();
    if (inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    const int64_t typeSize = static_cast<int64_t>(ge::GetSizeByDataType(inputDtype));
    if (typeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    if (totalNum <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (totalNum > std::numeric_limits<int64_t>::max() / typeSize) {
        return ge::GRAPH_FAILED;
    }

    const int64_t alignElem = 32 / typeSize;
    if (alignElem <= 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t blockNum = std::min(coreNum, SigmoidCeilDiv(totalNum, MIN_SPLIT_THRESHOLD));
    blockNum = std::max<int64_t>(blockNum, 1);
    int64_t blockFactor = SigmoidCeilAlign(SigmoidCeilDiv(totalNum, blockNum), alignElem);
    if (blockFactor <= 0) {
        return ge::GRAPH_FAILED;
    }

    blockNum = SigmoidCeilDiv(totalNum, blockFactor);
    if (blockNum <= 0 || blockNum > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    int64_t maxUbElem = static_cast<int64_t>(ubSize) / (BUFFER_NUM * 2 * typeSize);
    maxUbElem = SigmoidFloorAlign(maxUbElem, alignElem);
    const int64_t ubFactor = std::min(maxUbElem, blockFactor);
    if (ubFactor <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint64_t>(ubFactor * typeSize) > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(blockNum));

    const uint64_t tilingKey = (inputDtype == ge::DT_FLOAT16)
        ? GET_TPL_TILING_KEY(SIGMOID_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(SIGMOID_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSigmoid([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SigmoidCompileInfo {};

IMPL_OP_OPTILING(Sigmoid).Tiling(SigmoidTilingFunc).TilingParse<SigmoidCompileInfo>(TilingParseForSigmoid);

} // namespace optiling
