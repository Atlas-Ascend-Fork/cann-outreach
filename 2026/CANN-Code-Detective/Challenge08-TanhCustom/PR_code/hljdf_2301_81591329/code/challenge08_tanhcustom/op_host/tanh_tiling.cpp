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

constexpr uint32_t WS_SYS_SIZE = 0U;

/*
 * 题目固定输入：
 *
 * (8, 2048)
 *
 * 总元素个数：
 *
 * 8 * 2048 = 16384
 *
 * 使用 8 个 Vector Core，
 * 每个 Core 处理 2048 个元素。
 */
constexpr int64_t BLOCK_NUM = 8;

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(
            platformInfoPtr);

    coreNum =
        ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum == 0,
        OP_LOGE(
            context,
            "coreNum is 0"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(
            context,
            "ubSize is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        currentWorkspace);

    currentWorkspace[0] =
        WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TanhTilingFunc(
    gert::TilingContext* context)
{
    /*
     * 1. 获取 NPU 平台信息
     */
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    if (GetPlatformInfo(
            context,
            ubSize,
            coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    /*
     * 2. 获取输入 Shape
     */
    const gert::StorageShape* xShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        xShape);

    const int64_t totalNum =
        xShape
            ->GetOriginShape()
            .GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(
            context,
            "input size is invalid"),
        return ge::GRAPH_FAILED);

    /*
     * 3. 设置使用的 Vector Core 数量
     */
    const int64_t blockNum =
        (coreNum >= BLOCK_NUM)
            ? BLOCK_NUM
            : coreNum;

    OP_CHECK_IF(
        blockNum <= 0,
        OP_LOGE(
            context,
            "blockNum is invalid"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        totalNum % blockNum != 0,
        OP_LOGE(
            context,
            "totalNum must be divisible by blockNum"),
        return ge::GRAPH_FAILED);

    /*
     * shape = (8, 2048)
     *
     * totalNum    = 16384
     * blockNum    = 8
     * blockFactor = 2048
     * ubFactor    = 2048
     */
    const int64_t blockFactor =
        totalNum / blockNum;

    const int64_t ubFactor =
        blockFactor;

    /*
     * 4. 设置 TilingData
     */
    TanhTilingData* tilingData =
        context
            ->GetTilingData<TanhTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tilingData);

    tilingData->totalNum =
        totalNum;

    tilingData->blockFactor =
        blockFactor;

    tilingData->ubFactor =
        ubFactor;

    /*
     * 5. 设置启动核数
     */
    context->SetBlockDim(
        static_cast<uint32_t>(
            blockNum));

    /*
     * 6. 使用固定 TilingKey
     */
    ASCENDC_TPL_SEL_PARAM(
        context,
        TANH_TPL_SCH_MODE);

    /*
     * 7. Workspace
     */
    return GetWorkspaceSize(
        context);
}

static ge::graphStatus TilingParseForTanh(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh)
    .Tiling(TanhTilingFunc)
    .TilingParse<TanhCompileInfo>(
        TilingParseForTanh);

} // namespace optiling