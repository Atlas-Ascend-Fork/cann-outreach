/*!
 * \file sigmoid_tiling.cpp
 * \brief Sigmoid 算子 Tiling 实现
 *
 * Tiling 策略（DESIGN §2.1 / §2.2）：
 *   多核切分：每核至少 4KB 数据（MIN_TILING_BITS=32768 bits），blockFactor 对齐到 512 元素
 *   UB 切分  ：按 mode 的每元素 UB 字节数（mode1=20B / mode0=16B）预算，扣 UB_EXTRA_SIZE 头寸后
 *              对齐到 256B（ubFactor 元素数 = 256 / sizeof(T) 的整数倍）
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sigmoid_tiling_data.h"
#include "../op_kernel/sigmoid_tiling_key.h"

#include <algorithm>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_TILING_BITS = 32768;     // 每核至少 4KB（单位 bits）
constexpr int64_t ELEM_ALIGN_FACTOR = 512;     // 多核切分元素对齐因子
constexpr int64_t ALIGN_256B = 256;            // UB 对齐字节数
constexpr int64_t UB_EXTRA_SIZE = 16384;       // UB 预留头寸（字节），≈16KB
// mode1(float)：2×4B 输入 + 2×4B 输出 + 1×4B ones = 20 B/元素
constexpr int64_t BYTE_PER_ELEM_MODE1 = 20;
// mode0(half) ：2×2B 输入 + 2×2B 输出 + 1×4B ones + 1×4B f32Work = 16 B/元素
constexpr int64_t BYTE_PER_ELEM_MODE0 = 16;

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

static ge::graphStatus SigmoidTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t aivNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, aivNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    SigmoidTilingData* tiling = context->GetTilingData<SigmoidTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 根据输入 dtype 选择 tilingKey 与缓冲预算（DESIGN §0.2#2 / §2.3）
    // 注意：BF16 未注册（OpDef 仅注册 FLOAT/FLOAT16），不可路由到 mode0（Sigmoid<half>），
    //       若未来启用 BF16 需新增 Sigmoid<bfloat16_t> 实例化与 T 专用 Cast/Exp 路径。
    auto inputDesc = context->GetInputDesc(0);
    uint64_t tilingKey;
    int64_t dtypeSize;
    int64_t bytePerElem;
    int64_t alignFactor;   // ubFactor 元素对齐数 = 256 / sizeof(T)
    const bool isHalf = (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16);
    if (isHalf) {
        tilingKey   = GET_TPL_TILING_KEY(SIGMOID_TPL_SCH_MODE_0);
        dtypeSize   = 2;
        bytePerElem = BYTE_PER_ELEM_MODE0;
        alignFactor = ALIGN_256B / dtypeSize;   // 128
    } else {
        tilingKey   = GET_TPL_TILING_KEY(SIGMOID_TPL_SCH_MODE_1);
        dtypeSize   = 4;
        bytePerElem = BYTE_PER_ELEM_MODE1;
        alignFactor = ALIGN_256B / dtypeSize;   // 64
    }

    // 展平元素数（标量 shape 规整为 {1}）
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& inShape = EnsureNotScalar(inputShape->GetOriginShape());
    int64_t totalNum = inShape.GetShapeSize();

    // 空输入保护（DESIGN §2.6）
    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = alignFactor;   // 合法对齐值；kernel blockLength_=0，循环 0 次
        context->SetBlockDim(1);
        context->SetTilingKey(tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    // 多核切分（DESIGN §2.1）：每核至少 4KB，blockFactor 对齐 512
    int64_t coreNum = (totalNum * dtypeSize * 8 + MIN_TILING_BITS - 1) / MIN_TILING_BITS;
    coreNum = std::min(coreNum, aivNum);
    coreNum = std::max(coreNum, static_cast<int64_t>(1));
    int64_t blockFactor = ((totalNum + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1) /
                          ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
    int64_t blockNum = (totalNum + blockFactor - 1) / blockFactor;  // 实际块数（含尾块）
    (void)blockNum;

    // UB 切分（DESIGN §2.2）：扣头寸后按每元素字节预算，对齐到 256B
    int64_t maxElemNum = (int64_t)ubSize - UB_EXTRA_SIZE;
    if (maxElemNum < 0) {
        maxElemNum = 0;
    }
    maxElemNum /= bytePerElem;
    int64_t ubFactor = (maxElemNum / alignFactor) * alignFactor;
    if (ubFactor == 0) {
        ubFactor = alignFactor;   // UB 异常小保护（正常平台不触发）
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim((uint32_t)coreNum);
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
