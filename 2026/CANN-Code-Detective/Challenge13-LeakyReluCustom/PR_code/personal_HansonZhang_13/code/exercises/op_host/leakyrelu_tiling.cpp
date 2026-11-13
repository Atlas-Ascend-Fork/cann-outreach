/*!
 * \file leakyrelu_tiling.cpp
 * \brief Leakyrelu 算子 Tiling 实现
 *
 * 本期（第 13 期）与往期最大的不同：**引入了算子属性 negative_slope**。
 * 任务书要求「属性需在 op_def 中注册，由 Tiling 函数读取后写入 tilingData，
 * 再下发到 kernel 侧使用」，因此本文件负责：
 *   1) 读取输入 shape / dtype，算出 totalNum；
 *   2) 从算子属性里读出 negative_slope，写入 tiling->negativeSlope；
 *   3) 按「核数切分 + UB 分块」算出 blockFactor / ubFactor，并与 kernel 侧
 *      InitBuffer 的申请量严格一致；
 *   4) 按 dtype 选择 tilingKey（0 → float16，1 → float32）。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/leakyrelu_tiling_data.h"
#include "../op_kernel/leakyrelu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;   // float32 字节数
constexpr int64_t HALF_SIZE = 2;   // float16 字节数
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// ★ 与 op_kernel/leakyrelu.h 里的 BUFFER_NUM 必须保持一致（= 2）。
//   UB 用量 = 2 个队列 × BUFFER_NUM 个 buffer × ubFactor × sizeof(T)。
constexpr int64_t KERNEL_BUFFER_NUM = 2;

// ★ DMA 对齐粒度：Ascend 的 DataCopy 以 32 字节为最小搬运单位，
//   该值是硬件常量而非平台可配置项，故直接写死。
constexpr int64_t DMA_ALIGN_BYTES = 32;

// ★ UB 预留：给 UB 上的其它用途留一点余量，不要把 UB 吃满。
constexpr int64_t UB_RESERVED_BYTES = 1024;

// ★ 负半轴斜率的缺省值：与 torch.nn.functional.leaky_relu 的缺省值一致。
constexpr float DEFAULT_NEGATIVE_SLOPE = 0.01f;

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

static ge::graphStatus LeakyreluTilingFunc(gert::TilingContext* context)
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

    LeakyreluTilingData* tiling = context->GetTilingData<LeakyreluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---- 1. 总元素数：逐维相乘 ----
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = 1;
    for (size_t i = 0; i < inputShape->GetStorageShape().GetDimNum(); i++) {
        totalNum *= inputShape->GetStorageShape().GetDim(i);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "input tensor is empty"), return ge::GRAPH_FAILED);

    // ---- 2. dtype：只接受 float32 / float16（与 op_def 注册的 dtype 一致）----
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "Leakyrelu only supports float32/float16"),
        return ge::GRAPH_FAILED);
    const bool isFloat16 = (dataType == ge::DT_FLOAT16);
    const int64_t dtypeSize = isFloat16 ? HALF_SIZE : TYPE_SIZE;

    // ---- 3. ★ 读取算子属性 negative_slope ----
    //   op_def 里注册的是 `.Attr("negative_slope").Float()`，且只有这一个属性，
    //   所以它在 RuntimeAttrs 里的下标是 0。
    //   写法与本仓已合入的同族实现一致（Xian-Jiaotong-University 的 SoftshrinkGrad
    //   / MseLoss 的 op_host 里都是 `context->GetAttrs()->GetAttrPointer<float>(0)`）。
    float negativeSlope = DEFAULT_NEGATIVE_SLOPE;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* slopePtr = attrs->GetAttrPointer<float>(0);
        if (slopePtr != nullptr) {
            negativeSlope = *slopePtr;
        }
    }
    // 属性缺省时（没有传 attr）用 0.01f；NaN 属性直接拒绝，避免污染结果。
    OP_CHECK_IF(
        !(negativeSlope == negativeSlope),
        OP_LOGE(context, "negative_slope must not be NaN"),
        return ge::GRAPH_FAILED);

    // ---- 4. blockFactor / ubFactor ----
    //    对齐粒度取 32 字节（Ascend DMA 粒度），保证 DataCopy 的 DMA 粒度要求。
    const int64_t alignNum = DMA_ALIGN_BYTES / dtypeSize;
    OP_CHECK_IF(alignNum <= 0, OP_LOGE(context, "invalid align unit"), return ge::GRAPH_FAILED);

    // 核数不能超过「按对齐粒度切分后能覆盖的数据块数」，否则会出现空核
    int64_t usedCoreNum = coreNum;
    const int64_t coreNumByData = CeilDiv(totalNum, alignNum);
    if (usedCoreNum > coreNumByData) {
        usedCoreNum = coreNumByData;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // ★ UB 预算必须与 kernel 侧 InitBuffer 的实际申请严格一致（见 op_kernel/leakyrelu.h）：
    //     fp32: 2 个队列 × 2 个 buffer × 4 字节 = 16 字节/元素
    //     fp16: 2 个队列 × 2 个 buffer × 2 字节 =  8 字节/元素
    //   即 bytesPerElement = 2 × KERNEL_BUFFER_NUM × dtypeSize。
    const int64_t bytesPerElement = 2 * KERNEL_BUFFER_NUM * dtypeSize;
    int64_t ubFactorMax = (static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES) / bytesPerElement;
    // 向下对齐到对齐单位的整数倍（手写，避免依赖未经核对的 FloorAlign）
    ubFactorMax = (ubFactorMax / alignNum) * alignNum;
    // ★ UB 连一个对齐单元都放不下时直接失败，而不是抬高 ubFactorMax ——
    //   抬高会让 InitBuffer 的申请量超过平台 UB 预算（越界），比报错更危险。
    OP_CHECK_IF(
        ubFactorMax < alignNum,
        OP_LOGE(context, "ubSize %lu is too small for Leakyrelu tiling", static_cast<uint64_t>(ubSize)),
        return ge::GRAPH_FAILED);

    // 每个核负责的元素个数：先均分，再向上对齐到 32 字节的整数倍，
    // 保证每个核的 DMA 搬运都是整数个 DMA 单元（末尾核的截断尾块由 kernel 侧用
    // DataCopyPad 处理）。
    const int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);
    // 单个 tile 的元素个数：不超过一个核的数据量，且受 UB 预算约束。
    const int64_t ubFactor = blockFactor < ubFactorMax ? blockFactor : ubFactorMax;

    // ---- 5. 设置 tiling 数据 ----
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    // ★ 属性 α 随 tiling 数据下发到 kernel 侧
    tiling->negativeSlope = negativeSlope;
    // reserved：显式写 0，把结构体尾部填充（结构体按 int64 序列化时落在第 4 个
    // int64 的高 32 位）固定住 ⇒ 官方 host 单测的 expectTilingData 断言结果确定。
    tiling->reserved = 0;
    // 显式声明写入长度，不依赖 GetTilingData<T>() 的隐式 SetDataSize
    context->GetRawTilingData()->SetDataSize(sizeof(LeakyreluTilingData));

    // ---- 6. 设置核数：总数据被 blockFactor 切成多少份就需要多少个核 ----
    context->SetBlockDim(static_cast<uint32_t>(CeilDiv(totalNum, blockFactor)));

    // ---- 7. 选择 tilingKey：0 -> float16，1 -> float32 ----
    //    与 op_kernel/leakyrelu_tiling_key.h 中的 LEAKYRELU_TPL_SCH_MODE_0/1 一一对应，
    //    官方 host 单测期望 float32 的 tilingKey == 1。
    uint64_t tilingKey;
    if (isFloat16) {
        tilingKey = GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLeakyrelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LeakyreluCompileInfo {};

IMPL_OP_OPTILING(Leakyrelu).Tiling(LeakyreluTilingFunc).TilingParse<LeakyreluCompileInfo>(TilingParseForLeakyrelu);

} // namespace optiling
