/*!
 * \file abs_tiling.cpp
 * \brief Abs 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/abs_tiling_data.h"
#include "../op_kernel/abs_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
// ★ 故意不使用 Ops::Base::FloorAlign：本机没有 CANN toolkit，无法核对它在该版本
//   math_util.h 中是否真的存在（已合入的第十期实现同样只用了 CeilDiv / CeilAlign，
//   向下取整对齐一律手写 (x / n) * n）。手写等价且零依赖风险。

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;  // float32 字节数
constexpr int64_t HALF_SIZE = 2;  // float16 字节数
constexpr int64_t BUFFER_NUM = 1; // 与 op_kernel/abs.h 的 BUFFER_NUM 保持一致
// ★ DMA 对齐粒度：Ascend 的 DataCopy 以 32 字节为最小搬运单位，
//   该值是硬件常量而非平台可配置项，故直接写死，
//   避免依赖平台信息里「UB block size」这类在 host UT 假平台中并不存在的字段。
constexpr int64_t DMA_ALIGN_BYTES = 32;
// ★ UB 预留：给 UB 上的其它用途（临时变量、对齐填充）留一点余量，不要把 UB 吃满。
constexpr int64_t UB_RESERVED_BYTES = 1024;

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

static ge::graphStatus AbsTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    // 1. 获取平台信息（UB 大小、AIV 核数）
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 设置 Workspace 大小（Abs 是纯逐元素计算，不需要额外 workspace）
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    AbsTilingData* tiling = context->GetTilingData<AbsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 3. 取输入 shape 与 dtype
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    // ★ 逐维相乘求总元素数：写法与官方模板 code_add/op_host/add_tiling.cpp
    //   以及官方教程的 AddCustom tiling 完全一致（只用 GetDimNum / GetDim，
    //   不引入任何无法在本机核对的额外 API）。
    int64_t totalNum = 1;
    for (size_t i = 0; i < inputShape->GetStorageShape().GetDimNum(); i++) {
        totalNum *= inputShape->GetStorageShape().GetDim(i);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "input tensor is empty"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    // ★ 只接受 float32 / float16：op_def 中只声明了这两个 dtype，
    //   且 fp16 与 bf16 的位布局不同，误判会得到完全错误的数值，故显式拒绝。
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "Abs only supports float32/float16"),
        return ge::GRAPH_FAILED);

    const bool isFloat16 = (dataType == ge::DT_FLOAT16);
    const int64_t dtypeSize = isFloat16 ? HALF_SIZE : TYPE_SIZE;

    // 4. 计算 blockFactor / ubFactor
    //    对齐粒度取 32 字节（Ascend DMA 粒度），
    //    保证 DataCopy 的 DMA 粒度要求。
    const int64_t alignNum = DMA_ALIGN_BYTES / dtypeSize;
    OP_CHECK_IF(alignNum <= 0, OP_LOGE(context, "invalid align unit"), return ge::GRAPH_FAILED);
    const int64_t alignFactor = alignNum * BUFFER_NUM;

    // 核数不能超过「按对齐粒度切分后能覆盖的数据块数」，否则会出现空核
    int64_t usedCoreNum = coreNum;
    const int64_t coreNumByData = CeilDiv(totalNum, alignNum);
    if (usedCoreNum > coreNumByData) {
        usedCoreNum = coreNumByData;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // ★ UB 预算必须与 kernel 侧 InitBuffer 的实际申请严格一致（见 op_kernel/abs.h）：
    //     float32: in(T) + out(T) = 2 * 4 = 8 字节/元素
    //     float16: in(T) + out(T) = 2 * 2 = 4 字节/元素
    //   即 bytesPerElement = 2 * dtypeSize —— kernel 侧 float32 / float16 都直接
    //   对输入 dtype 调 AscendC::Abs，没有 fp32 中间缓冲（官方 asc-devkit 源码
    //   dav_c220/kernel_operator_vec_unary_impl.h 对 Abs 断言支持 half 与 float）。
    const int64_t bytesPerElement = 2 * dtypeSize;
    int64_t ubFactorMax = (static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES) / bytesPerElement;
    // 向下对齐到对齐单位的整数倍（手写，避免依赖未经核对的 FloorAlign）
    ubFactorMax = (ubFactorMax / alignFactor) * alignFactor;
    if (ubFactorMax < alignFactor) {
        ubFactorMax = alignFactor;
    }

    // 每个核负责的元素个数：先均分，再向上对齐到 32 字节的整数倍，
    // 保证每个核的 DMA 搬运都是整数个 DMA 单元（末尾核的截断尾块由 kernel 侧用
    // DataCopyPad 处理）。
    const int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignFactor);
    // 单个 tile 的元素个数：不超过一个核的数据量，且受 UB 预算约束。
    const int64_t ubFactor = blockFactor < ubFactorMax ? blockFactor : ubFactorMax;

    // 5. 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    // 显式声明写入长度，不依赖 GetTilingData<T>() 的隐式 SetDataSize
    // （与第八期已合入实现 tanh_tiling.cpp、第十期 exp_tiling.cpp 的写法一致）。
    context->GetRawTilingData()->SetDataSize(sizeof(AbsTilingData));

    // 6. 设置核数：总数据被 blockFactor 切成多少份就需要多少个核
    context->SetBlockDim(static_cast<uint32_t>(CeilDiv(totalNum, blockFactor)));

    // 7. 选择 tilingKey：0 -> float16，1 -> float32
    //    与 op_kernel/abs_tiling_key.h 中的 ABS_TPL_SCH_MODE_0/1 一一对应，
    //    官方 host 单测期望 float32 的 tilingKey == 1。
    uint64_t tilingKey;
    if (isFloat16) {
        tilingKey = GET_TPL_TILING_KEY(ABS_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(ABS_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAbs([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AbsCompileInfo {};

IMPL_OP_OPTILING(Abs).Tiling(AbsTilingFunc).TilingParse<AbsCompileInfo>(TilingParseForAbs);

} // namespace optiling
