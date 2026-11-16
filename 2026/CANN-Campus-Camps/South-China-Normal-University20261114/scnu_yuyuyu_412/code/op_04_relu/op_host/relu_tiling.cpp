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
// 每核单次搬运的最小字节数（性能旋钮）：
//   取值越大 -> 用的核越少、每核单块的 DMA 越长（单次搬运效率更高、但并行度下降）
//   注意：该规则只在小张量上生效（见 SMALL_TENSOR_BYTES），大张量一律用满核保证并行度
constexpr int64_t MIN_BYTES_PER_CORE = 4096;
// "小张量"阈值：小于该总字节数才合并核、提高每核搬运量（小数据下 DMA 固定开销占主导）
constexpr int64_t SMALL_TENSOR_BYTES = 65536;
// 单核数据量超过该字节数时，强制把一次搬运拆成 ≥2 块：
//   只有 ≥2 块时，"写回上一块(MTE3)" 才能与 "搬入下一块(MTE2)"、VEC 计算重叠，
//   否则搬入与搬出必然串行，带宽只能跑一半
constexpr int64_t PIPELINE_MIN_BYTES = 16384;
// 需与 op_kernel/relu.h 中的 BUFFER_NUM 保持一致
constexpr int64_t BUFFER_NUM = 2;

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

    // ------------------------------------------------------------------
    // 1. 输入 dtype：fp16/bf16 走 mode0（half 实例），fp32 走 mode1（float 实例）
    // ------------------------------------------------------------------
    ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    bool isFp16 = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    int64_t typeSize = isFp16 ? 2 : TYPE_SIZE;
    // 32B 对齐对应的元素个数：fp32 为 8，fp16 为 16
    int64_t alignElem = (TYPE_SIZE * 8) / typeSize;

    // ------------------------------------------------------------------
    // 2. 总元素数
    // ------------------------------------------------------------------
    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    gert::Shape xShape = EnsureNotScalar(xShapePtr->GetStorageShape());
    int64_t totalNum = xShape.GetShapeSize();
    if (totalNum < 0) {
        totalNum = 0;
    }

    // ------------------------------------------------------------------
    // 3. 多核切分：blockDim / blockFactor
    // ------------------------------------------------------------------
    int64_t blockDim = coreNum;
    int64_t blockFactor = 0;
    if (totalNum > 0) {
        // 下限一：保证每个核至少能拿到 1 个 32B 对齐块，避免开大量空核
        int64_t maxUsefulCoreNum = CeilDiv(totalNum, alignElem);
        // 下限二：仅小张量才合并核、把每核搬运量抬到 MIN_BYTES_PER_CORE 以上；
        //        大张量保持满核，并行度优先
        int64_t totalBytes = totalNum * typeSize;
        if (totalBytes <= SMALL_TENSOR_BYTES) {
            int64_t bytesUsefulCoreNum = CeilDiv(totalBytes, MIN_BYTES_PER_CORE);
            if (bytesUsefulCoreNum < 1) {
                bytesUsefulCoreNum = 1;
            }
            if (maxUsefulCoreNum > bytesUsefulCoreNum) {
                maxUsefulCoreNum = bytesUsefulCoreNum;
            }
        }
        if (blockDim > maxUsefulCoreNum) {
            blockDim = maxUsefulCoreNum;
        }
        if (blockDim < 1) {
            blockDim = 1;
        }
        // blockFactor 按 32B 对齐：保证每个核的 GM 起始地址对齐，kernel 才能走 DataCopy 快路径
        blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignElem);
    } else {
        blockDim = 1;
        blockFactor = 0;
    }

    // ------------------------------------------------------------------
    // 4. 单次 UB 循环处理的元素数 ubFactor
    //    每元素占用 UB 字节：输入 BUFFER_NUM 份 + 输出 BUFFER_NUM 份
    //    = 2 * BUFFER_NUM * typeSize
    // ------------------------------------------------------------------
    int64_t ubBytesPerElem = 2 * BUFFER_NUM * typeSize;
    int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize), ubBytesPerElem), alignElem);
    if (ubFactor < alignElem) {
        ubFactor = alignElem;
    }
    if (totalNum > 0) {
        // 单核内不需要超过 blockFactor，向上对齐到 32B，保证 DataCopyPad 有补齐空间
        int64_t maxUbFactor = CeilAlign(blockFactor, alignElem);
        if (ubFactor > maxUbFactor) {
            ubFactor = maxUbFactor;
        }
        // 大张量：强制每核 ≥2 块，让 load / compute / store 三级流水真正重叠
        if (blockFactor * typeSize > PIPELINE_MIN_BYTES) {
            // 注意：CeilDiv/CeilAlign 是模板函数，两个实参类型必须一致（字面量 2 要显式转 int64_t）
            int64_t halfUb = CeilAlign(CeilDiv(blockFactor, static_cast<int64_t>(2)), alignElem);
            if (halfUb >= alignElem && halfUb < ubFactor) {
                ubFactor = halfUb;
            }
        }
    } else {
        ubFactor = alignElem;
    }

    // ------------------------------------------------------------------
    // 5. 回填 tiling 数据并配置启动核数
    // ------------------------------------------------------------------
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // ------------------------------------------------------------------
    // 6. 根据输入 dtype 选择 tilingKey（与 kernel 的 schMode 模板参数对应）
    // ------------------------------------------------------------------
    uint64_t tilingKey;
    if (isFp16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
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
