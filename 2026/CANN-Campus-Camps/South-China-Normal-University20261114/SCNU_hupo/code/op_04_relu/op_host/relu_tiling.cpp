/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */
#include <algorithm>
#include <cmath>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"
namespace optiling {
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
constexpr uint32_t WS_SYS_SIZE = 0U;
/* ---------------- Tiling 常量 ---------------- */
// 元素字节数
constexpr int64_t ELEM_BYTES_FP32 = 4;
constexpr int64_t ELEM_BYTES_FP16 = 2;
// 每个 AIV 核最少处理的数据量（字节）：仅作为兜底下限
constexpr int64_t MIN_BYTES_PER_CORE = 1024;
// 块数硬上限（A/B 开关）：UB 限制的是"每块大小"而不是"块数"；块数越多，每块的固定开销
//（block 调度 + 核内初始化，实测 ~0.19us/块）累加越多。大 shape 先试 24。
constexpr int64_t MAX_BLOCK_NUM = 24;
// 多核切分时单核元素数的对齐粒度（字节）：512B。
// 910B 实测：GM 偏移按 512B 对齐可跑满 DMA 带宽，只做 32B 对齐时约只有 70%。
// 每核起始地址 = blockIdx * blockFactor，故 blockFactor 必须按 512B 对齐，
// 才能让全部核的搬运都落在 512B 边界上。
constexpr int64_t BLOCK_ALIGN_BYTES = 512;
// ---------- 核数选取成本模型 ----------
// 实测：每启动一个 block 约 0.194us（块调度+核内初始化，随块数线性累加），
// 单核 DMA 有效带宽约 30GB/s（read+write 各算一次）。故总时间
//     T(N) = A + c*N + 2*B/(N*BW)
// 最优块数 N* = sqrt(2*B/(BW*c))，即数据量小要少开核、数据量大要开满核。
constexpr double PER_BLOCK_COST_US = 0.194;
constexpr double DMA_BYTES_PER_US = 30000.0;
// UB 切分对齐字节数：取 512B —— 与 DMA/L2 的搬运粒度一致，
// 保证每核、每块（tile）的起始地址都落在 512B 边界上（256B 对齐时 fp16 的 tile 起点
// 只有 256B 对齐，整条流会持续跨线，实测有效带宽掉约 30%）
constexpr int64_t UB_ALIGN_BYTES = 512;
// 单条 Vector 指令最大 repeat 次数（repeatTimes 为 uint8_t）
constexpr int64_t MAX_REPEAT_TIMES = 255;
// Vector 单次 repeat 处理的字节数
constexpr int64_t BYTES_PER_REPEAT = 256;
// UB 预留空间（TPipe/框架自身占用），避免 UB 超限
constexpr int64_t UB_RESERVED_BYTES = 8192;
// 双缓冲块数：与 kernel 侧 BUFFER_NUM 保持一致
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
// Buffer 份数：输入 x 与输出 y 各一份
constexpr int64_t BUFFER_CNT = 2;
// 单块大小再细分倍数：1 = 用满 UB；2 = 单块减半（每核块数翻倍）。用于验证"每块固定开销"
// 究竟来自流水填充/排空（会随单块变小而下降）还是块调度（不受影响）。
constexpr int64_t UB_TILE_SPLIT = 2;
// 每个核的目标块数：实测（910B）块数 4→9 时流水填充/排空占比下降，9 块以后再增加块数
// 因每块 DMA 启动与同步开销反而变慢，故取 8（fp32/fp16 大 shape 落在 8~9 块）
constexpr int64_t TARGET_TILE_NUM = 8;
// 单块最小字节数：小 shape 下延迟主导，DMA 不拆碎（保证每核就是 1 块、只发 2 次 DMA）
constexpr int64_t MIN_TILE_BYTES = 8192;
static const gert::Shape g_vec_1_shape = {1};
static inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape) {
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
// 把输入 shape 展平为一维元素总数（Relu 逐元素，与 shape 无关）
static ge::graphStatus GetInputTotalNum(gert::TilingContext* context, int64_t& totalNum)
{
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t num = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim < 0, OP_LOGE(context, "dynamic shape is not supported by relu tiling"),
                    return ge::GRAPH_FAILED);
        num *= dim;
    }
    totalNum = num;
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
    int64_t totalNum = 0;
    OP_CHECK_IF(
        GetInputTotalNum(context, totalNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetInputTotalNum error"),
        return ge::GRAPH_FAILED);
    // 根据输入 dtype 选择 tilingKey：kernel 侧据此实例化 half / float 两套实现
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    const bool isFp16 = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    const int64_t dtypeSize = isFp16 ? ELEM_BYTES_FP16 : ELEM_BYTES_FP32;
    context->SetTilingKey(
        isFp16 ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0) : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    // 空 tensor：不搬运也不计算，仅需保证 kernel 能正常启动后立即退出
    if (totalNum <= 0) {
        tiling->totalNum = totalNum;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }
    // ---------- 1. 多核切分 ----------
    // 按 T(N) = A + c*N + 2B/(N*BW) 的最优解选核数：
    // 小 shape 少开核（块启动开销主导），大 shape 开满核（带宽主导）
    const int64_t totalBytes = totalNum * dtypeSize;
    const double optCore = std::sqrt(2.0 * static_cast<double>(totalBytes) /
                                     (DMA_BYTES_PER_US * PER_BLOCK_COST_US));
    int64_t usedCoreNum = static_cast<int64_t>(optCore + 0.5);
    usedCoreNum = std::max(usedCoreNum, static_cast<int64_t>(1));
    usedCoreNum = std::min(usedCoreNum, coreNum);
    // 兜底：每核至少 MIN_BYTES_PER_CORE 字节
    const int64_t maxCoreByBytes = std::max(CeilDiv(totalBytes, MIN_BYTES_PER_CORE), static_cast<int64_t>(1));
    usedCoreNum = std::min(usedCoreNum, maxCoreByBytes);
    // 块数硬上限：省掉每块的固定开销（块调度 + 核内初始化）
    usedCoreNum = std::min(usedCoreNum, MAX_BLOCK_NUM);
    // 单核处理量向上对齐到 512B（保证每核 GM 首地址 512B 对齐）
    const int64_t blockAlignElem = BLOCK_ALIGN_BYTES / dtypeSize;
    const int64_t blockFactor = std::max(
        CeilDiv(CeilDiv(totalNum, usedCoreNum), blockAlignElem) * blockAlignElem,
        blockAlignElem);
    // 实际启动的核数（尾核可能不足 blockFactor，由 kernel 侧裁剪）
    const int64_t blockNum = CeilDiv(totalNum, blockFactor);
    // ---------- 2. UB 切分 ----------
    // UB 预算 = 双缓冲 × (输入 + 输出) 共 DOUBLE_BUFFER_NUM * BUFFER_CNT 块 buffer
    int64_t ubBudget = static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES;
    if (ubBudget <= 0) {
        ubBudget = static_cast<int64_t>(ubSize);
    }
    const int64_t elemAlign = UB_ALIGN_BYTES / dtypeSize;   // fp32:64, fp16:128
    const int64_t bufferBytes = DOUBLE_BUFFER_NUM * BUFFER_CNT * UB_TILE_SPLIT * dtypeSize;
    // UB 能容纳的单块上限
    const int64_t ubFactorMax = FloorAlign(ubBudget / bufferBytes, elemAlign);
    int64_t ubFactor = ubFactorMax;
    // 单次 Vector 指令的 repeat 上限：保证 ubFactor 一屏即可被单条指令覆盖
    const int64_t maxElemPerRepeatGroup = MAX_REPEAT_TIMES * (BYTES_PER_REPEAT / dtypeSize);
    ubFactor = std::min(ubFactor, maxElemPerRepeatGroup);
    // 按目标块数限制单块大小：保证每核 >= TARGET_TILE_NUM 块，
    // 摊薄双缓冲流水的填充/排空开销（每核 n 块时理论效率 n/(n+1)）
    ubFactor = std::min(ubFactor, FloorAlign(CeilDiv(blockFactor, TARGET_TILE_NUM), elemAlign));
    // 单块不得小于 MIN_TILE_BYTES（小 shape 下延迟主导，DMA 拆碎只会增加启动延迟），
    // 但下限本身又不得突破 UB 容量
    const int64_t minTileElem = std::min(std::max(MIN_TILE_BYTES / dtypeSize, elemAlign), ubFactorMax);
    ubFactor = std::max(ubFactor, minTileElem);
    // 单块无需超过单核处理量
    ubFactor = std::min(ubFactor, blockFactor);
    // 保证至少有一个对齐单元可算（blockFactor 极小时以其为准，kernel 侧会再按实际数据裁剪）
    ubFactor = std::max(ubFactor, std::min(elemAlign, blockFactor));
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(blockNum));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
struct ReluCompileInfo {};
IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);
} // namespace optiling