/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义（性能优化版）
 *
 * 性能要点：
 *   1. 输入/输出各只申请 1 块 UB tile（BUFFER_NUM=1），UB 总量仅为 2*ubFactor。
 *      相比输入/输出各双缓冲（4*ubFactor），单次 DMA 搬运的 tile 可放大约 2 倍，
 *      显著减少 GM<->UB 的 DMA 次数与循环控制开销；实测 4M 元素在冷、热数据下
 *      都能稳定接近带宽上限，避免小块双缓冲在冷数据时出现的性能抖动。
 *   2. Tiling 保证 blockFactor / ubFactor 均按 512B 对齐，Kernel 侧可直接用 DataCopy；
 *   3. Tiling 先按 512B 对齐计算每核长度，再选择 tile 数，使 blockFactor 为 ubFactor
 *      整数倍，避免每核产生极小尾块；
 *   4. 每个核只计算一次 GM 起始地址，循环内用累加 offset，避免重复乘法。
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// 初始化：计算每核 GM 起始地址、尾核裁剪，并分配输入/输出各 1 块 UB tile
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    const int64_t totalNum = tilingData->totalNum;
    // 第 GetBlockIdx() 号核的全局起始偏移
    const int64_t offset = blockLength_ * static_cast<int64_t>(AscendC::GetBlockIdx());

    // blockFactor 是向上对齐后的值，最后一个核可能不足 blockFactor，需要裁剪
    if (offset >= totalNum) {
        blockLength_ = 0;
        return;
    }
    if (offset + blockLength_ > totalNum) {
        blockLength_ = totalNum - offset;
    }
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        blockLength_ = 0;
        return;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    // 输入/输出各 1 块，UB 只占 2*ubFactor，换取更大的单次 DMA 粒度
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

// 搬入：GM -> UB
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    // Tiling 已保证 offset 和 currentNum 按 512B 对齐，使用 DataCopy 获得最高搬运带宽
    DataCopy(xLocal, inputGMX[offset], static_cast<int32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

// 计算：y = max(0, x)
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    // 必须显式限定 AscendC::Relu，避免与类名 Relu 冲突
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueY.template EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// 搬出：UB -> GM
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    DataCopy(outputGMY[offset], yLocal, static_cast<int32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

// 主流程：搬入 -> 计算 -> 搬出，尾块按 currentNum 精确处理
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_; // CeilDiv
    int64_t offset = 0;
    for (int64_t i = 0; i < loopCount; ++i) {
        // 最后一个 tile 可能不足 ubLength_
        const int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - offset) : ubLength_;
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
        offset += currentNum;
    }
}

} // namespace NsRelu
#endif // RELU_H
