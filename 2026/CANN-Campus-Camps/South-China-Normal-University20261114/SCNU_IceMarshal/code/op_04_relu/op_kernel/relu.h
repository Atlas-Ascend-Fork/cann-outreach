/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 *
 * 计算式：y[i] = max(0, x[i])，逐元素、无跨元素依赖，属于典型访存密集型算子。
 *
 * 性能相关设计：
 *  1. 多核并行：tiling 按 AIV 核数切分，每核只处理自己的 blockFactor 个元素，核间无同步；
 *  2. 双缓冲 + 预取：输入、输出 TQue 各 2 份 buffer，核内循环提前一个 tile 下发搬入，
 *     MTE2（GM->UB）、V（Relu）、MTE3（UB->GM）三条流水可跨 tile 重叠；
 *  3. 对齐搬运：每核起始偏移与 ubFactor 均按 32B 对齐，常规 tile 走 DataCopy 整块搬运，
 *     仅末尾非 32B 对齐的尾块用 DataCopyPad 兜底；
 *  4. half 直接计算：Relu 在 half 下是精确的 max(0, x)，无需升精度到 float，
 *     省掉 Cast 指令和临时 buffer，UB 占用减半、单次可搬运的数据量翻倍。
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;       // 输入/输出队列各 2 份 buffer，构成双缓冲流水
constexpr int64_t UB_ALIGN_BYTES = 32;  // UB 与 GM 间搬运的基本块长度（字节）

// InitBuffer 的 buffer 字节数需按 32B 对齐
__aicore__ inline uint32_t AlignUpTo32(int64_t bytes)
{
    return static_cast<uint32_t>(((bytes + UB_ALIGN_BYTES - 1) / UB_ALIGN_BYTES) * UB_ALIGN_BYTES);
}

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline int64_t TileLength(int64_t index) const;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 每个核处理 [blockIdx * blockFactor, blockIdx * blockFactor + blockLength) 区间，
    // tiling 保证该偏移按 32B 对齐
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t offset = blockIdx * tilingData->blockFactor;

    blockLength_ = tilingData->totalNum - offset;
    if (blockLength_ > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    }
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    // tiling 侧保证 ubFactor 为 32B 对齐的正整数；空 Tensor 时兜底为 1，避免 0 长度 buffer
    ubLength_ = tilingData->ubFactor > 0 ? tilingData->ubFactor : 1;

    if (blockLength_ > 0) {
        inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, static_cast<uint32_t>(blockLength_));
        outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, static_cast<uint32_t>(blockLength_));
    }

    const uint32_t queueBytes = AlignUpTo32(ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, queueBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, queueBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    const int64_t byteSize = currentNum * static_cast<int64_t>(sizeof(T));

    // 常规 tile 的 GM 地址与长度都按 32B 对齐，使用 DataCopy 获得最高的搬运效率；
    // 仅最后一个长度不对齐的尾块使用 DataCopyPad，保证不越界读写
    if ((byteSize & (UB_ALIGN_BYTES - 1)) == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(byteSize), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // y = max(0, x)：由矢量 Relu 接口完成，half/float 均为精确计算
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    const int64_t byteSize = currentNum * static_cast<int64_t>(sizeof(T));

    if ((byteSize & (UB_ALIGN_BYTES - 1)) == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(byteSize), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline int64_t Relu<T>::TileLength(int64_t index) const
{
    const int64_t remain = blockLength_ - index * ubLength_;
    return remain > ubLength_ ? ubLength_ : remain;
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;

    // 预取第一个 tile
    CopyIn(0, TileLength(0));
    for (int64_t i = 0; i < loopCount; i++) {
        // 先把下一个 tile 的搬运下发出去：MTE2 搬运 tile i+1 的同时，
        // V 计算 tile i、MTE3 搬出 tile i-1，三条流水线重叠
        if (i + 1 < loopCount) {
            CopyIn(i + 1, TileLength(i + 1));
        }
        const int64_t currentNum = TileLength(i);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
