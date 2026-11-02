
/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

// 启用 DoubleBuffer
constexpr int32_t BUFFER_NUM = 2;

// 每次处理的最大元素数
constexpr int64_t MAX_TILE_LENGTH = 2048;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum);

private:
    TPipe pipe;

    // 输入双缓冲队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;

    // 输出双缓冲队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    // 全局内存
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    // 当前核处理的元素数
    int64_t blockLength_ = 0;

    // 每次 UB 处理的元素数
    int64_t ubLength_ = 0;
};


// ==========================================
// 1. Init
// ==========================================

template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    // 当前 AI Core 编号
    int64_t blockIdx =
        static_cast<int64_t>(GetBlockIdx());

    // 当前核的起始元素位置
    int64_t startOffset =
        blockIdx * tilingData->blockFactor;

    // 当前核实际需要处理的元素数量
    int64_t remain =
        tilingData->totalNum - startOffset;

    blockLength_ = tilingData->blockFactor;

    if (remain < blockLength_) {
        blockLength_ = remain;
    }

    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    // 每次 UB 处理的元素数
    ubLength_ = tilingData->ubFactor;

    // 控制单次 UB 分块大小
    if (ubLength_ > MAX_TILE_LENGTH) {
        ubLength_ = MAX_TILE_LENGTH;
    }

    // 设置 GM 输入输出地址
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + startOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + startOffset,
        blockLength_);

    // 初始化输入双缓冲队列
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    // 初始化输出双缓冲队列
    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}


// ==========================================
// 2. CopyIn
// ==========================================

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    // 从输入队列申请 LocalTensor
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    // 计算当前数据块在 GM 中的偏移
    int64_t offset =
        progress * ubLength_;

    // 从 GM 搬运到 UB
    DataCopy(
        xLocal,
        inputGMX[offset],
        static_cast<uint32_t>(currentNum));

    // 数据入队
    inputQueueX.EnQue(xLocal);
}


// ==========================================
// 3. Compute
// ==========================================

template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    // 获取输入数据
    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();

    // 申请输出缓冲区
    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();

    // ReLU: y = max(0, x)
    AscendC::Relu(
        yLocal,
        xLocal,
        static_cast<uint32_t>(currentNum));

    // 输出数据入队
    outputQueueY.EnQue(yLocal);

    // 释放输入缓冲区
    inputQueueX.FreeTensor(xLocal);
}


// ==========================================
// 4. CopyOut
// ==========================================

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    // 获取输出数据
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    // 计算输出偏移
    int64_t offset =
        progress * ubLength_;

    // 从 UB 搬运回 GM
    DataCopy(
        outputGMY[offset],
        yLocal,
        static_cast<uint32_t>(currentNum));

    // 释放输出缓冲区
    outputQueueY.FreeTensor(yLocal);
}


// ==========================================
// 5. Process
// ==========================================

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    // 计算当前核所需的循环次数
    int64_t loopCount =
        (blockLength_ + ubLength_ - 1)
        / ubLength_;

    // 分块执行
    for (int64_t i = 0; i < loopCount; i++) {

        // 当前数据块的起始位置
        int64_t offset = i * ubLength_;

        // 当前数据块的实际长度
        int64_t currentNum = ubLength_;

        // 最后一个数据块可能不足 ubLength_
        if (offset + currentNum > blockLength_) {
            currentNum = blockLength_ - offset;
        }

        // 1. 数据搬入
        CopyIn(i, currentNum);

        // 2. ReLU 计算
        Compute(currentNum);

        // 3. 数据搬出
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu

#endif // RELU_H
