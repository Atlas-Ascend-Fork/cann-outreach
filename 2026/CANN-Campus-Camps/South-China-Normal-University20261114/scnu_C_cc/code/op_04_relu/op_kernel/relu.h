
/*!
 * \file relu.h
 * \brief Optimized Relu kernel V3
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t SMALL_TENSOR_SIZE = 16384;

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
        int64_t offset,
        int64_t count);

    __aicore__ inline void Compute(
        int64_t count);

    __aicore__ inline void CopyOut(
        int64_t offset,
        int64_t count);

private:
    TPipe pipe;

    // 输入输出共享 UB
    TQueBind<
        QuePosition::VECIN,
        QuePosition::VECOUT,
        1
    > dataQueue;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// 初始化
template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    const int64_t blockOffset =
        static_cast<int64_t>(GetBlockIdx()) *
        tilingData->blockFactor;

    const int64_t totalNum = tilingData->totalNum;

    // 防止空核
    if (blockOffset >= totalNum) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }

    // 当前核实际处理的数据长度
    blockLength_ = tilingData->blockFactor;

    const int64_t remaining =
        totalNum - blockOffset;

    if (blockLength_ > remaining) {
        blockLength_ = remaining;
    }

    // UB 每次计算的长度
    ubLength_ = tilingData->ubFactor;

    if (ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    // 当前核的 GM 起始地址
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + blockOffset,
        blockLength_);

    // 小张量每核只需要一个 Buffer
    // 大张量使用 Double Buffer
    const int32_t bufferNum =
        (totalNum <= SMALL_TENSOR_SIZE)
            ? 1
            : DOUBLE_BUFFER_NUM;

    pipe.InitBuffer(
        dataQueue,
        bufferNum,
        ubLength_ * sizeof(T));
}

// GM -> UB
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t offset,
    int64_t count)
{
    LocalTensor<T> local =
        dataQueue.AllocTensor<T>();

    DataCopy(
        local,
        inputGMX[offset],
        static_cast<uint32_t>(count));

    dataQueue.EnQue<
        QuePosition::GM,
        QuePosition::VECIN,
        T
    >(local);
}

// UB 原地计算
template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t count)
{
    LocalTensor<T> local =
        dataQueue.DeQue<
            QuePosition::GM,
            QuePosition::VECIN,
            T
        >();

    // 原地 ReLU
    AscendC::Relu(
        local,
        local,
        static_cast<int32_t>(count));

    dataQueue.EnQue<
        QuePosition::VECIN,
        QuePosition::VECOUT,
        T
    >(local);
}

// UB -> GM
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t offset,
    int64_t count)
{
    LocalTensor<T> local =
        dataQueue.DeQue<
            QuePosition::VECIN,
            QuePosition::VECOUT,
            T
        >();

    DataCopy(
        outputGMY[offset],
        local,
        static_cast<uint32_t>(count));

    dataQueue.FreeTensor(local);
}

// 主计算流程
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    // 快速路径：
    // 当前核的数据可以一次放入 UB
    if (blockLength_ <= ubLength_) {
        CopyIn(0, blockLength_);
        Compute(blockLength_);
        CopyOut(0, blockLength_);
        return;
    }

    // 多 Tile 路径
    // 将完整 Tile 和尾 Tile 分开处理
    const int64_t fullTiles =
        blockLength_ / ubLength_;

    const int64_t tailNum =
        blockLength_ % ubLength_;

    // 完整 Tile：避免每轮做尾块判断
    for (int64_t i = 0; i < fullTiles; ++i) {
        const int64_t offset = i * ubLength_;

        CopyIn(offset, ubLength_);
        Compute(ubLength_);
        CopyOut(offset, ubLength_);
    }

    // 尾 Tile
    if (tailNum > 0) {
        const int64_t offset =
            fullTiles * ubLength_;

        CopyIn(offset, tailNum);
        Compute(tailNum);
        CopyOut(offset, tailNum);
    }
}

} // namespace NsRelu

#endif // RELU_H
