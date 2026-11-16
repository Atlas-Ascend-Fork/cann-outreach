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

constexpr int32_t BUFFER_NUM = 2;
// 单个 buffer 的最大元素数预算：按 192KB UB、双缓冲（2 队列 × 2 块）保守预留后计算，
// 防止 tiling 传入过大的 ubFactor 导致 UB 越界
constexpr int64_t MAX_UB_BYTES = 192 * 1024;
constexpr int64_t UB_RESERVE_BYTES = 8 * 1024;

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
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;

    // 每次 UB 循环处理的元素数：钳制在单 buffer 预算内并向下对齐，避免越界
    int64_t alignUnit = 32 / (int64_t)sizeof(T);
    int64_t maxTileNum =
        (MAX_UB_BYTES - UB_RESERVE_BYTES) / (2 * BUFFER_NUM * (int64_t)sizeof(T));
    maxTileNum = (maxTileNum / alignUnit) * alignUnit;
    ubLength_ = (tilingData->ubFactor < maxTileNum) ? tilingData->ubFactor : maxTileNum;
    ubLength_ = (ubLength_ / alignUnit) * alignUnit;
    if (ubLength_ <= 0) {
        ubLength_ = alignUnit;
    }

    int64_t blockNum = AscendC::GetBlockNum();
    int64_t blockIdx = AscendC::GetBlockIdx();
    // 每核基础长度 blockFactor 已 32B 对齐；未被整除的余量统一交给 0 号核
    int64_t leftover = totalNum - blockFactor * blockNum;

    // 设置每个核的 Global Memory 起始地址与处理长度（关键的多核切分逻辑）
    int64_t offset;
    if (blockIdx == 0) {
        blockLength_ = blockFactor + leftover;
        offset = 0;
    } else {
        blockLength_ = blockFactor;
        offset = blockFactor + leftover + (blockIdx - 1) * blockFactor;
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    // 为队列分配 UB 内存（双缓冲）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, currentNum);
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // 每次循环处理 ubLength_ 个元素，剩余不足一块的部分走尾部处理
    int64_t loopCount = blockLength_ / ubLength_;
    int64_t tailNum = blockLength_ - loopCount * ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        CopyIn(i, ubLength_);
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }
    if (tailNum > 0) {
        CopyIn(loopCount, tailNum);
        Compute(tailNum);
        CopyOut(loopCount, tailNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
