/*!
 * \file exp.h
 * \brief Exp 算子 kernel 类定义
 */

#ifndef EXP_H
#define EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Exp {
public:
    __aicore__ inline Exp(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData);
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

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Exp<T>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    int64_t blockIdx   = static_cast<int64_t>(GetBlockIdx());
    int64_t totalNum   = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;

    // 当前核处理的起始偏移与元素数
    int64_t offset = blockIdx * blockFactor;
    if (offset >= totalNum) {
        blockLength_ = 0;
    } else {
        int64_t remain = totalNum - offset;
        blockLength_ = (remain < blockFactor) ? remain : blockFactor;
    }

    // 绑定 GM 地址（从当前核负责的起始偏移开始）
    if (blockLength_ > 0) {
        inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
        outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    }

    ubLength_ = tilingData->ubFactor;

    // 队列按 ubLength_ * sizeof(T) 分配（BUFFER_NUM 份双缓冲）
    pipe.InitBuffer(inputQueueX,  BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T)));
}

template <typename T>
__aicore__ inline void Exp<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
	LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    
    // 使用 DataCopyPad 处理非对齐长度
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // 使用 AscendC 内置向量 Exp 接口，显式命名空间以区别类模板名 Exp
    ::AscendC::Exp(yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
	LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    
    // 写出时使用 DataCopyPad，只会把 currentNum 个元素写回 GM，绝不越界
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Process()
{
    // TODO: 实现 Process 逻辑
    if (blockLength_ <= 0){
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
}
}
} // namespace NsExp
#endif // EXP_H

