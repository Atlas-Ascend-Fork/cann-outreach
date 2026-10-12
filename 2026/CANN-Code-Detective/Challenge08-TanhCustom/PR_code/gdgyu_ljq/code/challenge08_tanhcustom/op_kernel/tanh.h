/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // TBuf<TPosition::VECCALC> tmpBuf;  // 临时UB buffer，存放expX、expNX、tmp
    TBuf<TPosition::VECCALC> tmpBuf0; // expX
    TBuf<TPosition::VECCALC> tmpBuf1; // expNX
    TBuf<TPosition::VECCALC> tmpBuf2; // tmp中间结果

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    int64_t offset = AscendC::GetBlockIdx() * tilingData->blockFactor;
    
    blockLength_ = tilingData->blockFactor;
    // 防止最后一个核越界：如果 blockFactor 超过了剩余数据，则截断
    if (offset + blockLength_ > tilingData->totalNum) {
        blockLength_ = tilingData->totalNum - offset;
    }
    
    if (blockLength_ <= 0) {
        return;
    }
    // 直接使用对齐后的 ubFactor
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    // tmpBuf 分配3份 ubLength_ 的T空间：expX, expNX, tmp
    // pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(T) * 3);
    pipe.InitBuffer(tmpBuf0, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    
    // 从 tmpBuf 切三块UB空间：expX、expNX、tmpBuf
    AscendC::LocalTensor<T> expX = tmpBuf0.Get<T>();
    AscendC::LocalTensor<T> expNX = tmpBuf1.Get<T>();
    AscendC::LocalTensor<T> tmp = tmpBuf2.Get<T>();

    // 计算Tanh公式
    // 先算 tmp = -1.0 * x
    // AscendC::Muls<T>(tmp, xLocal, (-1.0f), currentNum);
    T neg_one = T(-1.0f);
    AscendC::Muls<T>(tmp, xLocal, neg_one, currentNum);

    // expX = exp(x); expNX = exp(tmp)
    AscendC::Exp<T>(expX, xLocal, currentNum);
    AscendC::Exp<T>(expNX, tmp, currentNum);

    // tmp = expX - expNX
    AscendC::Sub<T>(tmp, expX, expNX, currentNum);
    // expX = expX + expNX
    AscendC::Add<T>(expX, expX, expNX, currentNum);

    // y = tmp / expX
    AscendC::Div<T>(yLocal, tmp, expX, currentNum);

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    // TODO: 实现 Process 逻辑
    if (blockLength_ <= 0) {
        return;
    }
    
    int64_t loopCount = blockLength_ / ubLength_;
    int64_t tailNum = blockLength_ % ubLength_;

    for (int64_t i = 0; i < loopCount; i++) {
        CopyIn(i, ubLength_);
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }
    
    // 处理尾块
    if (tailNum > 0) {
        CopyIn(loopCount, tailNum);
        Compute(tailNum);
        CopyOut(loopCount, tailNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
