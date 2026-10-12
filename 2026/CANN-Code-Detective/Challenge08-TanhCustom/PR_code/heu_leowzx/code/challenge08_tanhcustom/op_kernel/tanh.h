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

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress);
    __aicore__ inline void CopyOut(int64_t progress);
    __aicore__ inline void Compute();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> bufx,bufy;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t tileNum = 1;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    int coreIdx = GetBlockIdx(), coreNum = GetBlockNum();
    blockLength_ = tilingData->totalNum / coreNum;
    ubLength_ = blockLength_ / BUFFER_NUM / tileNum;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockLength_ * coreIdx, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockLength_ * coreIdx, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, sizeof(T) * ubLength_);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, sizeof(T) * ubLength_);
    if constexpr (!std::is_same_v<T,float>){
        pipe.InitBuffer(bufx, sizeof(float) * ubLength_);
        pipe.InitBuffer(bufy, sizeof(float) * ubLength_);
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], ubLength_);
    inputQueueX.EnQue(xLocal);
}
/**
    (exp(2x) - 1) / (exp(2x) + 1)
*/
template <typename T>
__aicore__ inline void Tanh<T>::Compute()
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    if constexpr (std::is_same_v<T, float>) {
        Muls(xLocal, xLocal, 2.f, ubLength_);
        Exp(xLocal, xLocal, ubLength_);
        Adds(xLocal, xLocal, -1.f, ubLength_);
        Adds(yLocal, xLocal, 2.f, ubLength_);
        Div(yLocal, xLocal, yLocal, ubLength_);
    } else {
        LocalTensor<float> xFloat = bufx.Get<float>();
        LocalTensor<float> yFloat = bufy.Get<float>();
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, ubLength_);
        Muls(xFloat, xFloat, 2.f, ubLength_);
        Exp(xFloat, xFloat, ubLength_);
        Adds(xFloat, xFloat, -1.f, ubLength_);
        Adds(yFloat, xFloat, 2.f, ubLength_);
        Div(yFloat, xFloat, yFloat, ubLength_);
        Cast(yLocal, yFloat, RoundMode::CAST_FLOOR, ubLength_);
    }

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[ubLength_*progress], yLocal, ubLength_);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    // TODO: 实现 Process 逻辑
    // int lp = BUFFER_NUM * tileNum; default value 0
    CopyIn(0);
    Compute();
    CopyOut(0);
}

} // namespace NsTanh
#endif // TANH_H
