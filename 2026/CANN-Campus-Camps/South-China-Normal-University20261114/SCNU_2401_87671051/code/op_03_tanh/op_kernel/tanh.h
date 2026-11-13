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

// 数据量小、每核一次搬完，单缓冲即可，减少队列状态机开销
constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> calcBuf; // fp16 升精度计算缓存

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t actualLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    this->totalNum_ = tilingData->totalNum;
    this->blockLength_ = tilingData->blockFactor;

    // 多核切分：每核 blockFactor 个元素，最后一核只处理剩余元素
    int64_t offset = this->blockLength_ * static_cast<int64_t>(GetBlockIdx());
    if (offset >= this->totalNum_) {
        this->actualLength_ = 0;
        return;
    }
    this->actualLength_ = this->totalNum_ - offset;
    if (this->actualLength_ > this->blockLength_) {
        this->actualLength_ = this->blockLength_;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, this->actualLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, this->actualLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->blockLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->blockLength_ * sizeof(T));

    if constexpr (sizeof(T) == 2) {
        // fp16 精度护城河：开辟双倍容量的 fp32 临时计算缓存
        pipe.InitBuffer(calcBuf, this->blockLength_ * 2 * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX, currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xDe = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    if constexpr (sizeof(T) == 2) {
        // fp16 精度防丢失：先 Cast 到 fp32，计算 Tanh，再 Cast 回来
        LocalTensor<float> xFloat = calcBuf.Get<float>();
        LocalTensor<float> yFloat = calcBuf.Get<float>(currentNum);
        Cast(xFloat, xDe, RoundMode::CAST_NONE, currentNum);
        AscendC::Tanh(yFloat, xFloat, currentNum);
        Cast(yLocal, yFloat, RoundMode::CAST_NONE, currentNum);
    } else {
        // fp32 精度足够，直接调用底层硬件 Tanh 指令
        AscendC::Tanh(yLocal, xDe, currentNum);
    }

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xDe);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t currentNum)
{
    LocalTensor<T> yDe = outputQueueY.DeQue<T>();
    DataCopy(outputGMY, yDe, currentNum);
    outputQueueY.FreeTensor(yDe);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (this->actualLength_ <= 0) {
        return;
    }
    CopyIn(this->actualLength_);
    Compute(this->actualLength_);
    CopyOut(this->actualLength_);
}

} // namespace NsTanh
#endif // TANH_H
