/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr float TANH_EXP_SCALE = 2.0f;  // exp(2x) 中的系数 2
constexpr float TANH_ONE = 1.0f;

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

    // 计算过程所需临时变量（均为 fp32 中间结果）
    TBuf<QuePosition::VECCALC> tmpWorkBuf;  // exp(2x) 的中间结果（fp16 输入时先在此承接升精度后的 x）
    TBuf<QuePosition::VECCALC> tmpDenBuf;   // 分母 exp(2x) + 1

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;

    this->ubLength_ = tilingData->ubFactor;
    this->blockLength_ = blockFactor;

    // 多核切分：最后一个核处理剩余数据
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    if (blockIdx == (blockNum - 1)) {
        this->blockLength_ = totalNum - blockFactor * blockIdx;
    }
    if (this->blockLength_ < 0) {
        this->blockLength_ = 0;
    }
    // 兜底：tiling 中 ubFactor 非法时退化为单块处理
    if (this->ubLength_ <= 0) {
        this->ubLength_ = (this->blockLength_ > 0) ? this->blockLength_ : 1;
    }

    int64_t offset = blockFactor * blockIdx;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, this->blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpWorkBuf, this->ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpDenBuf, this->ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    int64_t offset = progress * this->ubLength_;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(inputLocal, inputGMX[offset], copyParams, padParams);
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> workLocal = tmpWorkBuf.Get<float>();  // 中间结果（fp32）
    LocalTensor<float> denLocal = tmpDenBuf.Get<float>();    // 分母（fp32）
    int32_t count = static_cast<int32_t>(currentNum);

    if constexpr (std::is_same<T, float>::value) {
        // fp32 输入：无需 Cast，直接在 fp32 临时 buffer 上运算
        Muls(workLocal, inputLocal, TANH_EXP_SCALE, count);          // 2x
        Exp(workLocal, workLocal, count);                            // exp(2x)
        Adds(denLocal, workLocal, TANH_ONE, count);                  // exp(2x) + 1
        Adds(workLocal, workLocal, -TANH_ONE, count);                // exp(2x) - 1
        Div(outputLocal, workLocal, denLocal, count);                // 结果
    } else {
        // fp16 输入：升精度到 fp32 计算后再回写 fp16
        Cast(workLocal, inputLocal, RoundMode::CAST_NONE, count);
        Muls(workLocal, workLocal, TANH_EXP_SCALE, count);           // 2x
        Exp(workLocal, workLocal, count);                            // exp(2x)
        Adds(denLocal, workLocal, TANH_ONE, count);                  // exp(2x) + 1
        Adds(workLocal, workLocal, -TANH_ONE, count);                // exp(2x) - 1
        Div(workLocal, workLocal, denLocal, count);                  // 结果（fp32）
        Cast(outputLocal, workLocal, RoundMode::CAST_RINT, count);   // 回写 fp16
    }

    inputQueueX.FreeTensor(inputLocal);
    outputQueueY.EnQue<T>(outputLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    int64_t offset = progress * this->ubLength_;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(outputGMY[offset], outputLocal, copyParams);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    // 单核内按 ubLength_ 分块循环，末块按实际剩余长度处理
    int64_t loopCount = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = this->blockLength_ - i * this->ubLength_;
        if (currentNum > this->ubLength_) {
            currentNum = this->ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H