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
constexpr int64_t BLOCK_BYTE_SIZE = 32;   // 256bit，对齐搬运的块粒度
constexpr int64_t UB_PAD_SIZE = 32;       // 尾块非对齐补齐，避免 UB 越界访问
constexpr int64_t HALF_TEMP_NUM = 2;      // half 场景以 float 中间结果计算，需要分子/分母两块空间
constexpr float ARG_SCALE = 2.0f;         // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
constexpr float EXP_ARG_MAX = 17.6f;      // exp 参数限幅，防止上溢为 inf 后出现 inf / inf
constexpr float EXP_ARG_MIN = -17.6f;

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
    __aicore__ inline void CalTanh(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                   const LocalTensor<float>& aux, int64_t count);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> tempBuffer;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t alignElements_ = 0;
    int64_t startIdx_ = 0;     // 本核负责处理的元素起点
    int64_t coreDataNum_ = 0;  // 本核负责处理的元素个数
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    alignElements_ = BLOCK_BYTE_SIZE / static_cast<int64_t>(sizeof(T));

    startIdx_ = static_cast<int64_t>(GetBlockIdx()) * blockLength_;
    const int64_t remain = tilingData->totalNum - startIdx_;
    coreDataNum_ = remain > 0 ? (remain < blockLength_ ? remain : blockLength_) : 0;

    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T) + UB_PAD_SIZE);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
    if constexpr (sizeof(T) < sizeof(float)) {
        pipe.InitBuffer(tempBuffer,
                        static_cast<uint32_t>(ubLength_ * sizeof(float) * HALF_TEMP_NUM + UB_PAD_SIZE));
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xTensor = inputQueueX.AllocTensor<T>();
    const int64_t alignNum = currentNum / alignElements_ * alignElements_;
    if (alignNum > 0) {
        DataCopy(xTensor, inputGMX[progress], static_cast<uint32_t>(alignNum));
    }
    if (currentNum > alignNum) { // 尾块按非对齐通路搬运，读取长度精确到元素个数
        DataCopyExtParams copyParams{1, static_cast<uint32_t>((currentNum - alignNum) * sizeof(T)), 0, 0, 0};
        DataCopyPad(xTensor[alignNum], inputGMX[progress + alignNum], copyParams, DataCopyPadExtParams<T>{});
    }
    inputQueueX.EnQue(xTensor);
}

template <typename T>
__aicore__ inline void Tanh<T>::CalTanh(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                         const LocalTensor<float>& aux, int64_t count)
{
    const int32_t cnt = static_cast<int32_t>(count);
    Muls(dst, src, ARG_SCALE, cnt);
    Mins(dst, dst, EXP_ARG_MAX, cnt);
    Maxs(dst, dst, EXP_ARG_MIN, cnt);
    Exp(dst, dst, cnt);      // dst = e^(2x)
    Adds(aux, dst, 1.0f, cnt);  // aux = e^(2x) + 1
    Adds(dst, dst, -1.0f, cnt); // dst = e^(2x) - 1
    Div(dst, dst, aux, cnt);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xTensor = inputQueueX.DeQue<T>();
    LocalTensor<T> yTensor = outputQueueY.AllocTensor<T>();
    const int32_t cnt = static_cast<int32_t>(currentNum);
    if constexpr (sizeof(T) < sizeof(float)) { // 低精度先升精度到 float 计算，避免中间结果精度损失
        LocalTensor<float> tempTensor = tempBuffer.AllocTensor<float>();
        Cast<float, T>(tempTensor, xTensor, RoundMode::CAST_NONE, cnt);
        inputQueueX.FreeTensor(xTensor);
        CalTanh(tempTensor, tempTensor, tempTensor[ubLength_], currentNum);
        Cast<T, float>(yTensor, tempTensor, RoundMode::CAST_RINT, cnt);
        tempBuffer.FreeTensor(tempTensor);
    } else {
        CalTanh(yTensor, xTensor, xTensor, currentNum); // 输入缓冲区已读取完毕，复用其空间存放分母
        inputQueueX.FreeTensor(xTensor);
    }
    outputQueueY.EnQue(yTensor);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yTensor = outputQueueY.DeQue<T>();
    const int64_t alignNum = currentNum / alignElements_ * alignElements_;
    if (alignNum > 0) {
        DataCopy(outputGMY[progress], yTensor, static_cast<uint32_t>(alignNum));
    }
    if (currentNum > alignNum) { // 尾块按非对齐通路搬运，写出长度精确到元素个数
        DataCopyExtParams copyParams{1, static_cast<uint32_t>((currentNum - alignNum) * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[progress + alignNum], yTensor[alignNum], copyParams);
    }
    outputQueueY.FreeTensor(yTensor);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (coreDataNum_ <= 0) {
        return;
    }
    const int64_t loopCount = (coreDataNum_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        const int64_t progress = startIdx_ + i * ubLength_;
        const int64_t remain = coreDataNum_ - i * ubLength_;
        const int64_t currentNum = remain < ubLength_ ? remain : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
