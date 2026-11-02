/*!
 * \file sqrt.h
 * \brief Sqrt 算子 kernel 类定义
 */

#ifndef SQRT_H
#define SQRT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sqrt_tiling_data.h"
#include "sqrt_tiling_key.h"

namespace NsSqrt {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Sqrt {
public:
    __aicore__ inline Sqrt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData);
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
__aicore__ inline void Sqrt<T>::Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
}

template <typename T>
__aicore__ inline void Sqrt<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
}

template <typename T>
__aicore__ inline void Sqrt<T>::Process()
{
    // TODO: 实现 Process 逻辑
}

} // namespace NsSqrt
#endif // SQRT_H
