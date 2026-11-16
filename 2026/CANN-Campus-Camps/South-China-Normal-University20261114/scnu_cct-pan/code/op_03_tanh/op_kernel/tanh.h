/*!
 * \file tanh.h
 * \brief Tanh with float intermediates and bounded exponent.
 */
#ifndef TANH_H
#define TANH_H
#include "kernel_operator.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"
namespace NsTanh {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* data)
    {
        blockLength_ = data->blockFactor;
        ubLength_ = data->ubFactor;
        const int64_t offset = GetBlockIdx() * blockLength_;
        inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(expBuffer, ubLength_ * sizeof(float));
        pipe.InitBuffer(workBuffer, ubLength_ * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // Fixed shape and power-of-two tiling: no unaligned tail.
        for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
            CopyIn(offset);
            Compute();
            CopyOut(offset);
        }
    }
private:
    __aicore__ inline void CopyIn(int64_t offset)
    {
        auto x = inputQueueX.AllocTensor<T>();
        DataCopy(x, inputGMX[offset], ubLength_);
        inputQueueX.EnQue(x);
    }
    __aicore__ inline void Compute()
    {
        auto x = inputQueueX.DeQue<T>();
        auto y = outputQueueY.AllocTensor<T>();
        auto e = expBuffer.Get<float>();
        auto w = workBuffer.Get<float>();
        if constexpr (sizeof(T) == sizeof(half)) {
            Cast(w, x, RoundMode::CAST_NONE, ubLength_);
        } else {
            Adds(w, x, 0.0f, ubLength_);
        }
        // Clamping at +/-10 has <4.2e-9 mathematical saturation error.
        // It prevents exp overflow; all arithmetic is performed in float32.
        Maxs(w, w, -10.0f, ubLength_);
        Mins(w, w, 10.0f, ubLength_);
        Muls(e, w, 2.0f, ubLength_);
        Exp(e, e, ubLength_);
        Adds(e, e, 1.0f, ubLength_);
        Duplicate(w, 2.0f, ubLength_);
        Div(w, w, e, ubLength_);
        Adds(w, w, -1.0f, ubLength_);
        Muls(w, w, -1.0f, ubLength_);
        if constexpr (sizeof(T) == sizeof(half)) {
            Cast(y, w, RoundMode::CAST_RINT, ubLength_);
        } else {
            Adds(y, w, 0.0f, ubLength_);
        }
        outputQueueY.EnQue(y);
        inputQueueX.FreeTensor(x);
    }
    __aicore__ inline void CopyOut(int64_t offset)
    {
        auto y = outputQueueY.DeQue<T>();
        DataCopy(outputGMY[offset], y, ubLength_);
        outputQueueY.FreeTensor(y);
    }
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> expBuffer;
    TBuf<QuePosition::VECCALC> workBuffer;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};
} // namespace NsTanh
#endif