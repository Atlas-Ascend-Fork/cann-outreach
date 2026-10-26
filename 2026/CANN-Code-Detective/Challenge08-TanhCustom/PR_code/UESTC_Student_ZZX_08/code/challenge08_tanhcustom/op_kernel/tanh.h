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
#include <type_traits>

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        blockLength = tilingData->totalLength / GetBlockNum();
        tileNum = tilingData->tileNum;
        tileLength = blockLength / tileNum / BUFFER_NUM;

        inputGM.SetGlobalBuffer((__gm__ DT_X*)x + blockLength * GetBlockIdx(), blockLength);
        outputGM.SetGlobalBuffer((__gm__ DT_X*)y + blockLength * GetBlockIdx(), blockLength);
        pipe.InitBuffer(inputQueue, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(workBuffer, 2 * tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // 循环次数 = tileNum × BUFFER_NUM（tileLength 已按 BUFFER_NUM 折半）
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> input = inputQueue.AllocTensor<DT_X>();
        DataCopy(input, inputGM[progress * tileLength], tileLength);
        inputQueue.EnQue(input);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> input = inputQueue.DeQue<DT_X>();
        LocalTensor<DT_X> output = outputQueue.AllocTensor<DT_X>();
        LocalTensor<float> work = workBuffer.Get<float>();
        LocalTensor<float> denominator = work;

        // tanh(x) = 2 / (1 + exp(-2x)) - 1. The exponential is clipped
        // only for large negative inputs to avoid float32 overflow.
        if constexpr (std::is_same_v<DT_X, float>) {
            Muls(denominator, input, -2.0f, tileLength);
            Mins(denominator, denominator, 80.0f, tileLength);
            Exp(denominator, denominator, tileLength);
            Adds(denominator, denominator, 1.0f, tileLength);
            Reciprocal(output, denominator, tileLength);

            Mul(denominator, denominator, output, tileLength);
            Muls(denominator, denominator, -1.0f, tileLength);
            Adds(denominator, denominator, 2.0f, tileLength);
            Mul(output, output, denominator, tileLength);
            Muls(output, output, 2.0f, tileLength);
            Adds(output, output, -1.0f, tileLength);
        } else {
            LocalTensor<float> result = work[tileLength];
            Cast(denominator, input, RoundMode::CAST_NONE, tileLength);
            Muls(denominator, denominator, -2.0f, tileLength);
            Mins(denominator, denominator, 80.0f, tileLength);
            Exp(denominator, denominator, tileLength);
            Adds(denominator, denominator, 1.0f, tileLength);
            Reciprocal(result, denominator, tileLength);

            Mul(denominator, denominator, result, tileLength);
            Muls(denominator, denominator, -1.0f, tileLength);
            Adds(denominator, denominator, 2.0f, tileLength);
            Mul(result, result, denominator, tileLength);
            Muls(result, result, 2.0f, tileLength);
            Adds(result, result, -1.0f, tileLength);
            Cast(output, result, RoundMode::CAST_RINT, tileLength);
        }

        outputQueue.EnQue(output);
        inputQueue.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> output = outputQueue.DeQue<DT_X>();
        DataCopy(outputGM[progress * tileLength], output, tileLength);
        outputQueue.FreeTensor(output);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<TPosition::VECCALC> workBuffer;
    GlobalTensor<DT_X> inputGM;
    GlobalTensor<DT_X> outputGM;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

} // namespace NsTanh
#endif // TANH_H

