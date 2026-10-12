/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 每队列 buffer 块数（双缓冲）

template <typename DT_X>
class Sigmoid {
public:
    __aicore__ inline Sigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
    {
        this->blockLength = tilingData->totalLength / AscendC::GetBlockNum();
        this->tileNum = tilingData->tileNum;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        inputGMX.SetGlobalBuffer(
            (__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMY.SetGlobalBuffer(
            (__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        // float16 也在 float 中完成指数与倒数计算；两段空间分别保存分母和结果。
        pipe.InitBuffer(computeBuffer, this->tileLength * sizeof(float) * 2);
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
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress * this->tileLength], this->tileLength);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();

        LocalTensor<float> workLocal = computeBuffer.Get<float>();
        LocalTensor<float> denominator = workLocal;

        if constexpr (std::is_same_v<DT_X, float>) {
            Muls(denominator, xLocal, -1.0f, this->tileLength);
            Mins(denominator, denominator, 80.0f, this->tileLength);
            Exp(denominator, denominator, this->tileLength);
            Adds(denominator, denominator, 1.0f, this->tileLength);
            Reciprocal(yLocal, denominator, this->tileLength);

            // Reciprocal 的基础精度不足，执行一次 r = r * (2 - d * r) 牛顿迭代。
            Mul(denominator, denominator, yLocal, this->tileLength);
            Muls(denominator, denominator, -1.0f, this->tileLength);
            Adds(denominator, denominator, 2.0f, this->tileLength);
            Mul(yLocal, yLocal, denominator, this->tileLength);
        } else {
            LocalTensor<float> result = workLocal[this->tileLength];
            Cast(denominator, xLocal, RoundMode::CAST_NONE, this->tileLength);
            Muls(denominator, denominator, -1.0f, this->tileLength);
            Mins(denominator, denominator, 80.0f, this->tileLength);
            Exp(denominator, denominator, this->tileLength);
            Adds(denominator, denominator, 1.0f, this->tileLength);
            Reciprocal(result, denominator, this->tileLength);

            Mul(denominator, denominator, result, this->tileLength);
            Muls(denominator, denominator, -1.0f, this->tileLength);
            Adds(denominator, denominator, 2.0f, this->tileLength);
            Mul(result, result, denominator, this->tileLength);
            Cast(yLocal, result, RoundMode::CAST_RINT, this->tileLength);
        }

        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * this->tileLength], yLocal, this->tileLength);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> computeBuffer;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

} // namespace NsSigmoid
#endif // SIGMOID_H
