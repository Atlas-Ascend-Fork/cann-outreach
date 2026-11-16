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

constexpr int32_t BUFFER_NUM = 2; // 每队列 buffer 块数（双缓冲）

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        // 从 Host 侧读取每核的元素数和每次搬运的元素数
        blockLength = static_cast<uint32_t>(tilingData->blockFactor);
        tileLength = static_cast<uint32_t>(tilingData->ubFactor);

        // 设置每个核的 Global Memory 起始地址
        uint32_t offset = blockLength * AscendC::GetBlockIdx();
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength);

        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));

        // 为计算分配 FP32 临时空间
        uint32_t floatBytes = tileLength * sizeof(float);
        pipe.InitBuffer(absBuffer, floatBytes);
        pipe.InitBuffer(expBuffer, floatBytes);
        pipe.InitBuffer(tempBuffer1, floatBytes);
        pipe.InitBuffer(tempBuffer2, floatBytes);
        // 每个元素占 1 位，掩码空间按 32 字节对齐
        pipe.InitBuffer(maskBuffer, ((tileLength + 255) / 256) * 32);

        if constexpr (sizeof(DT_X) == sizeof(half)) {
            pipe.InitBuffer(castBuffer, floatBytes);
        }
    }

    __aicore__ inline void Process()
    {
        // 每核将自己的数据分成等长小块，逐块搬入、计算、写回
        int32_t loopCount = blockLength / tileLength;
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
        DataCopy(xLocal, inputGMX[progress * tileLength], tileLength);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();

        LocalTensor<float> absX = absBuffer.Get<float>();
        LocalTensor<float> expValue = expBuffer.Get<float>();
        LocalTensor<float> temp1 = tempBuffer1.Get<float>();
        LocalTensor<float> temp2 = tempBuffer2.Get<float>();
        LocalTensor<uint8_t> mask = maskBuffer.Get<uint8_t>();

        LocalTensor<float> xFloat;
        LocalTensor<float> yFloat = temp1;
        if constexpr (sizeof(DT_X) == sizeof(half)) {
            xFloat = castBuffer.Get<float>();
            Cast(xFloat, xLocal, RoundMode::CAST_NONE, tileLength);
        } else {
            xFloat = xLocal;
            yFloat = yLocal;
        }

        // 先算 e = exp(-2 * abs(x))，再求 (1 - e) / (1 + e)
        Abs(absX, xFloat, tileLength);
        Muls(temp1, absX, -2.0f, tileLength);
        Exp(expValue, temp1, tileLength);
        Adds(temp2, expValue, 1.0f, tileLength);
        Muls(temp1, expValue, -1.0f, tileLength);
        Adds(expValue, temp1, 1.0f, tileLength);
        Div(temp1, expValue, temp2, tileLength);

        // x 小于 0 时，把结果取负
        Muls(expValue, temp1, -1.0f, tileLength);
        CompareScalar(mask, xFloat, 0.0f, CMPMODE::LT, tileLength);
        Select(temp2, mask, expValue, temp1, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileLength);

        // x 很接近 0 时用 tanh(x) ≈ x，减小 1 - e 的舍入误差
        constexpr float SMALL_X = 1.0f / 256.0f;
        CompareScalar(mask, absX, SMALL_X, CMPMODE::LT, tileLength);
        Select(yFloat, mask, xFloat, temp2, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileLength);

        if constexpr (sizeof(DT_X) == sizeof(half)) {
            Cast(yLocal, yFloat, RoundMode::CAST_RINT, tileLength);
        }

        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * tileLength], yLocal, tileLength);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<TPosition::VECCALC> absBuffer;
    TBuf<TPosition::VECCALC> expBuffer;
    TBuf<TPosition::VECCALC> tempBuffer1;
    TBuf<TPosition::VECCALC> tempBuffer2;
    TBuf<TPosition::VECCALC> maskBuffer;
    TBuf<TPosition::VECCALC> castBuffer;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H