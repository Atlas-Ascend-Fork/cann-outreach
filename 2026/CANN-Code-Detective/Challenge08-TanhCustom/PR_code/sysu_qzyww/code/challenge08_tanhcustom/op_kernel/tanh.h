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

constexpr int32_t BUFFER_NUM = 2; // 每队列 buffer 块数（双缓冲），避免 MTE2 搬入与 VEC 计算争抢同一块 UB

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        // 每核处理的数据长度（多核并行）
        this->blockLength = tilingData->totalLength / AscendC::GetBlockNum();
        this->tileNum = tilingData->tileNum;
        // 单次处理的元素数（每块长度）
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        // 为中间结果分配 UB 内存：exp(x)、exp(2x)、分母、常数、倒数
        pipe.InitBuffer(expBuf, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(sqBuf, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(denBuf, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(twoBuf, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(recBuf, this->tileLength * sizeof(DT_X));
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
        LocalTensor<DT_X> expLocal = expBuf.Get<DT_X>();
        LocalTensor<DT_X> sqLocal = sqBuf.Get<DT_X>();
        LocalTensor<DT_X> denLocal = denBuf.Get<DT_X>();
        LocalTensor<DT_X> twoLocal = twoBuf.Get<DT_X>();
        LocalTensor<DT_X> recLocal = recBuf.Get<DT_X>();
        // y = tanh(x) = 1 - 2 / (exp(2x) + 1)，该形式在大 |x| 时不会产生 inf/inf = NaN；
        // 用高精度 Div 而非近似的 Reciprocal（vrec 查表精度只有约 2^-10）
        AscendC::Exp(expLocal, xLocal, this->tileLength);                                // exp(x)
        AscendC::Mul(sqLocal, expLocal, expLocal, this->tileLength);                     // exp(2x)
        AscendC::Adds(denLocal, sqLocal, static_cast<DT_X>(1.0), this->tileLength);      // exp(2x) + 1
        AscendC::Duplicate(twoLocal, static_cast<DT_X>(-2.0), this->tileLength);         // 常数 -2
        AscendC::Div(recLocal, twoLocal, denLocal, this->tileLength);                    // -2/(exp(2x)+1)
        AscendC::Adds(yLocal, recLocal, static_cast<DT_X>(1.0), this->tileLength);       // tanh(x)
        outputQueueY.EnQue<DT_X>(yLocal);
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
    TBuf<TPosition::VECCALC> expBuf;
    TBuf<TPosition::VECCALC> sqBuf;
    TBuf<TPosition::VECCALC> denBuf;
    TBuf<TPosition::VECCALC> twoBuf;
    TBuf<TPosition::VECCALC> recBuf;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H
