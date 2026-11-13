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
        // 每核处理的数据长度（多核并行）
        this->blockLength = tilingData->totalLength / AscendC::GetBlockNum();
        this->tileNum = tilingData->tileNum;
        // 单次处理的元素数（每块长度）
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        // 为计算过程的临时变量分配 UB 内存
        pipe.InitBuffer(tbufExpX, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tbufExpNegX, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tbufNumer, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tbufDenom, this->tileLength * sizeof(DT_X));
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

        // 计算过程所需临时变量
        LocalTensor<DT_X> expX = tbufExpX.Get<DT_X>();       // e^x
        LocalTensor<DT_X> expNegX = tbufExpNegX.Get<DT_X>(); // e^-x
        LocalTensor<DT_X> numer = tbufNumer.Get<DT_X>();     // 分子
        LocalTensor<DT_X> denom = tbufDenom.Get<DT_X>();     // 分母

        // tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        Exp(expX, xLocal, this->tileLength);                 // e^x
        Muls(expNegX, xLocal, static_cast<DT_X>(-1.0), this->tileLength); // -x
        Exp(expNegX, expNegX, this->tileLength);             // e^-x
        Sub(numer, expX, expNegX, this->tileLength);         // 分子 e^x - e^-x
        Add(denom, expX, expNegX, this->tileLength);         // 分母 e^x + e^-x
        Div(yLocal, numer, denom, this->tileLength);         // y = 分子 / 分母

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
    // 计算过程所需临时变量（UB 上的独立缓冲，逻辑位置 VECCALC）
    TBuf<TPosition::VECCALC> tbufExpX;       // e^x
    TBuf<TPosition::VECCALC> tbufExpNegX;    // e^-x
    TBuf<TPosition::VECCALC> tbufNumer;      // 分子 e^x - e^-x
    TBuf<TPosition::VECCALC> tbufDenom;      // 分母 e^x + e^-x

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H