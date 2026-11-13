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
        // 为计算分配临时 UB 缓冲区（分别存放 exp(x)、exp(-x)、分子）
        pipe.InitBuffer(tmpBufExp, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tmpBufExpNeg, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tmpBufNum, this->tileLength * sizeof(DT_X));
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
        LocalTensor<DT_X> expLocal = tmpBufExp.Get<DT_X>();
        LocalTensor<DT_X> expNegLocal = tmpBufExpNeg.Get<DT_X>();
        LocalTensor<DT_X> numLocal = tmpBufNum.Get<DT_X>();

        // exp(x)
        AscendC::Exp(expLocal, xLocal, this->tileLength);
        // exp(-x)：先计算 -x 再求 exp，避免使用低精度的 Reciprocal
        AscendC::Muls(expNegLocal, xLocal, static_cast<DT_X>(-1.0f), this->tileLength);
        AscendC::Exp(expNegLocal, expNegLocal, this->tileLength);
        // 分子 = exp(x) - exp(-x)
        AscendC::Sub(numLocal, expLocal, expNegLocal, this->tileLength);
        // 分母 = exp(x) + exp(-x)，复用 expLocal 存放
        AscendC::Add(expLocal, expLocal, expNegLocal, this->tileLength);
        // y = 分子 / 分母，使用高精度 Div（Reciprocal 精度不足，无法通过 1e-4 阈值）
        AscendC::Div(yLocal, numLocal, expLocal, this->tileLength);

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
    // 计算过程所需临时缓冲区
    TBuf<TPosition::VECCALC> tmpBufExp;    // exp(x)
    TBuf<TPosition::VECCALC> tmpBufExpNeg; // exp(-x)
    TBuf<TPosition::VECCALC> tmpBufNum;    // 分子

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H
