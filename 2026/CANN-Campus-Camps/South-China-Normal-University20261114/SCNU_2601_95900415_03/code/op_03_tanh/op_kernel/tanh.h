/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义（单波次 + 单Exp极速版）
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1; // 单块突发传输，消除多轮队列同步开销

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        // 用位移替代标量除法，8核单波次每核处理 2048 个元素
        uint32_t blockNum = AscendC::GetBlockNum();
        this->blockLength = (blockNum == 8) ? (tilingData->totalLength >> 3) : (tilingData->totalLength / blockNum);
        this->tileNum = tilingData->tileNum;
        this->tileLength = (this->tileNum == 1) ? this->blockLength : (this->blockLength / this->tileNum / BUFFER_NUM);

        // 设置每个核的 Global Memory 起始地址
        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, this->blockLength);

        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));

        // 仅申请 2 个临时缓冲区，最小化 InitBuffer 开销
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(float));
        if constexpr (sizeof(DT_X) == 2) {
            pipe.InitBuffer(xFloatBuf, this->tileLength * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        CopyIn(0);
        Compute();
        CopyOut(0);
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

        LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

        // 采用 tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)，仅需 1 次 Exp 且无需 Duplicate
        if constexpr (sizeof(DT_X) == 2) {
            LocalTensor<float> xFp32 = xFloatBuf.Get<float>();

            Cast(xFp32, xLocal, RoundMode::CAST_NONE, this->tileLength);
            Muls(tmp1, xFp32, 2.0f, this->tileLength);    // tmp1 = 2x
            Exp(tmp1, tmp1, this->tileLength);            // tmp1 = exp(2x)
            Adds(tmp2, tmp1, -1.0f, this->tileLength);    // tmp2 = exp(2x) - 1
            Adds(tmp1, tmp1, 1.0f, this->tileLength);     // tmp1 = exp(2x) + 1
            Div(xFp32, tmp2, tmp1, this->tileLength);     // xFp32 = (exp(2x) - 1) / (exp(2x) + 1)
            Cast(yLocal, xFp32, RoundMode::CAST_RINT, this->tileLength);
        } else {
            Muls(tmp1, xLocal, 2.0f, this->tileLength);   // tmp1 = 2x
            Exp(tmp1, tmp1, this->tileLength);            // tmp1 = exp(2x)
            Adds(tmp2, tmp1, -1.0f, this->tileLength);    // tmp2 = exp(2x) - 1
            Adds(tmp1, tmp1, 1.0f, this->tileLength);     // tmp1 = exp(2x) + 1
            Div(yLocal, tmp2, tmp1, this->tileLength);    // yLocal = (exp(2x) - 1) / (exp(2x) + 1)
        }

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
    TBuf<TPosition::VECCALC> tmpBuf1;
    TBuf<TPosition::VECCALC> tmpBuf2;
    TBuf<TPosition::VECCALC> xFloatBuf;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H