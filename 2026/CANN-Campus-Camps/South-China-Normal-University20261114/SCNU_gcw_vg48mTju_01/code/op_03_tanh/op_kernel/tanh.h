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
        // TODO 考生自行补齐 为计算补齐临时变量的初始化
        // 三块 FP32 临时空间：输入、两个指数分量
        pipe.InitBuffer(tmpBuf, 3 * this->tileLength * sizeof(float));
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
        // TODO 考生自行补齐 计算逻辑
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        const uint32_t n = this->tileLength;

        // FP16 转 FP32 后进行小算子运算，提高精度
        LocalTensor<float> fX = tmpBuf.Get<float>();
        LocalTensor<float> a = fX[n];
        LocalTensor<float> b = fX[2 * n];

        if constexpr (IsSameType<DT_X, half>::value) {
            AscendC::Cast(fX, xLocal, RoundMode::CAST_NONE, n);
        } else {
            AscendC::Adds(fX, xLocal, 0.0f, n);
        }
        PipeBarrier<PIPE_V>();

        // a = exp(2 * min(x, 0))
        // b = exp(-2 * max(x, 0))
        AscendC::Mins(a, fX, 0.0f, n);
        AscendC::Maxs(b, fX, 0.0f, n);
        PipeBarrier<PIPE_V>();

        AscendC::Muls(a, a, 2.0f, n);
        AscendC::Muls(b, b, -2.0f, n);
        PipeBarrier<PIPE_V>();

        AscendC::Exp(a, a, n);
        AscendC::Exp(b, b, n);
        PipeBarrier<PIPE_V>();

        // fX = (a - b) / (a + b)
        AscendC::Sub(fX, a, b, n);
        PipeBarrier<PIPE_V>();
        AscendC::Add(a, a, b, n);
        PipeBarrier<PIPE_V>();
        AscendC::Div(fX, fX, a, n);
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<DT_X, half>::value) {
            AscendC::Cast(yLocal, fX, RoundMode::CAST_RINT, n);
        } else {
            AscendC::Adds(yLocal, fX, 0.0f, n);
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
    // TODO 考生自行补齐 为计算补齐临时变量
    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H
