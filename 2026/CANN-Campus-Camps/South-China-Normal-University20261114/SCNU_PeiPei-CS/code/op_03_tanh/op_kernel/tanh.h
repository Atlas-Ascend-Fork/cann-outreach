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

// 计算过程需要的 float32 中间向量个数：
// xf（输入转 fp32）、neg（-x）、expX（e^x）、expNegX（e^-x）、
// numerator（分子）、denominator（分母）、result（结果）
constexpr int32_t TMP_BUFFER_NUM = 7;

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
        // 为计算补齐临时变量：所有中间结果统一用 float32 存放，
        // 这样 float16 输入也会在 fp32 上计算，避免 exp/div 在 fp16 下精度损失过大
        pipe.InitBuffer(tmpBuffer, TMP_BUFFER_NUM * this->tileLength * sizeof(float));
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

        // 从临时缓冲区切出 7 个 float32 向量
        LocalTensor<float> tmp = tmpBuffer.Get<float>();
        LocalTensor<float> xf = tmp;
        LocalTensor<float> neg = tmp[this->tileLength];
        LocalTensor<float> expX = tmp[2 * this->tileLength];
        LocalTensor<float> expNegX = tmp[3 * this->tileLength];
        LocalTensor<float> numerator = tmp[4 * this->tileLength];
        LocalTensor<float> denominator = tmp[5 * this->tileLength];
        LocalTensor<float> result = tmp[6 * this->tileLength];

        // 1. Float16 输入先转 float32；Float32 输入直接搬一份，统一后续计算路径
        if constexpr (sizeof(DT_X) == sizeof(half)) {
            Cast(xf, xLocal, RoundMode::CAST_NONE, this->tileLength);
        } else {
            Adds(xf, xLocal, 0.0f, this->tileLength);
        }
        PipeBarrier<PIPE_V>();

        // 2. 把 x 截断到 [-40, 40]：exp(40) 仍在 float32 范围内不会溢出成 inf，
        //    而 tanh 在 |x| > 40 时早已饱和到 ±1，截断不影响结果
        Maxs(xf, xf, -40.0f, this->tileLength);
        PipeBarrier<PIPE_V>();
        Mins(xf, xf, 40.0f, this->tileLength);
        PipeBarrier<PIPE_V>();

        // 3. 用小算子拼出 tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        Muls(neg, xf, -1.0f, this->tileLength);
        PipeBarrier<PIPE_V>();
        Exp(expX, xf, this->tileLength);
        PipeBarrier<PIPE_V>();
        Exp(expNegX, neg, this->tileLength);
        PipeBarrier<PIPE_V>();
        Sub(numerator, expX, expNegX, this->tileLength);
        PipeBarrier<PIPE_V>();
        Add(denominator, expX, expNegX, this->tileLength);
        PipeBarrier<PIPE_V>();
        Div(result, numerator, denominator, this->tileLength);
        PipeBarrier<PIPE_V>();

        // 4. 写回：Float16 需要把 float32 结果舍入回 half
        if constexpr (sizeof(DT_X) == sizeof(half)) {
            Cast(yLocal, result, RoundMode::CAST_RINT, this->tileLength);
        } else {
            Adds(yLocal, result, 0.0f, this->tileLength);
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
    // 为计算补齐临时变量：统一放 float32 中间结果
    TBuf<TPosition::VECCALC> tmpBuffer;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H
