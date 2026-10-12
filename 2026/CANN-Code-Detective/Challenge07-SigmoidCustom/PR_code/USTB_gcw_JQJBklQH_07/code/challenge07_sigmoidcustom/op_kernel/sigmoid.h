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
        // 分配 fp32 辅助 buffer：全 1 向量（Div 的分子）与中间结果，保证子正常区间精度
        pipe.InitBuffer(oneBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(mid1Buf, this->tileLength * sizeof(float));
        pipe.InitBuffer(mid2Buf, this->tileLength * sizeof(float));
        LocalTensor<float> oneLocal = oneBuf.Get<float>();
        Duplicate(oneLocal, 1.0f, this->tileLength);
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
        // 逐元素计算 Sigmoid：y = 1 / (1 + exp(-x))，中间过程统一使用 fp32
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        LocalTensor<float> oneLocal = oneBuf.Get<float>();
        LocalTensor<float> mid1 = mid1Buf.Get<float>();
        LocalTensor<float> mid2 = mid2Buf.Get<float>();
        if constexpr (IsSameType<DT_X, half>::value) {
            // fp16 输入：先升位到 fp32，避免 fp16 指令在子正常区间的精度损失
            Cast(mid1, xLocal, RoundMode::CAST_NONE, this->tileLength);
            AscendC::Muls(mid1, mid1, -1.0f, this->tileLength);           // mid1 = -x
            AscendC::Exp(mid2, mid1, this->tileLength);                   // mid2 = exp(-x)
            AscendC::Adds(mid2, mid2, 1.0f, this->tileLength);            // mid2 = 1 + exp(-x)
            AscendC::Div(mid1, oneLocal, mid2, this->tileLength);         // mid1 = 1 / (1 + exp(-x))
            // 按最近偶数舍入转回 fp16
            Cast(yLocal, mid1, RoundMode::CAST_RINT, this->tileLength);
        } else {
            AscendC::Muls(mid1, xLocal, -1.0f, this->tileLength);         // mid1 = -x
            AscendC::Exp(mid2, mid1, this->tileLength);                   // mid2 = exp(-x)
            AscendC::Adds(mid2, mid2, 1.0f, this->tileLength);            // mid2 = 1 + exp(-x)
            AscendC::Div(yLocal, oneLocal, mid2, this->tileLength);       // yLocal = 1 / (1 + exp(-x))
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
    TBuf<QuePosition::VECCALC> oneBuf;
    TBuf<QuePosition::VECCALC> mid1Buf;
    TBuf<QuePosition::VECCALC> mid2Buf;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsSigmoid
#endif // SIGMOID_H
