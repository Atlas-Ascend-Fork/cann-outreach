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
        this->blockLength = tilingData->totalNum / AscendC::GetBlockNum();
        this->tileNum = tilingData->blockFactor;
       
        // 单次处理的元素数（每块长度）
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
       
        // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
       
        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));

        // 为 Tanh 计算过程分配临时变量 Buffer (统一使用 float 防止 half 溢出)
        pipe.InitBuffer(tmp1Buf, this->tileLength * sizeof(float)); // 存放分子 (exp(x) - exp(-x))
        pipe.InitBuffer(tmp2Buf, this->tileLength * sizeof(float)); // 存放 exp(-x)
        pipe.InitBuffer(tmp3Buf, this->tileLength * sizeof(float)); // 存放分母 (exp(x) + exp(-x)) 或 float输入
    }

    __aicore__ inline void Process()
    {
        // 循环次数 = tileNum × BUFFER_NUM
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

        LocalTensor<float> tmp1 = tmp1Buf.Get<float>();
        LocalTensor<float> tmp2 = tmp2Buf.Get<float>();
        LocalTensor<float> tmp3 = tmp3Buf.Get<float>();

        // 公式: y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
       
        if constexpr (sizeof(DT_X) == 4) {
            // -------- float32 直接计算 --------
            Exp(tmp1, xLocal, this->tileLength); // tmp1 = exp(x)
            Muls(tmp2, xLocal, -1.0f, this->tileLength); // tmp2 = -x
            Exp(tmp2, tmp2, this->tileLength); // tmp2 = exp(-x)
           
            Add(tmp3, tmp1, tmp2, this->tileLength); // tmp3 = 分母
            Sub(tmp1, tmp1, tmp2, this->tileLength); // tmp1 = 分子
           
            Div(yLocal, tmp1, tmp3, this->tileLength); // y = 分子 / 分母
        } else {
            // -------- float16 升精度计算 --------
            Cast(tmp3, xLocal, RoundMode::CAST_NONE, this->tileLength); // tmp3 = (float)x
           
            Exp(tmp1, tmp3, this->tileLength); // tmp1 = exp(x)
            Muls(tmp2, tmp3, -1.0f, this->tileLength); // tmp2 = -x
            Exp(tmp2, tmp2, this->tileLength); // tmp2 = exp(-x)
           
            Add(tmp3, tmp1, tmp2, this->tileLength); // tmp3 = 分母
            Sub(tmp1, tmp1, tmp2, this->tileLength); // tmp1 = 分子
           
            Div(tmp1, tmp1, tmp3, this->tileLength); // tmp1 = 结果
           
            // 降精度回 float16 (就近舍入)
            Cast(yLocal, tmp1, RoundMode::CAST_RINT, this->tileLength);
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

    // 计算过程所需临时变量
    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> tmp2Buf;
    TBuf<TPosition::VECCALC> tmp3Buf;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum; // 单核内分块数
    uint32_t tileLength; // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H