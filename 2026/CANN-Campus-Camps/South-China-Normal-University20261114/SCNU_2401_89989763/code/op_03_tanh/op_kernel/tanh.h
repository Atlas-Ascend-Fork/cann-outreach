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
        this->blockLength = static_cast<uint32_t>(tilingData->blockFactor);
        // 单次处理的元素数（每块长度）
        this->tileLength = static_cast<uint32_t>(tilingData->ubFactor);
        // 循环次数 = tileNum × BUFFER_NUM，故 tileNum 由 blockLength / tileLength / BUFFER_NUM 得到
        this->tileNum = (this->tileLength == 0) ? 0 : (this->blockLength / this->tileLength / BUFFER_NUM);
        if (this->tileLength == 0 || this->tileNum == 0) {
            this->blockLength = 0;
            return;
        }
        // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
        int64_t start = tilingData->blockFactor * static_cast<int64_t>(AscendC::GetBlockIdx());
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + start, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + start, this->blockLength);
        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        // 为计算补齐临时变量：两块 VECCALC 临时 buffer，大小与单块数据一致
        pipe.InitBuffer(tmpBufA, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tmpBufB, this->tileLength * sizeof(DT_X));
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
        // 钳位上限：fp32 取 15（e^30 ≈ 1e13 不溢出），fp16 取 5（e^10 ≈ 22026 < 65504 不溢出）
        constexpr float CLAMP_MAX = (sizeof(DT_X) == 4) ? 15.0f : 5.0f;

        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        LocalTensor<DT_X> tmpA = tmpBufA.Get<DT_X>();
        LocalTensor<DT_X> tmpB = tmpBufB.Get<DT_X>();

        // 1) 钳位到 [-CLAMP_MAX, CLAMP_MAX]：tanh 在范围外已饱和到 ±1，钳位不改变结果
        Mins(tmpA, xLocal, (DT_X)CLAMP_MAX, this->tileLength);
        Maxs(tmpA, tmpA, (DT_X)(0.0f - CLAMP_MAX), this->tileLength);

        // 2) tmpA = exp(2x)
        Muls(tmpA, tmpA, (DT_X)2, this->tileLength);
        Exp(tmpA, tmpA, this->tileLength);

        // 3) tmpB = exp(2x) - 1
        Adds(tmpB, tmpA, (DT_X)(-1), this->tileLength);

        // 4) tmpA = exp(2x) + 1
        Adds(tmpA, tmpA, (DT_X)1, this->tileLength);

        // 5) y = (exp(2x) - 1) / (exp(2x) + 1)
        Div(yLocal, tmpB, tmpA, this->tileLength);

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
    // 为计算补齐临时变量
    TBuf<TPosition::VECCALC> tmpBufA;
    TBuf<TPosition::VECCALC> tmpBufB;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H