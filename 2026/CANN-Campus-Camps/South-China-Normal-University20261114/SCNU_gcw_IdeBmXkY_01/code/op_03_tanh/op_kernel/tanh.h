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
        // 为计算补齐临时变量：统一使用 float 进行中间计算，保证 float16 的精度
        pipe.InitBuffer(tmpBufferExpX, this->tileLength * sizeof(float));   // exp(x)
        pipe.InitBuffer(tmpBufferExpNegX, this->tileLength * sizeof(float)); // exp(-x)
        pipe.InitBuffer(tmpBufferTmp, this->tileLength * sizeof(float));     // 临时：float 输入 / 分母
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

        LocalTensor<float> expX = tmpBufferExpX.Get<float>();
        LocalTensor<float> expNegX = tmpBufferExpNegX.Get<float>();
        LocalTensor<float> tmp = tmpBufferTmp.Get<float>();

        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        if constexpr (IsSameType<DT_X, float>::value) {
            // exp(x)
            AscendC::Exp(expX, xLocal, this->tileLength);
            // exp(-x)
            AscendC::Muls(expNegX, xLocal, static_cast<float>(-1), this->tileLength);
            AscendC::Exp(expNegX, expNegX, this->tileLength);
        } else {
            // float16 输入先转为 float，避免小算子拼接过程中的精度损失
            AscendC::Cast(tmp, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
            // exp(x)
            AscendC::Exp(expX, tmp, this->tileLength);
            // exp(-x)
            AscendC::Muls(expNegX, tmp, static_cast<float>(-1), this->tileLength);
            AscendC::Exp(expNegX, expNegX, this->tileLength);
        }

        // 分母 = exp(x) + exp(-x)
        AscendC::Add(tmp, expX, expNegX, this->tileLength);
        // 分子 = exp(x) - exp(-x)
        AscendC::Sub(expX, expX, expNegX, this->tileLength);
        // 结果 = 分子 / 分母
        AscendC::Div(expX, expX, tmp, this->tileLength);

        if constexpr (IsSameType<DT_X, float>::value) {
            AscendC::Adds(yLocal, expX, static_cast<float>(0), this->tileLength);
        } else {
            // float 结果转回 float16
            AscendC::Cast(yLocal, expX, AscendC::RoundMode::CAST_RINT, this->tileLength);
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
    // 为计算补齐临时变量（统一 float 中间计算）
    TBuf<TPosition::VECCALC> tmpBufferExpX;
    TBuf<TPosition::VECCALC> tmpBufferExpNegX;
    TBuf<TPosition::VECCALC> tmpBufferTmp;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H

