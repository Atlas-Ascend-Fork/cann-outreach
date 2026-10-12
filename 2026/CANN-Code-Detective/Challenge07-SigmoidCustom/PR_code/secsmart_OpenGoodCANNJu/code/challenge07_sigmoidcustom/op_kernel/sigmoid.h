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

constexpr int32_t BUFFER_NUM = 1; // 每队列 buffer 块数（双缓冲）

template <typename DT_X>
class Sigmoid {
public:
    __aicore__ inline Sigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
    {
        // 单核处理长度与核内分片长度均由 tiling 计算得出
        this->blockLength = tilingData->blockFactor;
        this->tileLength = tilingData->ubFactor;
        this->tileNum = this->blockLength / this->tileLength;
        // 每个核只处理自己的 GM 分片，按 blockIdx 偏移
        xGM.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGM.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // 为输入/输出队列分配 UB 内存
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        // 数值稳定的临时缓存：内部统一以 float 计算（fp16 直接算 exp 会溢出）
        pipe.InitBuffer(xBufF32, this->tileLength * sizeof(float));
        pipe.InitBuffer(tBufF32, this->tileLength * sizeof(float));
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
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, xGM[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        // 内部统一用 float 计算，保证 fp16 的数值精度
        AscendC::LocalTensor<float> xF = xBufF32.Get<float>();
        AscendC::LocalTensor<float> tF = tBufF32.Get<float>();
        if constexpr (AscendC::IsSameType<DT_X, half>::value) {
            // float16 -> float32
            AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
        } else {
            // float 输入：UB 内拷贝到 float 工作区
            AscendC::Muls(xF, xLocal, 1.0f, this->tileLength);
        }
        // 数值稳定的 Sigmoid：y = 0.5 + 0.5 * tanh(x / 2)
        // （tanh 对任意 x 饱和收敛，不会像 exp(-x) 在大负数处溢出）
        AscendC::Muls(tF, xF, 0.5f, this->tileLength);
        AscendC::Tanh(tF, tF, this->tileLength);
        AscendC::Muls(tF, tF, 0.5f, this->tileLength);
        AscendC::Adds(tF, tF, 0.5f, this->tileLength);
        // 写回输出 dtype
        if constexpr (AscendC::IsSameType<DT_X, half>::value) {
            AscendC::Cast(yLocal, tF, AscendC::RoundMode::CAST_RINT, this->tileLength); // float -> float16
        } else {
            AscendC::Muls(yLocal, tF, 1.0f, this->tileLength);
        }
        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        DataCopy(yGM[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> xBufF32;
    TBuf<TPosition::VECCALC> tBufF32;

    GlobalTensor<DT_X> xGM;
    GlobalTensor<DT_X> yGM;

    uint32_t totalNum;    // 输入总元素数
    uint32_t blockLength; // 单核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsSigmoid
#endif // SIGMOID_H