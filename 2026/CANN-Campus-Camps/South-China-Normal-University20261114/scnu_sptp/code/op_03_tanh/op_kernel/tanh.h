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
#include <type_traits>

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1; // 数据量小，单缓冲整块处理，减少循环/同步标量开销
constexpr float CLIP_MIN = -8.8f;  // tanh 输入截断下界，防止 exp 溢出
constexpr float CLIP_MAX = 8.8f;   // tanh 输入截断上界

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
        // 计算临时变量：tanh 使用 e^(2x) 中间结果，统一提升到 float 精度计算。
        // float 分支只需 1 个 buffer（分子直接写 yLocal）；float16 需要 2 个
        pipe.InitBuffer(tmpBufDeno, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBufNumer, this->tileLength * sizeof(float));
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
        LocalTensor<float> tmpDeno = tmpBufDeno.Get<float>();
        LocalTensor<float> tmpNumer = tmpBufNumer.Get<float>();

        if constexpr (std::is_same_v<DT_X, float>) {
            // 公式 y = tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
            Mins(tmpNumer, xLocal, CLIP_MAX, this->tileLength);
            Maxs(tmpNumer, tmpNumer, CLIP_MIN, this->tileLength);
            Muls(tmpNumer, tmpNumer, 2.0f, this->tileLength);
            Exp(tmpNumer, tmpNumer, this->tileLength);
            Adds(tmpDeno, tmpNumer, 1.0f, this->tileLength);
            Adds(tmpNumer, tmpNumer, -1.0f, this->tileLength);
            Div(yLocal, tmpNumer, tmpDeno, this->tileLength);
        } else {
            // float16 输入：先转换为 float 计算，保证 exp 精度，再转回输出
            Cast(tmpNumer, xLocal, RoundMode::CAST_NONE, this->tileLength);
            Mins(tmpNumer, tmpNumer, CLIP_MAX, this->tileLength);
            Maxs(tmpNumer, tmpNumer, CLIP_MIN, this->tileLength);
            Muls(tmpNumer, tmpNumer, 2.0f, this->tileLength);
            Exp(tmpNumer, tmpNumer, this->tileLength);
            Adds(tmpDeno, tmpNumer, 1.0f, this->tileLength);
            Adds(tmpNumer, tmpNumer, -1.0f, this->tileLength);
            Div(tmpDeno, tmpNumer, tmpDeno, this->tileLength);
            Cast(yLocal, tmpDeno, RoundMode::CAST_NONE, this->tileLength);
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
    // 计算临时变量：float 精度的中间 buffer（分母 e^(2x)+1、分子 e^(2x)-1）
    TBuf<TPosition::VECOUT> tmpBufDeno;
    TBuf<TPosition::VECOUT> tmpBufNumer;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H
