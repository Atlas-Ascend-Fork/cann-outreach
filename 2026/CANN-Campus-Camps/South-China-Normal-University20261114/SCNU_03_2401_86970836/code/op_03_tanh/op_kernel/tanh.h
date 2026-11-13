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
        // 直接读取 host 预计算好的分块参数（核内免除法）
        this->blockLength = tilingData->blockLength; // 每核处理的元素数
        this->tileNum = tilingData->tileNum;         // 单核内循环块数
        this->tileLength = tilingData->tileLength;   // 每块元素数

        // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        // 为计算补齐临时变量（exp(x) 与 exp(-x) 中间结果）的 UB 内存
        pipe.InitBuffer(tmpQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tmpQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 单核内循环 tileNum 次，每次处理 tileLength 个元素
        int32_t loopCount = this->tileNum;
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
        // 临时变量：分别存放 exp(x) 与 exp(-x)
        LocalTensor<DT_X> tmpExpX = tmpQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> tmpExpNegX = tmpQueueY.AllocTensor<DT_X>();

        // 计算 tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // 1. tmpExpX = exp(x)
        AscendC::Exp(tmpExpX, xLocal, this->tileLength);
        // 2. tmpExpNegX = exp(-x) = exp(x * (-1))
        AscendC::Muls(tmpExpNegX, xLocal, static_cast<DT_X>(-1), this->tileLength);
        AscendC::Exp(tmpExpNegX, tmpExpNegX, this->tileLength);

        // 3. yLocal = exp(x) - exp(-x)  （分子）
        AscendC::Sub(yLocal, tmpExpX, tmpExpNegX, this->tileLength);
        // 4. tmpExpX = exp(x) + exp(-x)  （分母，复用 tmpExpX）
        AscendC::Add(tmpExpX, tmpExpX, tmpExpNegX, this->tileLength);
        // 5. yLocal = 分子 / 分母
        AscendC::Div(yLocal, yLocal, tmpExpX, this->tileLength);

        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
        tmpQueueX.FreeTensor(tmpExpX);
        tmpQueueY.FreeTensor(tmpExpNegX);
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
    // 为计算补齐临时变量：exp(x) 与 exp(-x) 中间结果队列
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueueX;
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内循环块数
    uint32_t tileLength;  // 每块元素数
};

} // namespace NsTanh
#endif // TANH_H
