/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 *
 * tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * 说明：不使用 tanh 高阶 API，全部由 Exp / Muls / Sub / Add / Div 等小算子拼接实现。
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
        // 为计算补齐临时变量的初始化：存放 exp(-x)（后续被复用为分子 / 分母），
        // 与输入输出队列分开申请，保证 Compute 执行期间该块 UB 一直有效。
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(DT_X));
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
        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        LocalTensor<DT_X> tmpLocal = tmpBuffer.Get<DT_X>();

        // yLocal = exp(x)
        Exp(yLocal, xLocal, this->tileLength);

        // tmpLocal = exp(-x)，由 exp(-x) = exp(x * (-1)) 两步小算子拼接得到
        DT_X negOne = static_cast<DT_X>(-1.0f);
        Muls(tmpLocal, xLocal, negOne, this->tileLength);
        Exp(tmpLocal, tmpLocal, this->tileLength);

        // yLocal = exp(x) - exp(-x)，即分子
        Sub(yLocal, yLocal, tmpLocal, this->tileLength);

        // tmpLocal = 2 * exp(-x)
        Add(tmpLocal, tmpLocal, tmpLocal, this->tileLength);

        // tmpLocal = (exp(x) - exp(-x)) + 2 * exp(-x) = exp(x) + exp(-x)，即分母
        Add(tmpLocal, yLocal, tmpLocal, this->tileLength);

        // yLocal = 分子 / 分母 = tanh(x)
        Div(yLocal, yLocal, tmpLocal, this->tileLength);

        // 结果写回输出队列，交由 CopyOut 搬出
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
    // 为计算补齐临时变量：一个 VECCALC 临时 buffer，用于存放 exp(-x) 以及复用的分子 / 分母
    TBuf<QuePosition::VECCALC> tmpBuffer;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsTanh
#endif // TANH_H