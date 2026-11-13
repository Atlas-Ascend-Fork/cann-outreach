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
        
        // 申请计算所需的临时变量 Buffer，大小为 2 个 tileLength，用于存放 exp(x) 和 exp(-x)
        pipe.InitBuffer(tempQueue, BUFFER_NUM, this->tileLength * sizeof(DT_X) * 2);
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
        LocalTensor<DT_X> tmpLocal = tempQueue.AllocTensor<DT_X>();

        // 划分临时变量
        LocalTensor<DT_X> expX = tmpLocal;
        LocalTensor<DT_X> expNegX = tmpLocal[this->tileLength];

        // 1. 计算 exp(x) 存入 expX
        AscendC::Exp(expX, xLocal, this->tileLength);
        
        // 2. 计算 exp(-x) 存入 expNegX。
        // 为了防止原地修改 xLocal 导致异步流水线数据依赖冲突，用 yLocal 中转取反
        AscendC::Muls(yLocal, xLocal, (DT_X)(-1), this->tileLength); // yLocal = -x
        AscendC::Exp(expNegX, yLocal, this->tileLength);

        // 3. 计算分子：exp(x) - exp(-x) 存入 yLocal
        AscendC::Sub(yLocal, expX, expNegX, this->tileLength);
        
        // 4. 计算分母：exp(x) + exp(-x) 存入 expX
        AscendC::Add(expX, expX, expNegX, this->tileLength);
        
        // 5. 计算最终结果：(分子) / (分母) 存入 yLocal
        AscendC::Div(yLocal, yLocal, expX, this->tileLength);

        // 计算完成，入队并释放
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
        tempQueue.FreeTensor(tmpLocal);
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
    TQue<QuePosition::VECCALC, BUFFER_NUM> tempQueue;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

} // namespace NsTanh
#endif // TANH_H
