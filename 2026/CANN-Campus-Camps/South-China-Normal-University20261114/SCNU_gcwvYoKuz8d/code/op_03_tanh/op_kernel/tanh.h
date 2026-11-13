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
        
        // 设置每个核的 Global Memory 起始地址
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        
        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        
        // TODO 考生自行补齐 为计算补齐临时变量的初始化
        // 按照公式计算 Tanh，需要额外空间存储 exp(x) 和 exp(-x)
        // 这里申请 2 倍 tileLength 大小的临时 Buffer
        pipe.InitBuffer(tmpBuffer, 2 * this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
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
        // TODO 考生自行补齐 计算逻辑
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        
        // 获取临时 Buffer 并将其切分为两块：exp(x) 和 exp(-x)
        LocalTensor<DT_X> tmpLocal = tmpBuffer.Get<DT_X>();
        LocalTensor<DT_X> expX = tmpLocal;
        LocalTensor<DT_X> expNegX = tmpLocal[this->tileLength];

        // 1. 计算 -x 并存入 expNegX 以便计算 exp(-x)
        Muls(expNegX, xLocal, (DT_X)-1.0, this->tileLength);
        
        // 2. 计算 exp(x) -> expX; 计算 exp(-x) -> expNegX
        Exp(expX, xLocal, this->tileLength);
        Exp(expNegX, expNegX, this->tileLength);

        // 3. 计算分子：(exp(x) - exp(-x))，暂存入 yLocal
        Sub(yLocal, expX, expNegX, this->tileLength);
        
        // 4. 计算分母：(exp(x) + exp(-x))，存入 expX
        Add(expX, expX, expNegX, this->tileLength);

        // 5. 计算最终结果：分子 / 分母 -> yLocal
        Div(yLocal, yLocal, expX, this->tileLength);

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
    
    // TODO 考生自行补齐 为计算补齐临时变量
    TBuf<QuePosition::VECCALC> tmpBuffer; // 声明 VECCALC 的临时变量

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

} // namespace NsTanh
#endif // TANH_H