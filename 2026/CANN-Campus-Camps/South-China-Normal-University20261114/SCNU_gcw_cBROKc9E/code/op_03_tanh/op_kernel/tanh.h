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

// BUFFER_NUM 改为 1，因为我们不再需要双缓冲流水线
constexpr int32_t BUFFER_NUM = 1; 

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        this->blockLength = tilingData->totalLength / AscendC::GetBlockNum();
        this->tileLength = this->blockLength; // 一次性搬完，不分块

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        
        // 继续使用 TQue 保证底层搬运安全
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tempBuf, this->tileLength * sizeof(DT_X)); 
    }

    __aicore__ inline void Process()
    {
        // 没有 for 循环了，直接三步走
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX, this->tileLength);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        LocalTensor<DT_X> temp = tempBuf.Get<DT_X>();

        // 计算公式：tanh(x) = 1 - 2 / (exp(2x) + 1)
        Muls(temp, xLocal, static_cast<DT_X>(2.0), this->tileLength);
        Exp(temp, temp, this->tileLength);
        Adds(temp, temp, static_cast<DT_X>(1.0), this->tileLength);
        
        Duplicate(yLocal, static_cast<DT_X>(2.0), this->tileLength);
        Div(temp, yLocal, temp, this->tileLength);
        
        Duplicate(yLocal, static_cast<DT_X>(1.0), this->tileLength);
        Sub(yLocal, yLocal, temp, this->tileLength);

        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY, yLocal, this->tileLength);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> tempBuf;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileLength;
};

} // namespace NsTanh
#endif // TANH_H