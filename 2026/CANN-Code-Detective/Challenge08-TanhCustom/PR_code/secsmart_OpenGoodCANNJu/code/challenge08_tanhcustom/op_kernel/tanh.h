/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 *
 * 直接调用 Ascend C 高精度 Tanh 激活指令（支持 float16 / float32），
 * BUFFER_NUM = 2 双缓冲 + CopyIn/Compute/CopyOut 流水并行。
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲：搬运与计算流水并行

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        // 单核处理长度与核内分片长度均由 tiling 计算得出
        this->blockLength = tilingData->blockFactor;
        this->ubLength = tilingData->ubFactor;
        this->tileNum = this->blockLength / this->ubLength;
        // 每个核只处理自己的 GM 分片，按 blockIdx 偏移
        xGM.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGM.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // 为输入/输出队列分配 UB 内存
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ubLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
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
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, xGM[progress * this->ubLength], this->ubLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        // 逐元素 Tanh：y = tanh(x)，高精度激活指令（加 AscendC:: 前缀避免与类名冲突）
        AscendC::Tanh(yLocal, xLocal, this->ubLength);
        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        DataCopy(yGM[progress * this->ubLength], yLocal, this->ubLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<DT_X> xGM;
    GlobalTensor<DT_X> yGM;

    uint32_t totalNum;    // 输入总元素数
    uint32_t blockLength; // 单核处理的元素数
    uint32_t ubLength;    // 单次搬入 UB 的元素数
    uint32_t tileNum;     // 单核内分块数
};

} // namespace NsTanh
#endif // TANH_H