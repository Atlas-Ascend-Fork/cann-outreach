/*!
 * \file leakyrelu.h
 * \brief Leakyrelu 算子 kernel 类定义
 *
 * 计算公式：y = max(x, 0) + negativeSlope * min(x, 0)
 * 与 ReLU 等单指令算子不同，需要临时 Tensor 完成 Max/Min/Mul/Add 复合计算
 */

#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "leakyrelu_tiling_data.h"
#include "leakyrelu_tiling_key.h"

namespace NsLeakyrelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 每队列 buffer 块数（双缓冲）

template <typename DT_X>
class Leakyrelu {
public:
    __aicore__ inline Leakyrelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
    {
        // 属性 negativeSlope 经 tilingData 下发到 kernel 侧
        this->negativeSlope = tilingData->negativeSlope;
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
        // 临时 Tensor：zeroBuf 先存 0 向量、后复用为 α 向量；negBuf 存 min(x, 0) * α
        pipe.InitBuffer(zeroBuf, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(negBuf, this->tileLength * sizeof(DT_X));
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
        LocalTensor<DT_X> zeroLocal = zeroBuf.Get<DT_X>();
        LocalTensor<DT_X> negLocal = negBuf.Get<DT_X>();
        // y = max(x, 0) + α * min(x, 0)
        AscendC::Duplicate(zeroLocal, static_cast<DT_X>(0), this->tileLength);          // zero = 0
        AscendC::Max(yLocal, xLocal, zeroLocal, this->tileLength);                      // y = max(x, 0)
        AscendC::Min(negLocal, xLocal, zeroLocal, this->tileLength);                    // neg = min(x, 0)
        AscendC::Duplicate(zeroLocal, static_cast<DT_X>(this->negativeSlope), this->tileLength); // zero 复用为 α
        AscendC::Mul(negLocal, negLocal, zeroLocal, this->tileLength);                  // neg = α * min(x, 0)
        AscendC::Add(yLocal, yLocal, negLocal, this->tileLength);                       // y = max(x, 0) + α * min(x, 0)
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
    TBuf<TPosition::VECCALC> zeroBuf; // 临时：0 向量 / α 向量
    TBuf<TPosition::VECCALC> negBuf;  // 临时：min(x, 0) * α

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    float negativeSlope;  // 属性：负半轴斜率 α
    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
