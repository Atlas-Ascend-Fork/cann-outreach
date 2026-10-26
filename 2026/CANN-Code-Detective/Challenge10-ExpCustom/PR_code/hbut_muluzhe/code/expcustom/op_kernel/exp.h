/*!
 * \file exp.h
 * \brief Exp 算子 kernel 类定义
 */

#ifndef EXP_H
#define EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {

using namespace AscendC;

constexpr int64_t UB_ALIGN_BYTES = 32; // DataCopy 32B 对齐粒度

template <typename DT_X>
class Exp {
public:
    __aicore__ inline Exp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
    {
        // 多核切分：本核起始偏移与处理长度（尾核不足时截断）
        int64_t startOffset = tilingData->blockFactor * AscendC::GetBlockIdx();
        this->blockLength = tilingData->blockFactor;
        int64_t remainNum = tilingData->totalNum - startOffset;
        if (remainNum < this->blockLength) {
            this->blockLength = (remainNum > 0) ? remainNum : 0;
        }

        // 单次 UB 循环处理长度（tiling 已保证 32B 对齐且不超过 UB 容量）
        this->tileLength = tilingData->ubFactor;
        if (this->tileLength <= 0 || this->tileLength > this->blockLength) {
            this->tileLength = (this->blockLength > 0) ? this->blockLength : 1;
        }

        // UB 切分偏移单独按 32B 向上对齐：尾核 blockLength 可能不是对齐值，
        // 但 yLocal 的起始地址必须 32B 对齐（Vector / DataCopy 的要求）
        constexpr int64_t alignElems = UB_ALIGN_BYTES / static_cast<int64_t>(sizeof(DT_X));
        this->bufOffset = (this->tileLength + alignElems - 1) / alignElems * alignElems;

        // 设置本核在 Global Memory 上的起始地址
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + startOffset, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + startOffset, this->blockLength);

        // 输入/输出共用一块 UB 空间，减少 InitBuffer 次数
        pipe.InitBuffer(ioBuf, 2 * this->bufOffset * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength <= 0) {
            return;
        }
        LocalTensor<DT_X> xLocal = ioBuf.Get<DT_X>();
        LocalTensor<DT_X> yLocal = xLocal[this->bufOffset];

        for (int64_t i = 0; i < this->blockLength; i += this->tileLength) {
            int64_t currentNum = this->blockLength - i;
            if (currentNum > this->tileLength) {
                currentNum = this->tileLength;
            }
            // 上一轮输出已搬完，才能复用 UB
            if (i > 0) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            CopyIn(i, xLocal, currentNum);
            Compute(xLocal, yLocal, currentNum);
            CopyOut(i, yLocal, currentNum);
            if (i + this->tileLength < this->blockLength) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

private:
    // 判断当前块长度是否满足 DataCopy 的 32B 对齐要求
    __aicore__ inline bool IsAligned(int64_t currentNum)
    {
        return (currentNum * static_cast<int64_t>(sizeof(DT_X))) % UB_ALIGN_BYTES == 0;
    }

    __aicore__ inline void CopyIn(int64_t progress, const LocalTensor<DT_X>& xLocal, int64_t currentNum)
    {
        if (IsAligned(currentNum)) {
            DataCopy(xLocal, inputGMX[progress], currentNum);
        } else {
            // 尾块长度不足 32B 对齐，使用 DataCopyPad 按字节搬移
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
            DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void Compute(const LocalTensor<DT_X>& xLocal, const LocalTensor<DT_X>& yLocal,
                                   int64_t currentNum)
    {
        // y = e^x
        AscendC::Exp(yLocal, xLocal, static_cast<int32_t>(currentNum));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    }

    __aicore__ inline void CopyOut(int64_t progress, const LocalTensor<DT_X>& yLocal, int64_t currentNum)
    {
        if (IsAligned(currentNum)) {
            DataCopy(outputGMY[progress], yLocal, currentNum);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(outputGMY[progress], yLocal, copyParams);
        }
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> ioBuf;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    int64_t blockLength = 0; // 本核处理的元素数
    int64_t tileLength = 0;  // 单次 UB 循环处理的元素数
    int64_t bufOffset = 0;   // 输出缓冲区在 UB 中的起始偏移（32B 对齐）
};

} // namespace NsExp
#endif // EXP_H
