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

constexpr int32_t BUFFER_NUM = 1; // 每队列 buffer 块数（>1 时开启双缓冲）

template <typename DT_X>
class Exp {
public:
    __aicore__ inline Exp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
    {
        // 多核切分：本核起始偏移 = 核号 × 单核元素数，尾核按剩余元素截断
        int64_t begin = static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->blockFactor;
        if (begin >= tilingData->totalNum) {
            blockLength_ = 0; // 本核无数据（totalNum 不足摊满所有核）
            return;
        }
        int64_t remain = tilingData->totalNum - begin;
        blockLength_ = remain < tilingData->blockFactor ? remain : tilingData->blockFactor;
        ubFactor_ = tilingData->ubFactor;

        // 设置本核 GM 起始地址
        inputGM_.SetGlobalBuffer((__gm__ DT_X*)x + begin, blockLength_);
        outputGM_.SetGlobalBuffer((__gm__ DT_X*)y + begin, blockLength_);
        // 分配 UB 队列内存
        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, ubFactor_ * sizeof(DT_X));
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, ubFactor_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 核内按 ubFactor_ 分块循环，尾块长度单独处理
        int64_t offset = 0;
        while (offset < blockLength_) {
            int64_t currentNum = blockLength_ - offset;
            if (currentNum > ubFactor_) {
                currentNum = ubFactor_;
            }
            CopyIn(offset, currentNum);
            Compute(currentNum);
            CopyOut(offset, currentNum);
            offset += currentNum;
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum)
    {
        LocalTensor<DT_X> xLocal = inputQueue_.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGM_[offset], currentNum);
        inputQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        LocalTensor<DT_X> xLocal = inputQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueue_.AllocTensor<DT_X>();
        AscendC::Exp(yLocal, xLocal, currentNum); // 逐元素自然指数
        outputQueue_.EnQue(yLocal);
        inputQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum)
    {
        LocalTensor<DT_X> yLocal = outputQueue_.DeQue<DT_X>();
        DataCopy(outputGM_[offset], yLocal, currentNum);
        outputQueue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    GlobalTensor<DT_X> inputGM_;
    GlobalTensor<DT_X> outputGM_;
    int64_t blockLength_ = 0; // 本核实际处理元素数
    int64_t ubFactor_ = 0;    // 每次 UB 循环处理元素数
};

} // namespace NsExp
#endif // EXP_H
