#ifndef EXP_H
#define EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1; // 每队列 buffer 块数（单缓冲）

template <typename DT_X>
class Exp {
public:
    __aicore__ inline Exp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
    {
        this->tileLength_ = tilingData->tileLength;
        this->tileNum_ = tilingData->tileNum;
        this->blockLength_ = tilingData->blockLength;

        // 计算当前核的起始偏移
        int64_t blockIdx = GetBlockIdx();
        int64_t startOffset = blockIdx * this->blockLength_;

        // 设置 Global Memory 绑定
        this->inputGM_.SetGlobalBuffer((__gm__ DT_X*)x + startOffset, this->blockLength_);
        this->outputGM_.SetGlobalBuffer((__gm__ DT_X*)y + startOffset, this->blockLength_);

        // 初始化 UB 队列
        if (this->blockLength_ > 0) {
            this->pipe_.InitBuffer(this->inputQueue_, BUFFER_NUM, this->tileLength_ * sizeof(DT_X));
            this->pipe_.InitBuffer(this->outputQueue_, BUFFER_NUM, this->tileLength_ * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process()
    {
        // 循环次数 = tileNum × BUFFER_NUM（tileLength 已按 BUFFER_NUM 折半）
        int32_t loopCount = this->tileNum_ * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 【修改点】加上 template 关键字消除模板依赖歧义
        LocalTensor<DT_X> xLocal = this->inputQueue_.template AllocTensor<DT_X>();
        DataCopy(xLocal, this->inputGM_[progress * this->tileLength_], this->tileLength_);
        this->inputQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        // 【修改点】加上 template 关键字消除模板依赖歧义
        LocalTensor<DT_X> xLocal = this->inputQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = this->outputQueue_.template AllocTensor<DT_X>();

        // 注意：必须加 AscendC:: 前缀，否则会和类名 Exp 冲突
        AscendC::Exp(yLocal, xLocal, this->tileLength_);

        this->outputQueue_.EnQue(yLocal);
        this->inputQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 【修改点】加上 template 关键字消除模板依赖歧义
        LocalTensor<DT_X> yLocal = this->outputQueue_.template DeQue<DT_X>();
        DataCopy(this->outputGM_[progress * this->tileLength_], yLocal, this->tileLength_);
        this->outputQueue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    GlobalTensor<DT_X> inputGM_;
    GlobalTensor<DT_X> outputGM_;

    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
    int64_t tileNum_ = 0;
};

} // namespace NsExp
#endif // EXP_H