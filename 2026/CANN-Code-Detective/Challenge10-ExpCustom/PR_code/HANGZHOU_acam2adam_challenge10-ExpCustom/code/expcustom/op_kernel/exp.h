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

constexpr int32_t BUFFER_NUM = 2; // 每队列 buffer 块数（双缓冲）

template <typename DT_X>
class Exp {
public:
    __aicore__ inline Exp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
    {
        xGm_.SetGlobalBuffer((__gm__ DT_X*)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X*)y);

        blockNum_ = tilingData->blockNum;
        blockFactor_ = tilingData->blockFactor;
        ubFactor_ = tilingData->ubFactor;
        ubLoopOfFormerBlock_ = tilingData->ubLoopOfFormerBlock;
        ubTailOfFormerBlock_ = tilingData->ubTailOfFormerBlock;
        ubLoopOfTailBlock_ = tilingData->ubLoopOfTailBlock;
        ubTailOfTailBlock_ = tilingData->ubTailOfTailBlock;

        uint32_t tileBytes =
            static_cast<uint32_t>(ubFactor_ * static_cast<int32_t>(sizeof(DT_X)));
        // 防御双保险：空输入时 host 下发 block dim=1、9 个 tiling 字段全 0（blockNum=0、ubFactor=0），
        // 此处跳过 InitBuffer(0)（tikicpulib 对 0 字节 buffer abort）；Process 入口由 blockNum_<=0
        // 早退，不读写 GM（blockIdx 越界防御同样先行 return）。正常路径 ubFactor>0 恒成立，行为不变。
        if (tileBytes > 0) {
            pipe_.InitBuffer(inQueue_, BUFFER_NUM, tileBytes);
            pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileBytes);
        }
    }

    __aicore__ inline void Process()
    {
        const int32_t blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
        // 越界/非法 tiling 防御（先于任何 GM 寻址）：
        //   - blockNum_ <= 0：空输入（totalNum=0）产物或非法 tiling；
        //   - blockIdx < 0 / blockIdx >= blockNum_：实际启动块数大于 blockNum 时，
        //     超出块会按 former/tail 因子计算并越界读写 GM，必须直接返回。
        if (blockNum_ <= 0 || blockIdx < 0 || blockIdx >= blockNum_) {
            return;
        }
        const bool isLast = (blockIdx == blockNum_ - 1);
        const int32_t loopNum = isLast ? ubLoopOfTailBlock_ : ubLoopOfFormerBlock_;
        const int32_t tailNum = isLast ? ubTailOfTailBlock_ : ubTailOfFormerBlock_;
        if (loopNum <= 0 || tailNum <= 0) {
            return; // 防御：非法 tiling（loop/tail 非正）
        }
        // host 侧已保证 totalNum / blockFactor / blockNum ∈ (0, INT32_MAX]（越界 fail-fast），
        // 且 blockIdx < blockNum_ ⇒ blockIdx × blockFactor_ ≤ (blockNum_ - 1) × blockFactor_
        // < totalNum ≤ INT32_MAX；tile 累加不超过 blockFactor_。故 offset 及 offset+count
        // 全程处于 int32 安全域，不会溢出。
        int32_t offset = blockIdx * blockFactor_;
        // 对齐判断外提：完整 tile（ubFactor 元素，256B 对齐）恒走 DataCopy 快速路径；
        // 仅尾核最后一块可能非 32B 对齐，每块只判断一次，不在 tile 循环内重复取模。
        bool tailUnaligned = isLast && ((tailNum * static_cast<int32_t>(sizeof(DT_X))) % 32 != 0);

        for (int32_t i = 0; i < loopNum - 1; i++) {
            ProcessTileAligned(offset, ubFactor_);
            offset += ubFactor_;
        }
        // 尾 tile（含残差）：按对齐决定快速路径 / DataCopyPad 精确路径
        if (tailUnaligned) {
            ProcessTilePad(offset, tailNum);
        } else {
            ProcessTileAligned(offset, tailNum);
        }
    }

private:
    __aicore__ inline void ProcessTileAligned(int32_t offset, int32_t count)
    {
        // 32B 对齐路径：DataCopy 连续搬运 + Exp + DataCopy 搬出（双队列双缓冲流水）
        LocalTensor<DT_X> srcLocal = inQueue_.AllocTensor<DT_X>();
        DataCopy(srcLocal, xGm_[offset], static_cast<uint32_t>(count));   // MTE2: GM→UB
        inQueue_.EnQue(srcLocal);
        LocalTensor<DT_X> dstLocal = outQueue_.AllocTensor<DT_X>();
        srcLocal = inQueue_.DeQue<DT_X>();
        AscendC::Exp(dstLocal, srcLocal, count);                          // V: 逐元素指数
        outQueue_.EnQue(dstLocal);
        inQueue_.FreeTensor(srcLocal);
        dstLocal = outQueue_.DeQue<DT_X>();
        DataCopy(yGm_[offset], dstLocal, static_cast<uint32_t>(count));   // MTE3: UB→GM
        outQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ProcessTilePad(int32_t offset, int32_t count)
    {
        // 非 32B 对齐残差路径：DataCopyPad 精确搬运
        LocalTensor<DT_X> srcLocal = inQueue_.AllocTensor<DT_X>();
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * static_cast<int32_t>(sizeof(DT_X))), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(srcLocal, xGm_[offset], copyParams, padParams);
        inQueue_.EnQue(srcLocal);
        LocalTensor<DT_X> dstLocal = outQueue_.AllocTensor<DT_X>();
        srcLocal = inQueue_.DeQue<DT_X>();
        AscendC::Exp(dstLocal, srcLocal, count);
        outQueue_.EnQue(dstLocal);
        inQueue_.FreeTensor(srcLocal);
        dstLocal = outQueue_.DeQue<DT_X>();
        DataCopyPad(yGm_[offset], dstLocal, copyParams);
        outQueue_.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue_;
    int32_t blockNum_;
    int32_t blockFactor_;
    int32_t ubFactor_;
    int32_t ubLoopOfFormerBlock_;
    int32_t ubTailOfFormerBlock_;
    int32_t ubLoopOfTailBlock_;
    int32_t ubTailOfTailBlock_;
};

} // namespace NsExp
#endif // EXP_H
