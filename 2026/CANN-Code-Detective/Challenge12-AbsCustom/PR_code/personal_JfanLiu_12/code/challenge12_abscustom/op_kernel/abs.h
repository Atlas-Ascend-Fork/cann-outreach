/*!
 * \file abs.h
 * \brief Abs 算子核函数实现：多核连续分段 + 核内双缓冲流水
 */

#ifndef ABS_H
#define ABS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "abs_tiling_data.h"
#include "abs_tiling_key.h"

namespace NsAbs {

using namespace AscendC;

// 输入、输出队列各开两块，让 MTE2 / VEC / MTE3 三级流水重叠
constexpr int32_t BUFFER_NUM = 2;

template <typename DT_X>
class AbsVec {
public:
    __aicore__ inline AbsVec() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AbsTilingData* tilingData)
    {
        totalNum_ = tilingData->totalNum;
        blockFactor_ = tilingData->blockFactor;
        tileNum_ = tilingData->tileNum;
        // 本核在整网中的起点：核号 x 单核元素数，各核只看到自己那一段
        const int64_t begin = static_cast<int64_t>(AscendC::GetBlockIdx()) * blockFactor_;
        if (begin >= totalNum_ || tileNum_ <= 0) {
            blockLen_ = 0;
            return;
        }
        const int64_t remain = totalNum_ - begin;
        blockLen_ = remain < blockFactor_ ? remain : blockFactor_;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + begin, blockLen_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + begin, blockLen_);
        // 单块长度按 32B 向上对齐，保证 DataCopyPad 的 UB 容量足够且搬运高效
        const int64_t alignNum = 32 / static_cast<int64_t>(sizeof(DT_X));
        tileLen_ = ((blockLen_ + tileNum_ - 1) / tileNum_ + alignNum - 1) / alignNum * alignNum;
        const uint32_t tileBytes = static_cast<uint32_t>(tileLen_ * sizeof(DT_X));
        pipe_.InitBuffer(inQue_, BUFFER_NUM, tileBytes);
        pipe_.InitBuffer(outQue_, BUFFER_NUM, tileBytes);
    }

    __aicore__ inline void Process()
    {
        if (blockLen_ <= 0 || tileLen_ <= 0) {
            return;
        }
        for (int64_t offset = 0; offset < blockLen_; offset += tileLen_) {
            const int64_t remain = blockLen_ - offset;
            const int64_t curLen = remain < tileLen_ ? remain : tileLen_;
            CopyIn(offset, curLen);
            Compute(curLen);
            CopyOut(offset, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t curLen)
    {
        LocalTensor<DT_X> xLocal = inQue_.AllocTensor<DT_X>();
        // 尾块长度不保证 32B 整数倍，统一用 DataCopyPad 承载任意字节数
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t curLen)
    {
        LocalTensor<DT_X> xLocal = inQue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQue_.AllocTensor<DT_X>();
        AscendC::Abs(yLocal, xLocal, static_cast<int32_t>(curLen));
        outQue_.EnQue(yLocal);
        inQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t curLen)
    {
        LocalTensor<DT_X> yLocal = outQue_.DeQue<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    int64_t totalNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tileNum_ = 0;
    int64_t blockLen_ = 0; // 本核实际处理的元素数
    int64_t tileLen_ = 0;  // 单次搬入 UB 的元素数
};

} // namespace NsAbs
#endif // ABS_H
