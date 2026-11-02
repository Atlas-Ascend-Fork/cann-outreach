/*!
 * \file sqrt.h
 * \brief Sqrt 算子核函数实现：连续分段 + 双缓冲流水
 */

#ifndef SQRT_H
#define SQRT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sqrt_tiling_data.h"

namespace NsSqrt {

using namespace AscendC;

// 输入、输出队列各开两块，Compute 与 MTE2/MTE3 重叠执行
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class SqrtVec {
public:
    __aicore__ inline SqrtVec() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
    {
        totalNum_ = tilingData->totalNum;
        tileLen_ = tilingData->ubFactor;
        // 本核在整网中的起点：核号 × 单核元素数
        const int64_t begin = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
        if (begin >= totalNum_ || tileLen_ <= 0) {
            blockLen_ = 0;
            return;
        }
        const int64_t remain = totalNum_ - begin;
        blockLen_ = remain < tilingData->blockFactor ? remain : tilingData->blockFactor;
        // 各核只看到自己的那一段连续数据，偏移计算随后全部在段内进行
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + begin, blockLen_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + begin, blockLen_);
        pipe_.InitBuffer(inQue_, BUFFER_NUM, static_cast<uint32_t>(tileLen_ * sizeof(T)));
        pipe_.InitBuffer(outQue_, BUFFER_NUM, static_cast<uint32_t>(tileLen_ * sizeof(T)));
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
        LocalTensor<T> xLocal = inQue_.AllocTensor<T>();
        // 尾块长度不保证 32B 整数倍，统一用 DataCopyPad 承载任意长度
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t curLen)
    {
        LocalTensor<T> xLocal = inQue_.DeQue<T>();
        LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
        AscendC::Sqrt(yLocal, xLocal, static_cast<int32_t>(curLen));
        outQue_.EnQue(yLocal);
        inQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t curLen)
    {
        LocalTensor<T> yLocal = outQue_.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t totalNum_ = 0;
    int64_t blockLen_ = 0; // 本核实际处理的元素数
    int64_t tileLen_ = 0;  // 单次搬入 UB 的元素数
};

} // namespace NsSqrt
#endif // SQRT_H