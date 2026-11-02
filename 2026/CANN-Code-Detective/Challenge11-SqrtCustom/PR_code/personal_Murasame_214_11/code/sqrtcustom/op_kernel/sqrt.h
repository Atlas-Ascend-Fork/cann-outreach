/*!
 * \file sqrt.h
 * \brief Sqrt 算子核函数实现：tile 轮转分核 + 单缓冲；fp16 升 fp32 计算
 */

#ifndef SQRT_H
#define SQRT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sqrt_tiling_data.h"

namespace NsSqrt {

using namespace AscendC;

// 单缓冲即可：每个 tile 的处理链路 CopyIn -> Compute -> CopyOut 串行推进，
// 省下的 UB 全部让给 tile 长度，一次搬运覆盖更多元素。
constexpr int32_t BUFFER_NUM = 1;

template <typename T, bool kUpcastToFp32>
class SqrtTile {
public:
    __aicore__ inline SqrtTile() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
    {
        totalNum_ = tilingData->totalNum;
        tileLength_ = tilingData->ubFactor;
        maxTiles_ = tileLength_ <= 0 ? 0 : (tilingData->blockFactor / tileLength_);
        if (totalNum_ <= 0 || tileLength_ <= 0 || maxTiles_ <= 0) {
            return;
        }
        // 整块张量交给各核轮转取用，核内不再做连续段偏移
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum_);
        pipe_.InitBuffer(inQue_, BUFFER_NUM, static_cast<uint32_t>(tileLength_ * sizeof(T)));
        pipe_.InitBuffer(outQue_, BUFFER_NUM, static_cast<uint32_t>(tileLength_ * sizeof(T)));
        if constexpr (kUpcastToFp32) {
            // fp16 输入先升到 fp32 计算，再舍入回 fp16
            pipe_.InitBuffer(calcBuf_, static_cast<uint32_t>(tileLength_ * sizeof(float)));
        }
    }

    __aicore__ inline void Process()
    {
        if (totalNum_ <= 0 || tileLength_ <= 0 || maxTiles_ <= 0) {
            return;
        }
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t tileCount = (totalNum_ + tileLength_ - 1) / tileLength_;
        // 第 i 个核处理编号为 blockIdx + k * blockNum 的 tile，天然负载均衡
        for (int64_t used = 0, tile = blockIdx; used < maxTiles_ && tile < tileCount; ++used, tile += blockNum) {
            const int64_t offset = tile * tileLength_;
            const int64_t remain = totalNum_ - offset;
            const int64_t curLength = remain < tileLength_ ? remain : tileLength_;
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t curLength)
    {
        LocalTensor<T> xLocal = inQue_.AllocTensor<T>();
        if ((curLength * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
            DataCopy(xLocal, xGm_[offset], static_cast<uint32_t>(curLength));
        } else {
            // 仅整网最后一个 tile 会落到这里
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        }
        inQue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t curLength)
    {
        LocalTensor<T> xLocal = inQue_.DeQue<T>();
        LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
        if constexpr (kUpcastToFp32) {
            LocalTensor<float> calcLocal = calcBuf_.Get<float>();
            AscendC::Cast<float, T>(calcLocal, xLocal, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(curLength));
            AscendC::Sqrt(calcLocal, calcLocal, static_cast<int32_t>(curLength));
            AscendC::Cast<T, float>(yLocal, calcLocal, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(curLength));
        } else {
            AscendC::Sqrt(yLocal, xLocal, static_cast<int32_t>(curLength));
        }
        outQue_.EnQue(yLocal);
        inQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t curLength)
    {
        LocalTensor<T> yLocal = outQue_.DeQue<T>();
        if ((curLength * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
            DataCopy(yGm_[offset], yLocal, static_cast<uint32_t>(curLength));
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[offset], yLocal, copyParams);
        }
        outQue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    TBuf<TPosition::VECCALC> calcBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t totalNum_ = 0;
    int64_t tileLength_ = 0;
    int64_t maxTiles_ = 0;
};

} // namespace NsSqrt
#endif // SQRT_H