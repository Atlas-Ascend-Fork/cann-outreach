/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel：单输入单输出，逐元素
 */

#ifndef TANH_KERNEL_H
#define TANH_KERNEL_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace TanhVec {

using namespace AscendC;

constexpr int32_t kQueueDepth = 2;
constexpr int64_t kAlignBytes = 32;

template <typename T>
class TanhKernel {
public:
    __aicore__ inline TanhKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t TileCount(int64_t tile) const;
    __aicore__ inline bool CanUsePlainCopy(int64_t count) const;
    __aicore__ inline void LoadTile(int64_t tile, int64_t count);
    __aicore__ inline void CalcTile(int64_t count);
    __aicore__ inline void StoreTile(int64_t tile, int64_t count);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, kQueueDepth> inQueue_;
    TQue<QuePosition::VECOUT, kQueueDepth> outQueue_;
    TBuf<TPosition::VECCALC> calcBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int64_t coreLength_ = 0;
    int64_t tileLength_ = 1;
    int64_t loopCount_ = 0;
};

template <typename T>
__aicore__ inline void TanhKernel<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData *tilingData)
{
    const int64_t begin = tilingData->blockFactor * static_cast<int64_t>(GetBlockIdx());
    const int64_t rest = tilingData->totalNum - begin;
    coreLength_ = (rest < tilingData->blockFactor) ? rest : tilingData->blockFactor;
    if (coreLength_ < 0) {
        coreLength_ = 0;
    }

    const int64_t minTile = static_cast<int64_t>(kAlignBytes) / static_cast<int64_t>(sizeof(T));
    tileLength_ = (tilingData->ubFactor > 0) ? tilingData->ubFactor : coreLength_;
    if (tileLength_ < minTile) {
        tileLength_ = minTile;
    }
    constexpr int64_t kUbBudgetBytes = 128 * 1024;
    const int64_t perElemBytes = 4 * static_cast<int64_t>(sizeof(T)) + 3 * static_cast<int64_t>(sizeof(float));
    const int64_t tileCap = ((kUbBudgetBytes / perElemBytes) / minTile) * minTile;
    if (tileCap >= minTile && tileLength_ > tileCap) {
        tileLength_ = tileCap;
    }
    loopCount_ = (coreLength_ > 0) ? ((coreLength_ + tileLength_ - 1) / tileLength_) : 0;

    xGm_.SetGlobalBuffer((__gm__ T *)x + begin, static_cast<uint32_t>(coreLength_));
    yGm_.SetGlobalBuffer((__gm__ T *)y + begin, static_cast<uint32_t>(coreLength_));

    pipe_.InitBuffer(inQueue_, kQueueDepth, static_cast<uint32_t>(tileLength_ * static_cast<int64_t>(sizeof(T))));
    pipe_.InitBuffer(outQueue_, kQueueDepth, static_cast<uint32_t>(tileLength_ * static_cast<int64_t>(sizeof(T))));
    pipe_.InitBuffer(calcBuf_, static_cast<uint32_t>(3 * tileLength_ * static_cast<int64_t>(sizeof(float))));
}

template <typename T>
__aicore__ inline int64_t TanhKernel<T>::TileCount(int64_t tile) const
{
    const int64_t left = coreLength_ - tile * tileLength_;
    return (left < tileLength_) ? left : tileLength_;
}

template <typename T>
__aicore__ inline bool TanhKernel<T>::CanUsePlainCopy(int64_t count) const
{
    return (count * static_cast<int64_t>(sizeof(T)) % kAlignBytes) == 0;
}

template <typename T>
__aicore__ inline void TanhKernel<T>::LoadTile(int64_t tile, int64_t count)
{
    LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
    const int64_t offset = tile * tileLength_;

    if (CanUsePlainCopy(count)) {
        DataCopy(xLocal, xGm_[offset], static_cast<uint32_t>(count));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
    }
    inQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void TanhKernel<T>::CalcTile(int64_t count)
{
    LocalTensor<T> xLocal = inQueue_.DeQue<T>();
    LocalTensor<T> yLocal = outQueue_.AllocTensor<T>();

    const int32_t n = static_cast<int32_t>(count);
    const float two = 2.0f;
    const float one = 1.0f;

    LocalTensor<float> expBuf = calcBuf_.Get<float>();
    LocalTensor<float> numBuf = expBuf[static_cast<int32_t>(tileLength_)];
    LocalTensor<float> denBuf = expBuf[static_cast<int32_t>(2 * tileLength_)];

    if constexpr (std::is_same<T, float>::value) {
        Muls(expBuf, xLocal, -two, n);
    } else {
        Cast(expBuf, xLocal, RoundMode::CAST_NONE, n);
        Muls(expBuf, expBuf, -two, n);
    }
    Exp(expBuf, expBuf, n);

    Adds(denBuf, expBuf, one, n);
    Muls(numBuf, expBuf, -one, n);
    Adds(numBuf, numBuf, one, n);
    Div(expBuf, numBuf, denBuf, n);

    if constexpr (std::is_same<T, float>::value) {
        Muls(yLocal, expBuf, one, n);
    } else {
        Cast(yLocal, expBuf, RoundMode::CAST_NONE, n);
    }

    outQueue_.EnQue<T>(yLocal);
    inQueue_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void TanhKernel<T>::StoreTile(int64_t tile, int64_t count)
{
    LocalTensor<T> yLocal = outQueue_.DeQue<T>();
    const int64_t offset = tile * tileLength_;

    if (CanUsePlainCopy(count)) {
        DataCopy(yGm_[offset], yLocal, static_cast<uint32_t>(count));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(yGm_[offset], yLocal, copyParams);
    }
    outQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void TanhKernel<T>::Process()
{
    for (int64_t tile = 0; tile < loopCount_; ++tile) {
        const int64_t count = TileCount(tile);
        LoadTile(tile, count);
        CalcTile(count);
        StoreTile(tile, count);
    }
}

}  // namespace TanhVec

#endif  // TANH_KERNEL_H