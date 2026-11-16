/*!
 * \file relu.h
 * \brief Relu 算子 kernel：y = max(x, 0)
 *
 * 结构上走最朴素的一条路：
 *   - 每核拿一段连续数据，核内再按 ubFactor 切块；
 *   - 输入/输出各一个 TQue，深度 1，不做预取（单缓冲，简单、好对拍）；
 *   - 每块只调 Maxs（和标量 0 取大），count 超过单条指令的 repeat 上限时按段续算；
 *   - 搬运统一走 DataCopyPad，尾块不对齐也不用单独判分支。
 */

#ifndef RELU_KERNEL_H
#define RELU_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace ReluAct {

using namespace AscendC;

// 单缓冲：一个输入队列槽位 + 一个输出队列槽位
constexpr int32_t kQueueDepth = 1;
// 一次 repeat 处理 256B；repeatTimes 是 uint8_t，所以单条指令最多 255 次
constexpr int64_t kBytesPerRepeat = 256;
constexpr int64_t kRepeatTimesMax = 255;

template <typename T>
class ReluKernel {
public:
    __aicore__ inline ReluKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData *cfg);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count);
    __aicore__ inline void Compute(int64_t count);
    __aicore__ inline void CopyOut(int64_t offset, int64_t count);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, kQueueDepth> inQueue_;
    TQue<QuePosition::VECOUT, kQueueDepth> outQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int64_t coreLength_ = 0;  // 本核元素数
    int64_t tileLength_ = 1;  // 单块元素数
};

template <typename T>
__aicore__ inline void ReluKernel<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData *cfg)
{
    const int64_t begin = cfg->blockFactor * static_cast<int64_t>(GetBlockIdx());
    const int64_t rest = cfg->totalNum - begin;

    // 尾核不足一块，按剩余量裁；越界的核直接当空处理
    coreLength_ = (rest < cfg->blockFactor) ? rest : cfg->blockFactor;
    if (coreLength_ < 0) {
        coreLength_ = 0;
    }
    // 单块字节数必须是 32B 的倍数，否则 InitBuffer 拿不到合法容量；
    // 核上数据比一个对齐单元还少时，也按一个对齐单元申请
    const int64_t minTile = static_cast<int64_t>(32) / static_cast<int64_t>(sizeof(T));
    tileLength_ = (cfg->ubFactor > 0) ? cfg->ubFactor : coreLength_;
    if (tileLength_ < minTile) {
        tileLength_ = minTile;
    }

    xGm_.SetGlobalBuffer((__gm__ T *)x + begin, static_cast<uint32_t>(coreLength_));
    yGm_.SetGlobalBuffer((__gm__ T *)y + begin, static_cast<uint32_t>(coreLength_));

    pipe_.InitBuffer(inQueue_, kQueueDepth, static_cast<uint32_t>(tileLength_ * static_cast<int64_t>(sizeof(T))));
    pipe_.InitBuffer(outQueue_, kQueueDepth, static_cast<uint32_t>(tileLength_ * static_cast<int64_t>(sizeof(T))));
}

template <typename T>
__aicore__ inline void ReluKernel<T>::CopyIn(int64_t offset, int64_t count)
{
    LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
    inQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void ReluKernel<T>::Compute(int64_t count)
{
    LocalTensor<T> xLocal = inQueue_.DeQue<T>();
    LocalTensor<T> yLocal = outQueue_.AllocTensor<T>();

    // 一条 Maxs 能算的元素数有上限，超了就切段接着算
    constexpr int64_t elemsPerRepeat = kBytesPerRepeat / static_cast<int64_t>(sizeof(T));
    constexpr int64_t chunkElems = kRepeatTimesMax * elemsPerRepeat;

    int64_t done = 0;
    while (done < count) {
        const int64_t left = count - done;
        const int32_t cur = static_cast<int32_t>((left < chunkElems) ? left : chunkElems);
        Maxs(yLocal[static_cast<int32_t>(done)], xLocal[static_cast<int32_t>(done)], static_cast<T>(0), cur);
        done += cur;
    }

    outQueue_.EnQue<T>(yLocal);
    inQueue_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void ReluKernel<T>::CopyOut(int64_t offset, int64_t count)
{
    LocalTensor<T> yLocal = outQueue_.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(yGm_[offset], yLocal, copyParams);
    outQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void ReluKernel<T>::Process()
{
    if (coreLength_ <= 0) {
        return;
    }
    // 用 while 推进而不是先算 loopCount：尾块大小天然被 count 约束，少一个分支
    int64_t offset = 0;
    while (offset < coreLength_) {
        const int64_t left = coreLength_ - offset;
        const int64_t count = (left < tileLength_) ? left : tileLength_;
        CopyIn(offset, count);
        Compute(count);
        CopyOut(offset, count);
        offset += count;
    }
}

}  // namespace ReluAct

#endif  // RELU_KERNEL_H
