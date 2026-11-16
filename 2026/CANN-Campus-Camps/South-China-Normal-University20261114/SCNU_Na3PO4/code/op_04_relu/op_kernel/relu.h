/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义（优化版）
 *
 * 优化点：
 *   1. 单指令实现：Maxs(dst, src, 0) 一条矢量指令完成 ReLU，无多余算子；
 *   2. 双缓冲流水：输入/输出队列各 2 块 buffer，MTE2 / V / MTE3 重叠；
 *   3. 搬运快慢路径：32B 对齐时走 DataCopy（无 Pad 开销），仅尾块走 DataCopyPad；
 *   4. UB 用量最小：只有输入、输出两组 buffer，ubFactor 可开到 UB 上限；
 *   5. 多核均衡 + 512B 对齐切分（由 tiling 保证），DMA 起始地址大粒度对齐。
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t DMA_ALIGN_BYTES = 32; // DMA 最小对齐粒度

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeSingleTile(int64_t currentNum);
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

    __aicore__ inline bool IsAligned(int64_t num) const
    {
        return (num * static_cast<int64_t>(sizeof(T))) % DMA_ALIGN_BYTES == 0;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> singleTileBuf; // 单 tile 专用：绕开 TQue 记账

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 每个核负责的元素区间（尾核可能不足 blockFactor）
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t baseOffset = tilingData->blockFactor * blockIdx;
    int64_t remainderLength = tilingData->totalNum - baseOffset;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + baseOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + baseOffset, blockLength_);

    if (blockLength_ <= ubLength_) {
        // 单 tile：只需一块 UB，走最短路径
        pipe.InitBuffer(singleTileBuf, ubLength_ * sizeof(T));
    } else {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void Relu<T>::ComputeSingleTile(int64_t currentNum)
{
    const uint32_t len = static_cast<uint32_t>(currentNum);
    LocalTensor<T> x = singleTileBuf.Get<T>();

    // GM -> UB
    if (IsAligned(currentNum)) {
        DataCopy(x, inputGMX[0], len);
    } else {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(x, inputGMX[0], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    Maxs(x, x, static_cast<T>(0), len); // 原地 ReLU

    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);

    // UB -> GM
    if (IsAligned(currentNum)) {
        DataCopy(outputGMY[0], x, len);
    } else {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPad(outputGMY[0], x, copyParams);
    }
    // 退出前确保 MTE3 完成
    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();

    if (IsAligned(currentNum)) {
        // 快路径：32B 对齐，直接用 DataCopy，避免 DataCopyPad 的额外开销
        DataCopy(xLocal, inputGMX[gmOffset], static_cast<uint32_t>(currentNum));
    } else {
        // 慢路径：仅尾块使用 DataCopyPad
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[gmOffset], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    uint32_t len = static_cast<uint32_t>(currentNum);
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    // ReLU: y = max(x, 0)，一条矢量指令完成
    Maxs(yLocal, xLocal, static_cast<T>(0), len);

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();

    if (IsAligned(currentNum)) {
        DataCopy(outputGMY[gmOffset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPad(outputGMY[gmOffset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    if (blockLength_ <= ubLength_) {
        ComputeSingleTile(blockLength_);
        return;
    }
    // 用减法推进，避免 int64 除法（AI Core 上 64 位除法是软件实现）
    int64_t offset = 0;
    while (offset < blockLength_) {
        int64_t currentNum = blockLength_ - offset;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
        offset += ubLength_;
    }
}

} // namespace NsRelu
#endif // RELU_H
