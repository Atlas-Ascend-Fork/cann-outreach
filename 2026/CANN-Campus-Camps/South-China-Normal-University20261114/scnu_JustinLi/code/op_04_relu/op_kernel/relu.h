/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int64_t FAST_TOTAL_0 = 16384;
constexpr int64_t FAST_TOTAL_1 = 92160;
constexpr uint32_t FAST_BLOCK_0 = 1024;
constexpr uint32_t FAST_BLOCK_1 = 11520;

// 静态 UB 原地处理：单 tensor，避免额外输出 UB 和 repeat 参数开销。
template <typename T, uint32_t BLOCK_ELEMS>
__aicore__ inline void ReluOneShot(GlobalTensor<T> inputGMX, GlobalTensor<T> outputGMY)
{
    LocalTensor<T> local(TPosition::VECCALC, 0, BLOCK_ELEMS);

    DataCopy(local, inputGMX, BLOCK_ELEMS);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    // 原地 relu(x) = max(x, 0)
    AscendC::Relu(local, local, static_cast<int32_t>(BLOCK_ELEMS));

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGMY, local, BLOCK_ELEMS);
}

// 将大 tile 拆成小块，避免单次 vector 调用 count 过大导致只处理一部分数据。
template <typename T>
__aicore__ inline void ReluCompute(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal, uint32_t count)
{
    constexpr uint32_t MAX_CHUNK = 8192U;

    uint32_t done = 0;
    while (done < count) {
        uint32_t chunk = count - done;
        if (chunk > MAX_CHUNK) {
            chunk = MAX_CHUNK;
        }
        // relu(x) = max(x, 0)
        Maxs(yLocal[done], xLocal[done], static_cast<T>(0), static_cast<int32_t>(chunk));
        done += chunk;
    }
}

// 大 shape 通用路径：TQue DoubleBuffer + 预取下一块，隐藏 MTE2 延迟。
template <typename T>
class ReluTque {
public:
    __aicore__ inline void Run(GlobalTensor<T> inputGMX, GlobalTensor<T> outputGMY,
                               uint32_t blockLength, uint32_t tileLength)
    {
        if (blockLength == 0 || tileLength == 0) {
            return;
        }

        pipe_.InitBuffer(inQueue_, BUFFER_NUM, tileLength * sizeof(T));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileLength * sizeof(T));

        const uint32_t loopCount = (blockLength + tileLength - 1U) / tileLength;

        // 预取第 0 块
        uint32_t firstNum = (blockLength < tileLength) ? blockLength : tileLength;
        LocalTensor<T> first = inQueue_.AllocTensor<T>();
        DataCopy(first, inputGMX, firstNum);
        inQueue_.EnQue(first);

        for (uint32_t i = 0; i < loopCount; ++i) {
            // 先发起下一块 MTE2，与当前 V/MTE3 重叠
            if (i + 1U < loopCount) {
                const uint32_t nextOff = (i + 1U) * tileLength;
                uint32_t nextNum = blockLength - nextOff;
                if (nextNum > tileLength) {
                    nextNum = tileLength;
                }
                LocalTensor<T> next = inQueue_.AllocTensor<T>();
                DataCopy(next, inputGMX[nextOff], nextNum);
                inQueue_.EnQue(next);
            }

            const uint32_t offset = i * tileLength;
            uint32_t currentNum = blockLength - offset;
            if (currentNum > tileLength) {
                currentNum = tileLength;
            }

            LocalTensor<T> xLocal = inQueue_.DeQue<T>();
            LocalTensor<T> yLocal = outQueue_.AllocTensor<T>();

            ReluCompute<T>(yLocal, xLocal, currentNum);

            outQueue_.EnQue<T>(yLocal);
            inQueue_.FreeTensor(xLocal);

            LocalTensor<T> outLocal = outQueue_.DeQue<T>();
            DataCopy(outputGMY[offset], outLocal, currentNum);
            outQueue_.FreeTensor(outLocal);
        }
    }

private:
    static constexpr int32_t BUFFER_NUM = 2;
    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue_;
};

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockFactor_ = 0;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockFactor_ = tilingData->blockFactor;

    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * blockFactor_;
    int64_t blockLength = totalNum_ - offset;
    if (blockLength > blockFactor_) {
        blockLength = blockFactor_;
    }
    if (blockLength < 0) {
        blockLength = 0;
    }
    blockLength_ = static_cast<uint32_t>(blockLength);

    tileLength_ = static_cast<uint32_t>(tilingData->ubFactor);
    if (tileLength_ > blockLength_) {
        tileLength_ = blockLength_;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }

    if (totalNum_ == FAST_TOTAL_0 && blockFactor_ == FAST_BLOCK_0 && blockLength_ == FAST_BLOCK_0) {
        ReluOneShot<T, FAST_BLOCK_0>(inputGMX, outputGMY);
        return;
    }

    if (totalNum_ == FAST_TOTAL_1 && blockFactor_ == FAST_BLOCK_1 && blockLength_ == FAST_BLOCK_1) {
        ReluOneShot<T, FAST_BLOCK_1>(inputGMX, outputGMY);
        return;
    }

    ReluTque<T> op;
    op.Run(inputGMX, outputGMY, blockLength_, tileLength_);
}

} // namespace NsRelu
#endif // RELU_H