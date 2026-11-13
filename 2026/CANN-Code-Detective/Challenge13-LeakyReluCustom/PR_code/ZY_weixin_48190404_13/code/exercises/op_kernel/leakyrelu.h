/*!
 * \file leakyrelu.h
 * \brief Leakyrelu 算子 kernel 类定义
 */

#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "leakyrelu_tiling_data.h"
#include "leakyrelu_tiling_key.h"

namespace NsLeakyrelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t BLOCK_BYTES = 32; // GM/UB 搬运的对齐粒度

// 用 sizeof 判断 dtype（T 只会是 half=2 字节或 float=4 字节），
// 不依赖 type_traits，也不依赖 half 的具体类型定义
constexpr int64_t HALF_TYPE_SIZE = 2;

template <typename T>
class Leakyrelu {
public:
    __aicore__ inline Leakyrelu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

    // y = max(x, 0) + negative_slope * min(x, 0)，在 float 域上完成
    __aicore__ inline void ComputeFloat(const LocalTensor<float>& src, const LocalTensor<float>& dst,
                                        int64_t count);
    __aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
    {
        return (value + align - 1) / align * align;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> floatBufX; // 仅 half 分支使用
    TBuf<TPosition::VECCALC> floatBufY; // 仅 half 分支使用

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t alignElems_ = 1;
    float negativeSlope_ = LEAKYRELU_DEFAULT_SLOPE;
};

template <typename T>
__aicore__ inline void Leakyrelu<T>::Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
{
    alignElems_ = BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    // 兼容只填写了前三个字段的 CPU kernel UT fixture（未下发属性时使用默认 slope）
    negativeSlope_ = (tilingData->slopeTag == LEAKYRELU_SLOPE_TAG) ? tilingData->negativeSlope
                                                                   : LEAKYRELU_DEFAULT_SLOPE;
    blockLength_ = 0;
    ubLength_ = (tilingData->ubFactor > 0) ? tilingData->ubFactor : tilingData->blockFactor;
    if (ubLength_ <= 0) {
        ubLength_ = alignElems_;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t offset = blockIdx * tilingData->blockFactor;
    if (offset >= tilingData->totalNum) {
        return; // 多余的核空跑
    }
    const int64_t remain = tilingData->totalNum - offset;
    blockLength_ = (remain < tilingData->blockFactor) ? remain : tilingData->blockFactor;
    if (blockLength_ <= 0) {
        return;
    }
    if (ubLength_ > blockLength_) {
        ubLength_ = AlignUp(blockLength_, alignElems_);
    }
    if (ubLength_ < alignElems_) {
        ubLength_ = alignElems_;
    }
    ubLength_ = AlignUp(ubLength_, alignElems_);

    // 每个核只看自己那一段，起始地址天然 32B 对齐（blockFactor 已对齐）；
    // 声明长度按对齐长度取，兼容尾块按 copyNum 对齐搬入
    const int64_t gmLength = AlignUp(blockLength_, alignElems_);
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, gmLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, gmLength);

    // 输入/输出双缓冲；尾块按对齐长度计算时仍不会超过该容量
    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
    if constexpr (static_cast<int64_t>(sizeof(T)) == HALF_TYPE_SIZE) {
        // half 直接乘 slope 会引入半精度舍入误差，这里升到 float 计算
        const uint32_t floatBytes = static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(float)));
        pipe.InitBuffer(floatBufX, floatBytes);
        pipe.InitBuffer(floatBufY, floatBytes);
    }
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    // 按 32B 对齐长度搬入；尾部多读的元素只参与计算，不会被写回 GM
    const int64_t copyNum = AlignUp(currentNum, alignElems_);
    DataCopy(xLocal, inputGMX[offset], static_cast<int32_t>(copyNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::ComputeFloat(const LocalTensor<float>& src,
                                                  const LocalTensor<float>& dst, int64_t count)
{
    const int32_t n = static_cast<int32_t>(count);
    // dst = max(x, 0)，即正半轴
    Relu(dst, src, n);
    // src = x - max(x, 0) = min(x, 0)（原地复用输入缓冲）
    Sub(src, src, dst, n);
    // src = negative_slope * min(x, 0)
    Muls(src, src, negativeSlope_, n);
    // dst = max(x, 0) + negative_slope * min(x, 0)
    Add(dst, dst, src, n);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Compute(int64_t currentNum)
{
    const int64_t count = AlignUp(currentNum, alignElems_);
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    if constexpr (static_cast<int64_t>(sizeof(T)) == HALF_TYPE_SIZE) {
        LocalTensor<float> xFloat = floatBufX.Get<float>();
        LocalTensor<float> yFloat = floatBufY.Get<float>();
        Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(count));
        ComputeFloat(xFloat, yFloat, count);
        Cast(yLocal, yFloat, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(count));
    } else {
        // T == float：直接复用输入/输出缓冲，无需额外 UB 空间
        ComputeFloat(xLocal, yLocal, count);
    }

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    // 只写回有效元素，避免越界污染输出张量尾部
    DataCopy(outputGMY[offset], yLocal, static_cast<int32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t progress = 0; progress < loopCount; ++progress) {
        const int64_t remain = blockLength_ - progress * ubLength_;
        const int64_t currentNum = (remain < ubLength_) ? remain : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
