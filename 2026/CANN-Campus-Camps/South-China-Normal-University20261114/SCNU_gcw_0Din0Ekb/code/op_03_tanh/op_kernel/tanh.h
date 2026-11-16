/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t UB_BLOCK_BYTES = 32; // UB 最小访问粒度（字节），buffer 按此对齐分配

__aicore__ inline int64_t TanhCeilDiv(int64_t value, int64_t factor)
{
    return (value + factor - 1) / factor;
}

__aicore__ inline int64_t TanhAlignUp(int64_t value, int64_t align)
{
    return TanhCeilDiv(value, align) * align;
}

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // fp16/bf16 输入转为 fp32 计算时使用的临时缓存（x、y 各一份）
    TBuf<TPosition::VECCALC> tempBufXFp32;
    TBuf<TPosition::VECCALC> tempBufYFp32;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t startOffset = blockIdx * tilingData->blockFactor;

    // 尾核实际数据量可能不足 blockFactor，按 totalNum 截断，避免读出/写出越界
    blockLength_ = tilingData->totalNum - startOffset;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    if (blockLength_ > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    }

    // 单次 UB 处理量不超过本核的数据量
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }
    if (ubLength_ <= 0) {
        ubLength_ = 1; // 空 tensor 时仅用于避免除零，Process 会直接返回
    }

    // 每个核绑定到自己负责的那段 GM 数据
    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    // UB 按 32B 对齐分配（末尾不足一个块的部分留白，不参与搬入搬出）
    int64_t alignElem = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    int64_t alignedUbLength = TanhAlignUp(ubLength_, alignElem);
    uint32_t queueBytes = static_cast<uint32_t>(alignedUbLength * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, queueBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, queueBytes);
    if constexpr (sizeof(T) == sizeof(half)) {
        // fp16 计算精度不足，改用 fp32 做中间计算
        uint32_t tempBytes = static_cast<uint32_t>(alignedUbLength * static_cast<int64_t>(sizeof(float)));
        pipe.InitBuffer(tempBufXFp32, tempBytes);
        pipe.InitBuffer(tempBufYFp32, tempBytes);
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    // 按字节描述单行连续数据，允许尾部元素不足一个 32B 块
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // 矢量指令按 32B 对齐计算，尾块不足一个块的部分在 buffer 留白区内完成，CopyOut 只回拷有效元素
    int64_t alignElem = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    uint32_t calcNum = static_cast<uint32_t>(TanhAlignUp(currentNum, alignElem));
    // tanh(x) = 1 - 2 / (exp(2x) + 1)：x 趋向正无穷时分母溢出为 inf，2/inf 得到 0，结果自然收敛到 1，数值稳定
    if constexpr (sizeof(T) == sizeof(half)) {
        LocalTensor<float> xFp32 = tempBufXFp32.Get<float>();
        LocalTensor<float> yFp32 = tempBufYFp32.Get<float>();
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, calcNum);
        Muls(xFp32, xFp32, 2.0f, calcNum);
        Exp(xFp32, xFp32, calcNum);
        Adds(xFp32, xFp32, 1.0f, calcNum);
        Duplicate(yFp32, 2.0f, calcNum);
        Div(yFp32, yFp32, xFp32, calcNum);
        Muls(yFp32, yFp32, -1.0f, calcNum);
        Adds(yFp32, yFp32, 1.0f, calcNum);
        Cast(yLocal, yFp32, RoundMode::CAST_RINT, calcNum);
    } else {
        Muls(xLocal, xLocal, 2.0f, calcNum);
        Exp(xLocal, xLocal, calcNum);
        Adds(xLocal, xLocal, 1.0f, calcNum);
        Duplicate(yLocal, 2.0f, calcNum);
        Div(yLocal, yLocal, xLocal, calcNum);
        Muls(yLocal, yLocal, -1.0f, calcNum);
        Adds(yLocal, yLocal, 1.0f, calcNum);
    }
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    // 本核没有数据（空 tensor 或多余核）时直接返回
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = TanhCeilDiv(blockLength_, ubLength_);
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = blockLength_ - i * ubLength_;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
