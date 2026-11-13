/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 *
 *        计算方式按实验要求使用**小算子拼接**，不调用 Tanh 高阶 API：
 *            tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 *        为便于 float16 / float32 共用同一套计算流程并保证精度，
 *        所有计算统一在 float32 上进行：fp16 输入先 Cast 到 float32，算完再 Cast 回 fp16。
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;         // 每队列 buffer 块数（双缓冲）
constexpr int64_t ALIGN_BYTES = 32;       // UB 地址/长度对齐字节数
constexpr int32_t FLOAT_TMP_NUM = 5;      // 计算用 float32 临时 buffer 个数（与 host tiling 保持一致）
// 限幅值：|x| 大于该值时 float32 下 exp 会溢出（exp(88) 溢出），而 tanh 早已饱和为 ±1
constexpr float TANH_CLAMP_VALUE = 20.0f;

__aicore__ inline int64_t AlignUpInt(int64_t value, int64_t align)
{
    if (align <= 0) {
        return value;
    }
    return (value + align - 1) / align * align;
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

    // 计算过程所需的临时变量（5 个 float32 buffer，均按单次循环处理元素数分配）
    TBuf<TPosition::VECCALC> xFloatBuf;    // x                     （限幅后）
    TBuf<TPosition::VECCALC> expXBuf;      // exp(x)
    TBuf<TPosition::VECCALC> expNegXBuf;   // exp(-x)
    TBuf<TPosition::VECCALC> numeratorBuf; // 分子 exp(x) - exp(-x)
    TBuf<TPosition::VECCALC> denomBuf;     // 分母 exp(x) + exp(-x)

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0; // 当前核负责的元素个数
    int64_t ubLength_ = 0;    // 每次 UB 循环处理的元素个数（32B 对齐）
    int64_t alignNum_ = 1;    // 32B 对齐对应的元素个数
    int64_t totalNum_ = 0;    // 总元素个数
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    int64_t ubFactor = tilingData->ubFactor;

    alignNum_ = ALIGN_BYTES / static_cast<int64_t>(sizeof(T));
    if (alignNum_ < 1) {
        alignNum_ = 1;
    }
    if (blockFactor < 1) {
        blockFactor = 1;
    }
    if (ubFactor < alignNum_) {
        ubFactor = alignNum_;
    }
    ubLength_ = AlignUpInt(ubFactor, alignNum_);
    blockLength_ = 0;

    // 按核切分：第 blockIdx 个核处理 [blockIdx * blockFactor, (blockIdx + 1) * blockFactor)
    int64_t blockOffset = static_cast<int64_t>(AscendC::GetBlockIdx()) * blockFactor;
    if (blockOffset < totalNum_) {
        blockLength_ = totalNum_ - blockOffset;
        if (blockLength_ > blockFactor) {
            blockLength_ = blockFactor;
        }
    }
    // 尾核无数据时直接返回，避免申请超出实际长度的 UB 空间
    if (blockLength_ <= 0) {
        return;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);

    // 输入/输出队列：各 BUFFER_NUM 块，每块 ubLength_ 个元素
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T))));
    // 计算临时变量：统一 float32
    const uint32_t floatBytes = static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(xFloatBuf, floatBytes);
    pipe.InitBuffer(expXBuf, floatBytes);
    pipe.InitBuffer(expNegXBuf, floatBytes);
    pipe.InitBuffer(numeratorBuf, floatBytes);
    pipe.InitBuffer(denomBuf, floatBytes);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    int64_t offset = progress * ubLength_;
    int64_t alignedNum = AlignUpInt(currentNum, alignNum_);
    if (alignedNum == currentNum) {
        // 32B 对齐路径，走 DataCopy 快路径
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        // 非对齐尾块：按字节长度搬运，并把尾部补齐为 0（避免后续 exp 读到脏数据）
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
        Duplicate(xLocal[currentNum], static_cast<T>(0), static_cast<int32_t>(alignedNum - currentNum));
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // 向量指令按 32B 对齐的长度计算，尾部补 0 元素的结果不会写回 GM
    const int64_t alignedNum = AlignUpInt(currentNum, alignNum_);
    const int32_t calCount = static_cast<int32_t>(alignedNum);

    LocalTensor<float> xFloat = xFloatBuf.Get<float>();
    LocalTensor<float> expX = expXBuf.Get<float>();
    LocalTensor<float> expNegX = expNegXBuf.Get<float>();
    LocalTensor<float> numerator = numeratorBuf.Get<float>();
    LocalTensor<float> denom = denomBuf.Get<float>();

    // 1. 输入统一转成 float32（float16 精度不足，需在 float32 上计算完再转回）
    if constexpr (sizeof(T) == 2) {
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedNum));
    } else {
        Muls(xFloat, xLocal, 1.0f, calCount);
    }

    // 2. 限幅到 [-TANH_CLAMP_VALUE, TANH_CLAMP_VALUE]，避免 exp 溢出为 inf 后出现 inf/inf
    Mins(denom, xFloat, TANH_CLAMP_VALUE, calCount);
    Maxs(xFloat, denom, -TANH_CLAMP_VALUE, calCount);

    // 3. exp(x) 与 exp(-x)
    Exp(expX, xFloat, calCount);            // exp(x)
    Muls(denom, xFloat, -1.0f, calCount);   // -x
    Exp(expNegX, denom, calCount);          // exp(-x)

    // 4. tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Sub(numerator, expX, expNegX, calCount); // 分子
    Add(denom, expX, expNegX, calCount);     // 分母
    Div(xFloat, numerator, denom, calCount); // float32 结果

    // 5. 转回输出 dtype
    if constexpr (sizeof(T) == 2) {
        Cast(yLocal, xFloat, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedNum));
    } else {
        Muls(yLocal, xFloat, 1.0f, calCount);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    int64_t offset = progress * ubLength_;
    int64_t alignedNum = AlignUpInt(currentNum, alignNum_);
    if (alignedNum == currentNum) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
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
