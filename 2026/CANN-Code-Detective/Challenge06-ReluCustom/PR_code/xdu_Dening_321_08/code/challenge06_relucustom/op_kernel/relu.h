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

constexpr int32_t BUFFER_NUM = 2;        // 每队列 buffer 块数（双缓冲）
constexpr int32_t COPY_ALIGN_BYTE = 32;  // DataCopy 要求的最小对齐单元
constexpr int64_t MAX_UB_ELEM_NUM = 4096; // 单次 UB 处理元素数兜底上限（fp32 约 64KB，fp16 约 32KB）

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// 初始化：按 tiling 结果切分多核区间并分配 UB
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 本算子是逐元素运算，多核按连续区间等分（末尾核可能不满）
    const int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    if (blockFactor <= 0 || blockFactor > totalNum) {
        blockFactor = totalNum;
    }

    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t startIdx = blockFactor * blockIdx;
    const int64_t leftNum = totalNum - startIdx;
    blockLength_ = (leftNum > 0) ? ((leftNum < blockFactor) ? leftNum : blockFactor) : 0;

    // 单次 UB 处理元素数：取 tiling 下发值，并按 32 字节对齐 + UB 容量兜底裁剪
    const int64_t alignNum = COPY_ALIGN_BYTE / static_cast<int64_t>(sizeof(T));
    int64_t ubNum = tilingData->ubFactor;
    if (ubNum <= 0 || ubNum > blockLength_) {
        ubNum = blockLength_;
    }
    if (ubNum > MAX_UB_ELEM_NUM) {
        ubNum = MAX_UB_ELEM_NUM;
    }
    ubLength_ = (ubNum / alignNum) * alignNum;
    if (ubLength_ <= 0) {
        ubLength_ = alignNum;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startIdx, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startIdx, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

// GM -> UB：长度为 32 字节整数倍时用 DataCopy，尾块用 DataCopyPad
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    const int64_t alignNum = COPY_ALIGN_BYTE / static_cast<int64_t>(sizeof(T));
    const uint32_t offset = static_cast<uint32_t>(progress);
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    if (currentNum % alignNum == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams(1, static_cast<uint32_t>(currentNum), 0, 0, 0);
        DataCopyPadExtParams<T> padParams(true, 0, 0, static_cast<T>(0));
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

// UB -> GM：尾块用 3 参 DataCopyPad，不向 GM 补齐，避免越界写坏相邻核的数据
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    const int64_t alignNum = COPY_ALIGN_BYTE / static_cast<int64_t>(sizeof(T));
    const uint32_t offset = static_cast<uint32_t>(progress);
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    if (currentNum % alignNum == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams(1, static_cast<uint32_t>(currentNum), 0, 0, 0);
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

// y = max(0, x)：Maxs 直接对原精度取最大值，relu 不会引入任何精度损失
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// 分块流水线：CopyIn -> Compute -> CopyOut
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    for (int64_t progress = 0; progress < blockLength_; progress += ubLength_) {
        const int64_t leftNum = blockLength_ - progress;
        const int64_t currentNum = (leftNum < ubLength_) ? leftNum : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
