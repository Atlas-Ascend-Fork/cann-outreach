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

// 多轮循环时输入/输出队列的缓冲块数（双缓冲：MTE2 搬入 / V 计算 / MTE3 搬出 相互重叠）
constexpr int32_t BUFFER_NUM = 2;
// UB 数据块大小（字节）：搬运对齐与申请空间的粒度
constexpr int64_t UB_BLOCK_BYTES = 32;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // 单轮直通路径：整个核只有一次搬入/搬出，单缓冲即可，省掉一份 UB 与一层流水调度
    __aicore__ inline void RunSingle(int64_t num);
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    // 32B 对齐走 DataCopy，非 32B 对齐的尾块走 DataCopyPad
    __aicore__ inline void CopyInTo(const LocalTensor<T>& dst, int64_t offset, int64_t num);
    __aicore__ inline void CopyOutFrom(const LocalTensor<T>& src, int64_t offset, int64_t num);

private:
    TPipe pipe;
    // 多轮：输入/输出双缓冲队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // 单轮：单缓冲队列
    TQue<QuePosition::VECIN, 1> inputQueueX1;
    TQue<QuePosition::VECOUT, 1> outputQueueY1;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;  // 本核负责的元素个数
    int64_t ubLength_ = 0;     // 单次 UB 循环处理的元素个数（已按 32B 对齐）
};

// 向上对齐到 align 个元素
__aicore__ inline int64_t CeilAlignNum(int64_t num, int64_t align)
{
    return ((num + align - 1) / align) * align;
}

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 1. 多核切分：第 blockIdx 个核处理 [blockIdx * blockFactor, (blockIdx + 1) * blockFactor)
    int64_t gmOffset = tilingData->blockFactor * static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t remainNum = tilingData->totalNum - gmOffset;
    blockLength_ = 0;
    if (remainNum > 0) {
        blockLength_ = (remainNum < tilingData->blockFactor) ? remainNum : tilingData->blockFactor;
    }

    // 2. 单次循环长度：尾核数据量小于 ubFactor 时按实际数据量分配，并按 32B 向上取整
    //    （InitBuffer 申请的空间必须 32B 对齐）
    int64_t alignElems = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }
    ubLength_ = CeilAlignNum(ubLength_, alignElems);
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        // 数据量已经被前面的核分完，本核直接返回，不申请任何 UB
        return;
    }

    // 3. 绑定本核的 Global Memory 起始地址（按元素偏移，偏移量 = blockIdx * blockFactor）
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + gmOffset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + gmOffset, blockLength_);

    // 4. 按循环轮数选择缓冲策略
    uint32_t bufBytes = static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T)));
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    if (loopCount == 1) {
        pipe.InitBuffer(inputQueueX1, 1, bufBytes);
        pipe.InitBuffer(outputQueueY1, 1, bufBytes);
    } else {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufBytes);
    }
}

template <typename T>
__aicore__ inline void Relu<T>::CopyInTo(const LocalTensor<T>& dst, int64_t offset, int64_t num)
{
    if ((num * static_cast<int64_t>(sizeof(T))) % UB_BLOCK_BYTES == 0) {
        // 32B 对齐：走 DataCopy，没有 pad 的字节级处理
        DataCopy(dst, inputGMX[offset], static_cast<uint32_t>(num));
    } else {
        // 非对齐尾块：走 DataCopyPad，由硬件补齐 dummy
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(num * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dst, inputGMX[offset], copyParams, padParams);
    }
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOutFrom(const LocalTensor<T>& src, int64_t offset, int64_t num)
{
    if ((num * static_cast<int64_t>(sizeof(T))) % UB_BLOCK_BYTES == 0) {
        DataCopy(outputGMY[offset], src, static_cast<uint32_t>(num));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(num * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], src, copyParams);
    }
}

template <typename T>
__aicore__ inline void Relu<T>::RunSingle(int64_t num)
{
    LocalTensor<T> xLocal = inputQueueX1.AllocTensor<T>();
    CopyInTo(xLocal, 0, num);
    inputQueueX1.EnQue(xLocal);

    LocalTensor<T> yLocal = outputQueueY1.AllocTensor<T>();
    LocalTensor<T> xSrc = inputQueueX1.DeQue<T>();
    // y = max(x, 0)：一条标量双目指令完成，不需要 Cast、也不需要临时 buffer
    AscendC::Maxs(yLocal, xSrc, static_cast<T>(0), static_cast<int32_t>(num));
    inputQueueX1.FreeTensor(xSrc);
    outputQueueY1.EnQue<T>(yLocal);

    LocalTensor<T> yOut = outputQueueY1.DeQue<T>();
    CopyOutFrom(yOut, 0, num);
    outputQueueY1.FreeTensor(yOut);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    CopyInTo(xLocal, progress * ubLength_, currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    CopyOutFrom(yLocal, progress * ubLength_, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    if (loopCount == 1) {
        // 单轮（赛题 (8,2048) 就是这种情形）：单缓冲直通，指令数与同步开销最小
        RunSingle(blockLength_);
        return;
    }
    // 多轮：双缓冲流水，第 i 块的搬出与第 i+1 块的搬入重叠
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

} // namespace NsRelu
#endif // RELU_H
