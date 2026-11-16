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

// 多轮循环时输入/输出队列的缓冲块数（双缓冲流水）
constexpr int32_t BUFFER_NUM = 2;
// UB 数据块大小（字节）：搬运对齐与申请空间的粒度
constexpr int64_t UB_BLOCK_BYTES = 32;
// Tanh 临时空间按元素计算的字节数：half 需 2 个 float 缓冲、float 需 1 个，取 8B 覆盖两种情况
constexpr int64_t TMP_BYTES_PER_ELEM = 8;
// 临时空间上限：Tanh 内部会自己按 stackSize 分块循环，临时空间不必随元素数线性增长
constexpr int64_t TMP_BYTES_MAX = 16384;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // 单轮直通路径：只有一次搬入/搬出，不需要双缓冲流水
    __aicore__ inline void RunSingle(int64_t num);
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    // 对齐块走 DataCopy，非对齐尾块走 DataCopyPad
    __aicore__ inline void CopyInTo(const LocalTensor<T>& dst, int64_t offset, int64_t num);
    __aicore__ inline void CopyOutFrom(const LocalTensor<T>& src, int64_t offset, int64_t num);

private:
    TPipe pipe;
    // 多轮：输入/输出双缓冲队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // 单轮：单缓冲队列（只申请一份 UB，省掉一份空间与一层流水调度）
    TQue<QuePosition::VECIN, 1> inputQueueX1;
    TQue<QuePosition::VECOUT, 1> outputQueueY1;
    // Tanh 计算过程所需临时变量（half 需要中转 float，float 需要中间结果）
    TBuf<TPosition::VECCALC> tmpBuffer;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // 多核切分：第 blockIdx 个核处理 [blockIdx * blockFactor, (blockIdx + 1) * blockFactor)
    ubLength_ = tilingData->ubFactor;
    int64_t gmOffset = tilingData->blockFactor * static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t remainNum = tilingData->totalNum - gmOffset;
    blockLength_ = 0;
    if (remainNum > 0) {
        blockLength_ = (remainNum < tilingData->blockFactor) ? remainNum : tilingData->blockFactor;
    }
    // 尾核数据量小于 ubFactor 时按实际数据量分配 UB
    if (ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }
    // InitBuffer 申请的空间必须按 UB_BLOCK_BYTES 对齐，尾核需向上取整
    int64_t alignElems = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    ubLength_ = ((ubLength_ + alignElems - 1) / alignElems) * alignElems;
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + gmOffset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + gmOffset, blockLength_);

    uint32_t bufBytes = static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T)));
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    if (loopCount == 1) {
        // 单轮：一份输入 + 一份输出即可，省掉双缓冲那份额外 UB
        pipe.InitBuffer(inputQueueX1, 1, bufBytes);
        pipe.InitBuffer(outputQueueY1, 1, bufBytes);
    } else {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufBytes);
    }

    // Tanh 临时空间按实际元素数申请（并按上限封顶，避免尾核/大块把 UB 撑爆）
    int64_t tmpBytes = ubLength_ * TMP_BYTES_PER_ELEM;
    tmpBytes = ((tmpBytes + UB_BLOCK_BYTES - 1) / UB_BLOCK_BYTES) * UB_BLOCK_BYTES;
    if (tmpBytes > TMP_BYTES_MAX) {
        tmpBytes = TMP_BYTES_MAX;
    }
    pipe.InitBuffer(tmpBuffer, static_cast<uint32_t>(tmpBytes));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyInTo(const LocalTensor<T>& dst, int64_t offset, int64_t num)
{
    if ((num * static_cast<int64_t>(sizeof(T))) % UB_BLOCK_BYTES == 0) {
        // 32B 对齐：走 DataCopy，省掉 DataCopyPad 的字节级/pad 处理
        DataCopy(dst, inputGMX[offset], static_cast<uint32_t>(num));
    } else {
        // 非对齐尾块：走 DataCopyPad，由硬件补齐 dummy
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(num * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dst, inputGMX[offset], copyParams, padParams);
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOutFrom(const LocalTensor<T>& src, int64_t offset, int64_t num)
{
    if ((num * static_cast<int64_t>(sizeof(T))) % UB_BLOCK_BYTES == 0) {
        DataCopy(outputGMY[offset], src, static_cast<uint32_t>(num));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(num * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], src, copyParams);
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::RunSingle(int64_t num)
{
    LocalTensor<T> xLocal = inputQueueX1.AllocTensor<T>();
    CopyInTo(xLocal, 0, num);
    inputQueueX1.EnQue(xLocal);

    LocalTensor<T> yLocal = outputQueueY1.AllocTensor<T>();
    LocalTensor<T> xSrc = inputQueueX1.DeQue<T>();
    // Tanh 高阶 API：内部按 (e^(2x)-1)/(e^(2x)+1) 计算，half 会先转 float 再转回
    LocalTensor<uint8_t> sharedTmpBuffer = tmpBuffer.Get<uint8_t>();
    AscendC::Tanh(yLocal, xSrc, sharedTmpBuffer, static_cast<uint32_t>(num));
    inputQueueX1.FreeTensor(xSrc);
    outputQueueY1.EnQue<T>(yLocal);

    LocalTensor<T> yOut = outputQueueY1.DeQue<T>();
    CopyOutFrom(yOut, 0, num);
    outputQueueY1.FreeTensor(yOut);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    CopyInTo(xLocal, progress * ubLength_, currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<uint8_t> sharedTmpBuffer = tmpBuffer.Get<uint8_t>();
    AscendC::Tanh(yLocal, xLocal, sharedTmpBuffer, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    CopyOutFrom(yLocal, progress * ubLength_, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    if (loopCount == 1) {
        // 单轮：走单缓冲直通，减少一次 UB 申请与一层队列同步
        RunSingle(blockLength_);
        return;
    }
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
