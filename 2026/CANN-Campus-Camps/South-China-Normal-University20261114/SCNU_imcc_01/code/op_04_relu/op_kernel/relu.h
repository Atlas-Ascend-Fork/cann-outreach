/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义（软件流水版）
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

/* 输入队列深度 4：配合 PREFETCH_NUM = 3，可以有 3 块数据同时处于“正在搬运”状态，
   把 MTE2 的 outstanding 请求数拉满（大 shape 是纯带宽题，这是唯一有效的杠杆） */
constexpr int32_t IN_BUFFER_NUM = 4;
/* 输出队列深度 2：只要够“搬出第 i 块”和“写第 i+1 块”不打架即可 */
constexpr int32_t OUT_BUFFER_NUM = 2;
/* 软件流水预取深度：计算第 i 块时，第 i+1 / i+2 / i+3 块的搬入已经在路上 */
constexpr int64_t PREFETCH_NUM = 3;

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
    __aicore__ inline int64_t TileNum(int64_t progress) const;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, IN_BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, OUT_BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    /* 本核负责的数据段起点 = 核号 * 每核处理量 */
    int64_t offset = static_cast<int64_t>(GetBlockIdx()) * blockLength_;
    if (offset + blockLength_ > tilingData->totalNum) {
        blockLength_ = tilingData->totalNum - offset;
    }
    if (blockLength_ <= 0) {
        blockLength_ = 0;
        return;   /* 空核直接退出，不申请 UB */
    }
    if (ubLength_ <= 0 || ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    /* UB buffer 按 32B 向上取整分配，避免 ubLength_ 末块不满时长度不对齐 */
    const int64_t elemAlign = 32 / static_cast<int64_t>(sizeof(T));
    const int64_t allocLen = (ubLength_ + elemAlign - 1) / elemAlign * elemAlign;
    pipe.InitBuffer(inputQueueX, IN_BUFFER_NUM, allocLen * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outputQueueY, OUT_BUFFER_NUM, allocLen * static_cast<int64_t>(sizeof(T)));
}

/* 第 progress 块实际要处理多少个元素（末块可能不满） */
template <typename T>
__aicore__ inline int64_t Relu<T>::TileNum(int64_t progress) const
{
    int64_t left = blockLength_ - ubLength_ * progress;
    return (left < ubLength_) ? left : ubLength_;
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    /* ReLU 就是 max(x, 0)：一条 Maxs 指令搞定，不需要任何 float 中间 buffer、
       也不需要 Cast（输入输出同 dtype）。比 ReLU 高阶 API 更稳、更快 */
    Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

/* 软件流水：先把前 PREFETCH_NUM 块搬入发出去，之后每轮“先发下一块的搬入、再算当前块”，
   这样 MTE2 在算当前块 / 搬出上一块的时间里一直在搬数据，不会空转。
   （原来的写法 CopyIn→Compute→CopyOut 串行推进，任何时刻只有 1 个搬运请求在飞） */
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    const int64_t prologue = (loopCount < PREFETCH_NUM) ? loopCount : PREFETCH_NUM;

    /* 预热：先把前几块的搬入全部发出去（只发不等） */
    for (int64_t i = 0; i < prologue; ++i) {
        CopyIn(i, TileNum(i));
    }

    /* 主循环：发下一块搬入、算当前块、搬出当前块 */
    for (int64_t i = 0; i < loopCount; ++i) {
        const int64_t next = i + PREFETCH_NUM;
        if (next < loopCount) {
            CopyIn(next, TileNum(next));
        }
        const int64_t curNum = TileNum(i);
        Compute(curNum);
        CopyOut(i, curNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
