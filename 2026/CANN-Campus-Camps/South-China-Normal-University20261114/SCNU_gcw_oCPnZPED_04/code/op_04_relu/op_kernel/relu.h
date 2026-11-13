
/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

// #ifndef RELU_H
// #define RELU_H

// #include "kernel_operator.h"
// #include "kernel_tiling/kernel_tiling.h"
// #include "relu_tiling_data.h"
// #include "relu_tiling_key.h"

// namespace NsRelu {

// using namespace AscendC;

// /* 双缓冲：大 shape（2048x2048）每核要跑很多轮，靠它把 MTE2 搬入、VECTOR 计算、
//    MTE3 搬出重叠起来；小 shape（8x2048）每核只有一轮，depth=2 也不会变慢 */
// constexpr int32_t BUFFER_NUM = 2;

// template <typename T>
// class Relu {
// public:
//     __aicore__ inline Relu(){};

//     __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
//     __aicore__ inline void Process();

// private:
//     __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
//     __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
//     __aicore__ inline void Compute(int64_t currentNum);

// private:
//     TPipe pipe;
//     TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
//     TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

//     GlobalTensor<T> inputGMX;
//     GlobalTensor<T> outputGMY;

//     int64_t blockLength_ = 0;
//     int64_t ubLength_ = 0;
// };

// template <typename T>
// __aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
// {
//     blockLength_ = tilingData->blockFactor;
//     ubLength_ = tilingData->ubFactor;

//     /* 本核负责的数据段起点 = 核号 * 每核处理量 */
//     int64_t offset = static_cast<int64_t>(GetBlockIdx()) * blockLength_;
//     if (offset + blockLength_ > tilingData->totalNum) {
//         blockLength_ = tilingData->totalNum - offset;
//     }
//     if (blockLength_ <= 0) {
//         blockLength_ = 0;
//         return;   /* 空核直接退出，不申请 UB */
//     }
//     if (ubLength_ <= 0 || ubLength_ > blockLength_) {
//         ubLength_ = blockLength_;
//     }

//     inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
//     outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

//     pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
//     pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
// }

// template <typename T>
// __aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
// {
//     LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
//     DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
//     inputQueueX.EnQue(xLocal);
// }

// template <typename T>
// __aicore__ inline void Relu<T>::Compute(int64_t currentNum)
// {
//     LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
//     LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

//     /* ReLU 就是 max(x, 0)：一条 Maxs 指令搞定，不需要任何 float 中间 buffer、
//        也不需要 Cast（输入输出同 dtype）。比 ReLU 高阶 API 更稳、更快 */
//     Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));

//     outputQueueY.EnQue(yLocal);
//     inputQueueX.FreeTensor(xLocal);
// }

// template <typename T>
// __aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
// {
//     LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
//     DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
//     outputQueueY.FreeTensor(yLocal);
// }

// template <typename T>
// __aicore__ inline void Relu<T>::Process()
// {
//     if (blockLength_ <= 0) {
//         return;
//     }
//     const int64_t ubLoopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
//     for (int64_t i = 0; i < ubLoopCount; ++i) {
//         int64_t currentNum = (i == ubLoopCount - 1) ? (blockLength_ - ubLength_ * i) : ubLength_;
//         CopyIn(i, currentNum);
//         Compute(currentNum);
//         CopyOut(i, currentNum);
//     }
// }

// } // namespace NsRelu
// #endif // RELU_H

/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义（交错分块 + 软件流水）
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

/* 输入队列深度 3：配合 PREFETCH_NUM = 2，任何时刻有 2 块数据正在由 MTE2 搬运 */
constexpr int32_t IN_BUFFER_NUM = 3;
/* 输出队列深度 2 */
constexpr int32_t OUT_BUFFER_NUM = 2;
/* 软件流水预取深度（只能等于 IN_BUFFER_NUM - 1） */
constexpr int64_t PREFETCH_NUM = 2;

/* 【分块方式开关】true = 交错（循环）分块：核 b 处理 tile b、b+核数、b+2*核数…
   false = 每核一段连续区间（老做法）。如果交错反而慢，把这里改成 false 即可回退 */
constexpr bool TILE_INTERLEAVED = true;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t k, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t k, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    /* 第 k 个“本核任务块”对应的全局 tile 序号 */
    __aicore__ inline int64_t TileIndex(int64_t k) const;
    /* 第 k 个任务块在全局数据里的起始下标 */
    __aicore__ inline int64_t TileOffset(int64_t k) const;
    /* 第 k 个任务块实际元素数（末尾块可能不满） */
    __aicore__ inline int64_t TileCount(int64_t k) const;
    /* 第 k 个任务块是否还有效 */
    __aicore__ inline bool TileValid(int64_t k) const;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, IN_BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, OUT_BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;      /* 全局元素总数 */
    int64_t tileLen_ = 0;       /* 一个 tile 的元素数 */
    int64_t blockNum_ = 1;      /* 启动的核数 */
    int64_t blockIdx_ = 0;      /* 本核编号 */
    int64_t tileTotal_ = 0;     /* 全局 tile 总数 */
    int64_t perCoreTiles_ = 0;  /* 连续分块模式：每个核分到几个 tile */
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    tileLen_ = tilingData->ubFactor;
    blockNum_ = static_cast<int64_t>(GetBlockNum());
    blockIdx_ = static_cast<int64_t>(GetBlockIdx());

    if (totalNum_ <= 0 || tileLen_ <= 0 || blockNum_ <= 0) {
        totalNum_ = 0;
        return;   /* 空核直接退出，不申请 UB */
    }
    tileTotal_ = (totalNum_ + tileLen_ - 1) / tileLen_;
    perCoreTiles_ = (tileTotal_ + blockNum_ - 1) / blockNum_;
    if (!TileValid(0)) {
        totalNum_ = 0;
        return;   /* 分不到 tile 的核直接退出，不申请 UB */
    }

    /* 整块连续：所有核共用同一套全局 buffer，靠下标定位，不做 per-core 偏移 */
    inputGMX.SetGlobalBuffer((__gm__ T*)x, totalNum_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, totalNum_);

    /* UB buffer 按 32B 向上取整分配，避免末块不满时长度不对齐 */
    const int64_t elemAlign = 32 / static_cast<int64_t>(sizeof(T));
    const int64_t allocLen = (tileLen_ + elemAlign - 1) / elemAlign * elemAlign;
    pipe.InitBuffer(inputQueueX, IN_BUFFER_NUM, allocLen * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outputQueueY, OUT_BUFFER_NUM, allocLen * static_cast<int64_t>(sizeof(T)));
}

/* 交错（循环）分块：本核处理第 blockIdx_、blockIdx_+blockNum_、blockIdx_+2*blockNum_ … 个 tile。
   这样 40 个核在同一时刻读写的都是相邻的一大片地址（窗口只有 40*tileLen 那么宽），
   DRAM 行局部性和 L2 命中率都比“每个核各占一段很远的连续区间”好得多 */
template <typename T>
__aicore__ inline int64_t Relu<T>::TileIndex(int64_t k) const
{
    if constexpr (TILE_INTERLEAVED) {
        return blockIdx_ + k * blockNum_;
    } else {
        return blockIdx_ * perCoreTiles_ + k;
    }
}

template <typename T>
__aicore__ inline int64_t Relu<T>::TileOffset(int64_t k) const
{
    return TileIndex(k) * tileLen_;
}

template <typename T>
__aicore__ inline int64_t Relu<T>::TileCount(int64_t k) const
{
    int64_t left = totalNum_ - TileOffset(k);
    return (left < tileLen_) ? left : tileLen_;
}

template <typename T>
__aicore__ inline bool Relu<T>::TileValid(int64_t k) const
{
    return TileIndex(k) < tileTotal_;
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t k, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[TileOffset(k)], static_cast<uint32_t>(currentNum));
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
__aicore__ inline void Relu<T>::CopyOut(int64_t k, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[TileOffset(k)], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

/* 软件流水：先把前 PREFETCH_NUM 块的搬入发出去，之后每轮“先发后面第 k+PREFETCH 块的搬入、
   再算第 k 块、再搬出第 k 块”，让 MTE2 在计算/搬出的时间里持续搬运，不空转 */
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (totalNum_ <= 0 || tileLen_ <= 0) {
        return;
    }

    for (int64_t k = 0; k < PREFETCH_NUM && TileValid(k); ++k) {
        CopyIn(k, TileCount(k));
    }

    for (int64_t k = 0; TileValid(k); ++k) {
        if (TileValid(k + PREFETCH_NUM)) {
            CopyIn(k + PREFETCH_NUM, TileCount(k + PREFETCH_NUM));
        }
        const int64_t curNum = TileCount(k);
        Compute(curNum);
        CopyOut(k, curNum);
    }
}

} // namespace NsRelu
#endif // RELU_H

