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

constexpr int32_t BUFFER_NUM = 2;
// 一条 DataCopy 指令要求搬运长度按 32B 对齐，未对齐时退回 DataCopyPad
constexpr int64_t ALIGN_BYTES = 32;

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
    __aicore__ inline int64_t AlignElem() const
    {
        return ALIGN_BYTES / static_cast<int64_t>(sizeof(T));
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t loopCount_ = 0;
    int64_t lastTileNum_ = 0;
    // 本核 GM 起点是否 32B 对齐：对齐才能走 DataCopy，否则只能用 DataCopyPad
    bool baseAligned_ = true;
};

// ---------------------------------------------------------------------------
// Init：多核切分（每个核的 GM 起始地址） + UB 内存申请
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    int64_t totalNum = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;  // 每个核负责的元素数
    ubLength_ = tilingData->ubFactor;        // 每次 UB 循环处理的元素数

    // 多核切分：第 blockIdx 个核从 blockIdx * blockFactor 开始
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t offset = blockIdx * blockLength_;
    if (offset >= totalNum) {
        blockLength_ = 0;  // 该核没有数据
    } else if (offset + blockLength_ > totalNum) {
        blockLength_ = totalNum - offset;  // 最后一个核吃掉尾差
    }

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;  // 空张量 / 本核无数据，不需要申请 UB
    }

    inputGMX.SetGlobalBuffer((__gm__ T *)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T *)y + offset, blockLength_);

    // 循环次数 / 末块长度只算一次，省掉每轮循环里的除法和分支
    loopCount_ = (blockLength_ + ubLength_ - 1) / ubLength_;
    lastTileNum_ = blockLength_ - (loopCount_ - 1) * ubLength_;

    // 本核起点对齐情况（tiling 侧已保证 blockFactor 32B 对齐，这里只做兜底）
    baseAligned_ = (offset % AlignElem() == 0);

    // 按"实际用到的长度"申请 UB：本核数据不足一个 ubLength 时不必占满；向上对齐保证 pad 有空间
    int64_t alignElem = AlignElem();
    int64_t tileNum = (blockLength_ < ubLength_) ? blockLength_ : ubLength_;
    int64_t bufElem = ((tileNum + alignElem - 1) / alignElem) * alignElem;

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufElem * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufElem * sizeof(T));
}

// ---------------------------------------------------------------------------
// CopyIn：GM -> UB
//   32B 对齐走 DataCopy（一次 DMA 指令、效率最高）
//   尾部不足 32B 对齐时才退回 DataCopyPad（按字节搬运 + 自动 pad）
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    int64_t startOffset = progress * ubLength_;
    if (baseAligned_ && (currentNum * static_cast<int64_t>(sizeof(T))) % ALIGN_BYTES == 0) {
        DataCopy(xLocal, inputGMX[startOffset], static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, inputGMX[startOffset], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

// ---------------------------------------------------------------------------
// Compute：y = max(0, x)
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // 逐元素与标量 0 取最大值，即 relu
    Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<uint32_t>(currentNum));

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// ---------------------------------------------------------------------------
// CopyOut：UB -> GM（同样对齐走 DataCopy，未对齐走 DataCopyPad）
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    int64_t startOffset = progress * ubLength_;
    if (baseAligned_ && (currentNum * static_cast<int64_t>(sizeof(T))) % ALIGN_BYTES == 0) {
        DataCopy(outputGMY[startOffset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[startOffset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

// ---------------------------------------------------------------------------
// Process：三级流水（MTE2 搬入 / VEC 计算 / MTE3 搬出 三者互相重叠）
//   - 每轮先发起下一块搬入（MTE2 与 VEC 重叠）
//   - 当前块算完后不立刻写回，改成写回"上一块"，让 MTE3 与本轮 VEC 重叠
//   - inputQueue / outputQueue 各 BUFFER_NUM=2 份：
//       迭代 i 时 output 队列最多同时挂 i-1(待写回) 与 i(刚算完) 两块，深度 2 刚好、不死锁
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0 || loopCount_ <= 0) {
        return;
    }

    if (loopCount_ == 1) {  // 单块：没有可重叠的对象，走最简三段，省掉流水分支判断
        CopyIn(0, lastTileNum_);
        Compute(lastTileNum_);
        CopyOut(0, lastTileNum_);
        return;
    }

    // 首块搬入
    CopyIn(0, ubLength_);

    for (int64_t i = 0; i < loopCount_; i++) {
        // 预取下一块：DMA 先发出去，随后与本块的 Compute 并行执行
        if (i + 1 < loopCount_) {
            int64_t nextNum = (i + 1 == loopCount_ - 1) ? lastTileNum_ : ubLength_;
            CopyIn(i + 1, nextNum);
        }

        int64_t currentNum = (i == loopCount_ - 1) ? lastTileNum_ : ubLength_;
        Compute(currentNum);

        // 延迟写回上一块：i-1 一定不是末块，长度必为 ubLength_，与本次 Compute 重叠
        if (i > 0) {
            CopyOut(i - 1, ubLength_);
        }
    }

    // 末块写回
    CopyOut(loopCount_ - 1, lastTileNum_);
}

} // namespace NsRelu
#endif // RELU_H
