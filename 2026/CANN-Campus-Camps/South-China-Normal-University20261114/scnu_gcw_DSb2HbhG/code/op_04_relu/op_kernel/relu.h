/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 *
 * ReLU: y = max(0, x)
 *
 * 实现要点（ReLU 为访存受限算子，优化目标是打满 HBM 带宽并隐藏搬运时延）：
 *   1. 多核并行：每个核按 tiling 的 blockFactor 划分出一段连续的 Global Memory 区间，
 *      起始地址 = blockIdx * blockFactor（32B 对齐），从而各核之间互不重叠、且对齐搬运；
 *   2. 核内 DoubleBuffer：输入/输出 TQue 深度均为 BUFFER_NUM(=2)，配合 EnQue/DeQue 的
 *      事件同步，使 CopyIn(MTE2)、Relu(Vector)、CopyOut(MTE3) 三条流水并行执行，
 *      在计算当前块的同时预取下一块、回写上一块；
 *   3. 尾块处理：最后一个核的有效长度可能小于 blockFactor，按实际长度搬运；长度为 32B
 *      整数倍时走 DataCopy（带宽最优），否则走 DataCopyPad 完成非对齐搬运。
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
// DataCopy 的地址对齐粒度 / 单次搬运的最小单位（byte）
constexpr int64_t UB_BLOCK_SIZE = 32;
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
    __aicore__ inline int64_t GetTileLength(int64_t tileIndex) const
    {
        const int64_t remain = blockLength_ - tileIndex * ubLength_;
        return (remain < ubLength_) ? remain : ubLength_;
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;  // 当前核实际需要处理的元素个数
    int64_t ubLength_ = 0;     // 单次循环（一个 tile）处理的元素个数
};
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    const int64_t totalNum = tilingData->totalNum;
    const int64_t blockFactor = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }
    // 当前核负责的 GM 区间：[blockIdx * blockFactor, min(totalNum, ...))
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t gmOffset = blockIdx * blockFactor;
    if (blockFactor <= 0 || gmOffset >= totalNum) {
        blockLength_ = 0;
    } else {
        blockLength_ = (totalNum - gmOffset < blockFactor) ? (totalNum - gmOffset) : blockFactor;
    }
    if (blockLength_ > 0) {
        // 设置每个核的 Global Memory 起始地址（32B 对齐）与长度
        inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, static_cast<uint64_t>(blockLength_));
        outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, static_cast<uint64_t>(blockLength_));
    }
    // DoubleBuffer：输入、输出各 BUFFER_NUM 块，单块大小为 ubLength_ * sizeof(T)
    const uint32_t tileSize = static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileSize);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileSize);
}
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t byteSize = currentNum * static_cast<int64_t>(sizeof(T));
    if ((byteSize & (UB_BLOCK_SIZE - 1)) == 0) {
        // 32B 对齐，使用 DataCopy 获得最大搬运带宽
        DataCopyParams copyParams{1, static_cast<uint16_t>(byteSize / UB_BLOCK_SIZE), 0, 0};
        DataCopy(xLocal, inputGMX[progress], copyParams);
    } else {
        // 尾块非对齐，使用 DataCopyPad 按字节搬运
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(byteSize), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0.0F)};
        DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // y = max(0, x)，单条 Vector 指令完成
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t byteSize = currentNum * static_cast<int64_t>(sizeof(T));
    if ((byteSize & (UB_BLOCK_SIZE - 1)) == 0) {
        DataCopyParams copyParams{1, static_cast<uint16_t>(byteSize / UB_BLOCK_SIZE), 0, 0};
        DataCopy(outputGMY[progress], yLocal, copyParams);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(byteSize), 0, 0, 0};
        DataCopyPad(outputGMY[progress], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    // 核内按 ubLength_ 分 tile 循环，DoubleBuffer + 预取下一块：
    //   CopyIn(i+1) 先于 Compute(i) 下发，使 MTE2（拷入）、Vector（计算）、MTE3（拷出）
    //   三条流水并行，从而隐藏搬运时延。
    const int64_t tileNum = (blockLength_ + ubLength_ - 1) / ubLength_;
    CopyIn(0, GetTileLength(0));
    for (int64_t i = 0; i < tileNum; i++) {
        if (i + 1 < tileNum) {
            CopyIn((i + 1) * ubLength_, GetTileLength(i + 1));
        }
        Compute(GetTileLength(i));
        CopyOut(i * ubLength_, GetTileLength(i));
    }
}
} // namespace NsRelu
#endif // RELU_H