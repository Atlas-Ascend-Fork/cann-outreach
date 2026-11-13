/*!
 * \file exp.h
 * \brief Exp 算子 kernel 类定义
 */

#ifndef EXP_H
#define EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
// 单次 UB 循环元素数上限：fp32 下 2048 元素 = 8KB/队列，双队列双缓冲共 32KB，远小于 192KB UB
constexpr int64_t UB_MAX_ELEMS = 2048;
// DataCopy / UB 分配的最小对齐单元（字节）
constexpr int64_t ALIGN_BYTES = 32;

template <typename T>
class Exp {
public:
    __aicore__ inline Exp(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData);
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

    int64_t blockLength_ = 0;     // 本核实际负责的元素数（尾核可能不满 blockFactor）
    int64_t ubLength_ = 0;        // 单次 UB 循环处理的元素数（已按 32B 对齐）
    int64_t tileNum_ = 0;         // 本核内的 UB 循环次数
    int64_t lastTileLength_ = 0;  // 本核最后一块的实际元素数（可能不满 ubLength_）
};

template <typename T>
__aicore__ inline void Exp<T>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
{
    // 多核切分：每核按 blockFactor 窗口编址，尾核只处理剩余元素，保证不重叠不越界
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t totalNum = tilingData->totalNum;
    const int64_t offset = blockIdx * blockFactor;
    const int64_t remain = totalNum - offset;
    // 空 tensor 或超出数据的空核：blockLength_ 保持 0，Process 直接返回，不做任何搬运
    if (blockFactor <= 0 || remain <= 0) {
        blockLength_ = 0;
        return;
    }
    blockLength_ = (remain < blockFactor) ? remain : blockFactor;

    // 单次循环元素数：host 下发的 ubFactor 与本核长度取小，再依次做上限裁剪、32B 对齐、下限保护
    int64_t ubRaw = tilingData->ubFactor;
    if (ubRaw > blockLength_) {
        ubRaw = blockLength_;
    }
    // 上限：UB 预算（防止上游下发超大 ubFactor 导致 InitBuffer 超容量）
    if (ubRaw > UB_MAX_ELEMS) {
        ubRaw = UB_MAX_ELEMS;
    }
    const int64_t alignElems = ALIGN_BYTES / static_cast<int64_t>(sizeof(T));  // 32B 对应的元素数
    // 32B 对齐：向下取整，保证 buffer 不超预算
    ubLength_ = (ubRaw / alignElems) * alignElems;
    // 下限：至少 1 个对齐单元，避免 ubLength_ 为 0 造成除零与零字节分配
    if (ubLength_ <= 0) {
        ubLength_ = alignElems;
    }
    tileNum_ = (blockLength_ + ubLength_ - 1) / ubLength_;
    lastTileLength_ = blockLength_ - (tileNum_ - 1) * ubLength_;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    // bufSize 按 32B 向上取整，满足 UB 分配的粒度要求
    const int64_t bufRaw = ubLength_ * static_cast<int64_t>(sizeof(T));
    const int64_t bufBytes = ((bufRaw + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufBytes);
}

template <typename T>
__aicore__ inline void Exp<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // 逐元素自然指数：y[i] = exp(x[i])
    AscendC::Exp(yLocal, xLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Process()
{
    // 空 tensor / 空核直接返回，防止越界访问
    if (blockLength_ <= 0 || ubLength_ <= 0 || tileNum_ <= 0) {
        return;
    }
    // CopyIn -> Compute -> CopyOut 三段式流水线，最后一块按实际剩余元素数处理
    for (int64_t i = 0; i < tileNum_; i++) {
        const int64_t currentNum = (i == tileNum_ - 1) ? lastTileLength_ : ubLength_;
        if (currentNum <= 0) {
            break;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsExp
#endif // EXP_H
