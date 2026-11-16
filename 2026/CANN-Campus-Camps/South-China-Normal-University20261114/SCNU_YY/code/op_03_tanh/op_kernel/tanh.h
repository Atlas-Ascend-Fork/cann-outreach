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

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t totalNum = tilingData->totalNum;

    // 本核负责的元素区间 [start, start + blockLength_)，最后一个核可能不满
    const int64_t start = blockIdx * blockFactor;
    const int64_t remain = totalNum - start;
    if (remain <= 0) {
        blockLength_ = 0;
    } else {
        blockLength_ = (remain > blockFactor) ? blockFactor : remain;
    }

    // 每次 UB 循环处理的元素数，由 host 侧 tiling 给出
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = blockLength_;
    }
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    // 按核号平移 GM 指针，切出本核可见的数据视图
    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, blockLength_);

    // 输入、输出各开 BUFFER_NUM 块 UB，单块 ubLength_ 个元素
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // 注意：此处必须写 AscendC::Tanh，否则 Tanh 会被解析成本类的注入类名
    AscendC::Tanh(yLocal, xLocal, static_cast<uint32_t>(currentNum));

    outputQueueY.EnQue<T>(yLocal);
    // 输入 buffer 必须先释放，否则下一轮 AllocTensor 拿不到 UB 空间
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    int64_t offset = 0;
    while (offset < blockLength_) {
        // 最后一块可能不满，按实际剩余长度处理
        int64_t currentNum = blockLength_ - offset;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
        offset += currentNum;
    }
}

} // namespace NsTanh
#endif // TANH_H