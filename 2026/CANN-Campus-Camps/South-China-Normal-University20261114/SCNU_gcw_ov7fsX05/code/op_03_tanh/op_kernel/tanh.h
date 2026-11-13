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

// Init：绑定 GM 张量、按 blockIdx 切分本核数据、初始化 UB 队列
//   host 侧保证 blockFactor / ubFactor 均为 32B 对齐，故每核起始地址与每块长度天然对齐
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t totalNum = tilingData->totalNum;
    const int64_t offset = blockFactor * blockIdx;

    // 本核处理长度：尾核只取剩余部分，避免越界
    const int64_t remain = totalNum - offset;
    blockLength_ = (remain > blockFactor) ? blockFactor : remain;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 32 / static_cast<int64_t>(sizeof(T));
    }

    const int64_t gmLength = (blockLength_ > 0) ? blockLength_ : 1;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, gmLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, gmLength);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T)));
}

// CopyIn：GM -> UB，按元素个数搬运（保证 32B 对齐）
template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

// Compute：y = tanh(x)
//   三参数接口由框架内部申请临时空间，host 侧已按 UB_SPLIT_NUM 预留余量
template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    AscendC::Tanh(yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// CopyOut：UB -> GM，只回写有效元素
template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

// Process：按 ubLength_ 切块，CopyIn -> Compute -> CopyOut 流水
template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        const int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
