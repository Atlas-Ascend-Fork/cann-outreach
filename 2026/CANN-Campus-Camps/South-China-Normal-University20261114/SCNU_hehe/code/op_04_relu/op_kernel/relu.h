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

// 单次搬入 UB 的元素数上限。8192 个元素在 fp32 下每队列 32KB，
// in/out 各双缓冲合计 128KB，稳落在 256KB UB 预算内。
// 有它兜底，host 侧给出的 ubFactor 无论多大都不会让 InitBuffer 越界。
constexpr int64_t MAX_UB_ELEMS = 8192;

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

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 多核切分：第 blockIdx 个核从 blockIdx * blockFactor 处开始
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t start = blockIdx * tilingData->blockFactor;
    int64_t remain = tilingData->totalNum - start;

    // 尾核可能拿不满，甚至无数据可算（块数少于核数时）
    if (remain <= 0) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }
    blockLength_ = (remain < tilingData->blockFactor) ? remain : tilingData->blockFactor;

    // 单次处理的元素数不超过本核总量，也不超过 UB 上限
    int64_t ub = (tilingData->ubFactor > 0) ? tilingData->ubFactor : blockLength_;
    if (ub > blockLength_) {
        ub = blockLength_;
    }
    if (ub > MAX_UB_ELEMS) {
        ub = MAX_UB_ELEMS;
    }
    ubLength_ = ub;

    // GlobalBuffer 挂到本核负责的切片上，后续偏移都相对本核起点
    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // relu(x) = max(x, 0)，一条矢量指令完成，无需 float 中转
    AscendC::Maxs(yLocal, xLocal, static_cast<T>(0.0f), currentNum);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopy(outputGMY[progress], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // 空核直接返回，避免 0 长度搬运
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t offset = i * ubLength_;
        int64_t currentNum = (i + 1 == loopCount) ? (blockLength_ - offset) : ubLength_;
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
