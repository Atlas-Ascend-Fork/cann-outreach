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
    // 多核切分：第 blockIdx 个核处理 [blockIdx*blockFactor, ...) 区间
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t startOffset = blockIdx * tilingData->blockFactor;
    // 最后一个核可能不足 blockFactor 个元素，收敛成本核实际长度
    int64_t remain = tilingData->totalNum - startOffset;
    blockLength_ = (remain < tilingData->blockFactor) ? remain : tilingData->blockFactor;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    // 单次处理长度；单核数据量不足一块时直接收敛，省掉循环
    ubLength_ = tilingData->ubFactor;
    if (blockLength_ > 0 && ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    // 每个核的 Global Memory 起始地址（输入输出偏移一致）
    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    // UB 分配：输入/输出队列各 BUFFER_NUM 块，每块 ubLength_ 个元素
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    // tiling 保证每块长度都是 32B 整数倍，这里统一走 DataCopy
    DataCopy(xLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // y = max(x, 0)：向量-标量指令，一条指令完成整块计算
    AscendC::Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<uint32_t>(currentNum));

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    // 单核内按 ubLength_ 分块；BUFFER_NUM=2 形成 DoubleBuffer 流水
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t progress = i * ubLength_;
        int64_t currentNum = blockLength_ - progress;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}
} // namespace NsRelu
#endif // RELU_H
