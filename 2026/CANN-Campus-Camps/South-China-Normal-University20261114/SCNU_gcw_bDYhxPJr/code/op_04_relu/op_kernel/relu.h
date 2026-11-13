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
constexpr int64_t ALIGN_BYTES = 32; // DataCopy 单次搬运要求按 32B 对齐
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
    // 最后一个核可能不足 blockFactor 个元素，这里收敛成"本核实际长度"
    int64_t remain = tilingData->totalNum - startOffset;
    blockLength_ = (remain < tilingData->blockFactor) ? remain : tilingData->blockFactor;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    // 单次处理长度；单核数据比一次搬运量还少时直接收敛，省掉循环
    ubLength_ = tilingData->ubFactor;
    if (blockLength_ > 0 && ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    // 每个核的 Global Memory 起始地址（输入输出偏移一致：逐元素映射）
    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    // UB 分配：输入/输出队列各 BUFFER_NUM 块，每块 ubLength_ 个元素
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    if (currentNum % (ALIGN_BYTES / static_cast<int64_t>(sizeof(T))) == 0) {
        // 对齐场景走普通 DataCopy，带宽利用率最高
        DataCopy(xLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    } else {
        // 尾块非 32B 对齐时用 DataCopyPad 按字节搬运，避免越界读
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = static_cast<T>(0);
        DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // y = max(x, 0)：向量-标量指令，一条指令完成整块计算
    AscendC::Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    if (currentNum % (ALIGN_BYTES / static_cast<int64_t>(sizeof(T))) == 0) {
        DataCopy(outputGMY[progress], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
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
    // 单核内按 ubLength_ 分块，BUFFER_NUM=2 使相邻块之间形成 DoubleBuffer 流水
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t offset = i * ubLength_;
        int64_t currentNum = blockLength_ - offset;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}
} // namespace NsRelu
#endif // RELU_H
