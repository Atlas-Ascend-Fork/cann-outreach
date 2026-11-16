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

// 输入/输出队列各开 2 块，构成 DoubleBuffer：
// 多 tile 场景下，下一块的 CopyIn 可以和上一块的 Compute/CopyOut 流水重叠
constexpr int32_t BUFFER_NUM = 2;
// 单块 UB buffer 的字节数上限，防止 Tiling 传入过大的 ubFactor 导致 UB 申请越界
constexpr int64_t MAX_UB_BYTES = 32 * 1024;

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
    const int64_t totalNum = tilingData->totalNum;
    // 防御：blockFactor 非法时退回“单核处理全部数据”，避免漏算
    int64_t blockFactor = tilingData->blockFactor;
    if (blockFactor <= 0) {
        blockFactor = (totalNum > 0) ? totalNum : 1;
    }
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t offset = blockIdx * blockFactor;

    // 本核负责的数据长度；最后一个核可能不足 blockFactor，需要截断防止越界
    if (offset >= totalNum) {
        blockLength_ = 0;
        return;
    }
    blockLength_ = totalNum - offset;
    if (blockLength_ > blockFactor) {
        blockLength_ = blockFactor;
    }

    // 单次 UB 循环长度不能超过本核数据量，同时受 UB 容量（按字节）上限约束
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0 || ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }
    const int64_t maxUbLength = MAX_UB_BYTES / static_cast<int64_t>(sizeof(T));
    if (ubLength_ > maxUbLength) {
        ubLength_ = maxUbLength;
    }

    // 只把本核负责的数据段绑定到 GlobalTensor，后续按 tile 偏移访问
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);

    // 队列 buffer 大小按 32B 向上对齐，避免非对齐长度造成越界
    const uint32_t bufferBytes =
        static_cast<uint32_t>((ubLength_ * static_cast<int64_t>(sizeof(T)) + 31) / 32 * 32);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    const int64_t byteNum = currentNum * static_cast<int64_t>(sizeof(T));

    if ((byteNum & 31) == 0) {
        // 32B 对齐时使用 DataCopy，搬运效率更高
        AscendC::DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        // 尾块不足 32B 对齐时使用 DataCopyPad，由硬件补齐，避免越界读写
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(byteNum), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // ReLU：y = max(x, 0)，一条向量 max 指令即可完成，无需升精度也不会损失精度
    AscendC::Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    const int64_t byteNum = currentNum * static_cast<int64_t>(sizeof(T));

    if ((byteNum & 31) == 0) {
        AscendC::DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(byteNum), 0, 0, 0};
        AscendC::DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    const int64_t loopCount = blockLength_ / ubLength_;
    const int64_t tailNum = blockLength_ - loopCount * ubLength_;

    // 整块循环：搬运、计算、写回借助 DoubleBuffer 流水并行
    for (int64_t i = 0; i < loopCount; i++) {
        CopyIn(i, ubLength_);
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }

    // 尾块处理
    if (tailNum > 0) {
        CopyIn(loopCount, tailNum);
        Compute(tailNum);
        CopyOut(loopCount, tailNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
