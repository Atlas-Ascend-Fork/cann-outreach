/*!
 * \file relu.h
 * \brief 匹配 CANNJudge 2980 空工程的双缓冲 Relu
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
    __aicore__ inline Relu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void ProcessSmall();
    __aicore__ inline void CopyInFull(int64_t progress);
    __aicore__ inline void CopyOutFull(int64_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> smallInput;
    TBuf<QuePosition::VECCALC> smallOutput;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    bool smallPath_ = false;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    if (tilingData->totalNum <= 0 || tilingData->blockFactor <= 0 ||
        tilingData->ubFactor <= 0) {
        return;
    }
    const int64_t offset =
        static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->blockFactor;
    if (offset >= tilingData->totalNum) {
        return;
    }
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;

    // 兼容模板 CPU UT 中 ubFactor=16384 的 float 用例，限制单块 UB 用量。
    const int64_t maxTileElements = (46 * 1024) / sizeof(T);
    const int64_t alignElements = 32 / sizeof(T);
    ubLength_ = tilingData->ubFactor < maxTileElements ? tilingData->ubFactor : maxTileElements;
    ubLength_ = (ubLength_ + alignElements - 1) / alignElements * alignElements;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    smallPath_ = tilingData->totalNum <= 16384 &&
                 blockLength_ <= ubLength_ && blockLength_ % alignElements == 0;
    if (smallPath_) {
        // 单块输入无需排队或预取，用显式流水事件建立依赖。
        pipe.InitBuffer(smallInput, bufferBytes);
        pipe.InitBuffer(smallOutput, bufferBytes);
    } else {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
    }
}

template <typename T>
__aicore__ inline void Relu<T>::ProcessSmall()
{
    LocalTensor<T> xLocal = smallInput.Get<T>();
    LocalTensor<T> yLocal = smallOutput.Get<T>();
    const uint32_t count = static_cast<uint32_t>(blockLength_);
    const event_t loadEvent = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    const event_t computeEvent = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    const event_t storeEvent = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));

    DataCopy(xLocal, inputGMX, count);
    SetFlag<HardEvent::MTE2_V>(loadEvent);
    WaitFlag<HardEvent::MTE2_V>(loadEvent);
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(count));
    SetFlag<HardEvent::V_MTE3>(computeEvent);
    WaitFlag<HardEvent::V_MTE3>(computeEvent);
    DataCopy(outputGMY, yLocal, count);
    SetFlag<HardEvent::MTE3_S>(storeEvent);
    WaitFlag<HardEvent::MTE3_S>(storeEvent);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    if (currentNum % (32 / sizeof(T)) == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        // 非对齐尾块按真实字节数搬入，避免读取输入之外的数据。
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad<T>(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    if (currentNum % (32 / sizeof(T)) == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad<T>(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyInFull(int64_t progress)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(ubLength_));
    inputQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOutFull(int64_t progress)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(ubLength_));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    if (smallPath_) {
        ProcessSmall();
        return;
    }
    // 主体和尾块分别处理，主体循环不再反复计算实际长度与对齐分支。
    const int64_t fullCount = blockLength_ / ubLength_;
    const int64_t tailNum = blockLength_ % ubLength_;
    if (fullCount > 0) {
        CopyInFull(0);
        for (int64_t i = 0; i < fullCount; ++i) {
            if (i + 1 < fullCount) {
                CopyInFull(i + 1);
            } else if (tailNum > 0) {
                CopyIn(fullCount, tailNum);
            }
            Compute(ubLength_);
            CopyOutFull(i);
        }
    }
    if (tailNum > 0) {
        if (fullCount == 0) {
            CopyIn(0, tailNum);
        }
        Compute(tailNum);
        CopyOut(fullCount, tailNum);
    }
}

}  // namespace NsRelu
#endif  // RELU_H
