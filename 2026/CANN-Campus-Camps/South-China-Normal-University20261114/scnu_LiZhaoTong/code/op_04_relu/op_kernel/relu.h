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

constexpr int32_t BUFFER_NUM = 2;        // 输入 / 输出队列双缓冲
constexpr int64_t MAX_UB_ELEMS = 4096;  // 单次 UB 循环元素上限

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        // 本核负责的数据区间
        this->blockLength_ = tilingData->blockFactor;
        this->blockOffset_ = static_cast<int64_t>(AscendC::GetBlockIdx()) * this->blockLength_;
        int64_t totalNum = tilingData->totalNum;
        if (this->blockOffset_ >= totalNum) {
            this->currentTotal_ = 0;
        } else {
            int64_t remain = totalNum - this->blockOffset_;
            this->currentTotal_ = (remain < this->blockLength_) ? remain : this->blockLength_;
        }

        // 单次 UB 循环元素数，取 tiling 值与内部上限的较小者并保证 32 字节对齐
        this->ubLength_ = tilingData->ubFactor;
        if (this->ubLength_ <= 0 || this->ubLength_ > MAX_UB_ELEMS) {
            this->ubLength_ = MAX_UB_ELEMS;
        }
        this->alignNum_ = 32 / static_cast<int64_t>(sizeof(T));
        if (this->alignNum_ < 1) {
            this->alignNum_ = 1;
        }
        this->ubLength_ = FloorAlignLocal(this->ubLength_, this->alignNum_);
        if (this->ubLength_ < this->alignNum_) {
            this->ubLength_ = this->alignNum_;
        }

        // 每个核的 Global Memory 起始地址按 blockOffset_ 偏移
        inputGMX.SetGlobalBuffer((__gm__ T*)x + this->blockOffset_, this->currentTotal_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + this->blockOffset_, this->currentTotal_);

        // 输入 / 输出队列各 2 块，构成 DoubleBuffer
        uint32_t bufBytes = static_cast<uint32_t>(this->ubLength_ * static_cast<int64_t>(sizeof(T)));
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufBytes);
    }

    __aicore__ inline void Process()
    {
        if (this->currentTotal_ <= 0) {
            return;
        }
        int64_t loopCount = (this->currentTotal_ + this->ubLength_ - 1) / this->ubLength_;
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t offset = i * this->ubLength_;
            int64_t currentNum = this->currentTotal_ - offset;
            if (currentNum > this->ubLength_) {
                currentNum = this->ubLength_;
            }
            CopyIn(offset, currentNum);
            Compute(currentNum);
            CopyOut(offset, currentNum);
        }
    }

private:
    __aicore__ inline int64_t FloorAlignLocal(int64_t value, int64_t align)
    {
        return (value / align) * align;
    }

    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum)
    {
        LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, inputGMX[progress], static_cast<int32_t>(currentNum));
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
        // y = max(0, x)，使用专用 Relu 指令，单条向量指令完成
        AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
        outputQueueY.EnQue<T>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
    {
        LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
        AscendC::DataCopy(outputGMY[progress], yLocal, static_cast<int32_t>(currentNum));
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t alignNum_ = 1;
    int64_t blockOffset_ = 0;
    int64_t currentTotal_ = 0;
};

} // namespace NsRelu
#endif // RELU_H
