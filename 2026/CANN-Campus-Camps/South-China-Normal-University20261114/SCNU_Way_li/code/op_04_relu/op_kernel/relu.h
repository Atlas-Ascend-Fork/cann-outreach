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

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData)
    {
        // 每核处理的元素数。
        this->blockLength =
            tilingData->totalLength / AscendC::GetBlockNum();

        this->tileNum = tilingData->tileNum;

        // 每次搬运、计算的元素数。
        this->tileLength =
            this->blockLength / this->tileNum / BUFFER_NUM;

        // 每个核对应一段连续 GM。
        // 此处的指针偏移以 DT_X 元素为单位。
        const uint32_t offset =
            this->blockLength * AscendC::GetBlockIdx();

        inputGMX.SetGlobalBuffer(
            (__gm__ DT_X*)x + offset,
            this->blockLength);

        outputGMY.SetGlobalBuffer(
            (__gm__ DT_X*)y + offset,
            this->blockLength);

        // 输入队列双缓冲。
        pipe.InitBuffer(
            inputQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));

        // 输出队列双缓冲。
        pipe.InitBuffer(
            outputQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        const int32_t loopCount =
            this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.AllocTensor<DT_X>();

        DataCopy(
            xLocal,
            inputGMX[progress * this->tileLength],
            this->tileLength);

        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            outputQueueY.AllocTensor<DT_X>();

        // ReLU: y = max(0, x)。
        // 使用完整名称，避免与当前类名 Relu 冲突。
        AscendC::Relu(yLocal, xLocal, this->tileLength);

        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal =
            outputQueueY.DeQue<DT_X>();

        DataCopy(
            outputGMY[progress * this->tileLength],
            yLocal,
            this->tileLength);

        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

} // namespace NsRelu

#endif // RELU_H