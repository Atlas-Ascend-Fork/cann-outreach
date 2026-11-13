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
constexpr uint32_t TILE_LENGTH = 1024;
constexpr uint32_t ALIGN_BYTES = 32;

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        // 将总数据量均匀分配给各个计算核
        uint32_t baseLength = tilingData->totalLength / blockNum;
        uint32_t remainder = tilingData->totalLength % blockNum;

        this->blockLength =
            baseLength + (blockIdx < remainder ? 1 : 0);

        // 当前计算核在原始 Tensor 中的起始位置
        uint32_t blockOffset =
            blockIdx * baseLength +
            (blockIdx < remainder ? blockIdx : remainder);

        this->tileNum = tilingData->tileNum;
        this->tileLength = TILE_LENGTH;

        // 设置 Global Memory
        inputGMX.SetGlobalBuffer(
            (__gm__ DT_X*)x + blockOffset,
            this->blockLength
        );

        outputGMY.SetGlobalBuffer(
            (__gm__ DT_X*)y + blockOffset,
            this->blockLength
        );

        // 固定大小分配双缓冲
        pipe.InitBuffer(
            inputQueueX,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(DT_X)
        );

        pipe.InitBuffer(
            outputQueueY,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(DT_X)
        );
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->tileNum * BUFFER_NUM;

        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t offset = i * this->tileLength;

            if (offset >= this->blockLength) {
                break;
            }

            // 处理最后一个不完整的数据块
            uint32_t processLength =
                this->blockLength - offset;

            if (processLength > this->tileLength) {
                processLength = this->tileLength;
            }

            CopyIn(offset, processLength);
            Compute(processLength);
            CopyOut(offset, processLength);
        }
    }

private:
    // 从 Global Memory 搬运输入
    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t processLength)
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.AllocTensor<DT_X>();

        // 将尾块长度补齐到 32 字节
        uint32_t alignElements = ALIGN_BYTES / sizeof(DT_X);

        uint32_t paddedLength =
            ((processLength + alignElements - 1) /
             alignElements) * alignElements;

        DataCopyExtParams copyParams = {
            1,
            static_cast<uint32_t>(processLength * sizeof(DT_X)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<DT_X> padParams = {
            true,
            0,
            static_cast<uint8_t>(paddedLength - processLength),
            static_cast<DT_X>(0)
        };

        DataCopyPad(
            xLocal,
            inputGMX[offset],
            copyParams,
            padParams
        );

        inputQueueX.EnQue(xLocal);
    }

    // 执行 ReLU
    __aicore__ inline void Compute(uint32_t processLength)
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            outputQueueY.AllocTensor<DT_X>();

        AscendC::Relu(
            yLocal,
            xLocal,
            static_cast<int32_t>(processLength)
        );

        outputQueueY.EnQue<DT_X>(yLocal);

        inputQueueX.FreeTensor(xLocal);
    }

    // 将输出写回 Global Memory
    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t processLength)
    {
        LocalTensor<DT_X> yLocal =
            outputQueueY.DeQue<DT_X>();

        DataCopyExtParams copyParams = {
            1,
            static_cast<uint32_t>(processLength * sizeof(DT_X)),
            0,
            0,
            0
        };

        DataCopyPad(
            outputGMY[offset],
            yLocal,
            copyParams
        );

        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
};

} // namespace NsRelu

#endif // RELU_H