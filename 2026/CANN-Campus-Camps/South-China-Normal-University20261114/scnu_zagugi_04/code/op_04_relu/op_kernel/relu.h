#ifndef CUSTOM_RELU_KERNEL_H
#define CUSTOM_RELU_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData)
    {
        blockLength =
            tilingData->totalLength /
            AscendC::GetBlockNum();

        tileNum = tilingData->tileNum;
        tileLength = blockLength / tileNum;

        const uint32_t offset =
            AscendC::GetBlockIdx() * blockLength;

        inputGMX.SetGlobalBuffer(
            (__gm__ DT_X*)x + offset,
            blockLength);

        outputGMY.SetGlobalBuffer(
            (__gm__ DT_X*)y + offset,
            blockLength);

        // 小输入单缓冲，大输入双缓冲。
        const uint32_t bufferCount =
            tileNum > 1U ? 2U : 1U;

        pipe.InitBuffer(
            inputQueueX,
            bufferCount,
            tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            outputQueueY,
            bufferCount,
            tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 提前搬入第一块。
        CopyIn(0);

        for (uint32_t i = 0; i < tileNum; ++i) {
            // 预取下一块，与当前块计算及输出配合。
            if (i + 1U < tileNum) {
                CopyIn(i + 1U);
            }

            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inputQueueX.AllocTensor<DT_X>();

        AscendC::DataCopy(
            xLocal,
            inputGMX[progress * tileLength],
            tileLength);

        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inputQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            outputQueueY.AllocTensor<DT_X>();

        // y = max(0, x)
        AscendC::Relu(
            yLocal,
            xLocal,
            tileLength);

        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> yLocal =
            outputQueueY.DeQue<DT_X>();

        AscendC::DataCopy(
            outputGMY[progress * tileLength],
            yLocal,
            tileLength);

        outputQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, 2>
        inputQueueX;

    AscendC::TQue<AscendC::QuePosition::VECOUT, 2>
        outputQueueY;

    AscendC::GlobalTensor<DT_X> inputGMX;
    AscendC::GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

}  // namespace NsRelu

#endif