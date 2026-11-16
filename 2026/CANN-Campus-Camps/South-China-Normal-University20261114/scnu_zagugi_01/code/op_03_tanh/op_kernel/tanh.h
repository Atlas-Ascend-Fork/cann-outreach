#ifndef CUSTOM_TANH_KERNEL_H
#define CUSTOM_TANH_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/math/tanh.h"

#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const TanhTilingData* tilingData)
    {
        blockLength =
            tilingData->totalLength / AscendC::GetBlockNum();

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

        pipe.InitBuffer(
            inputQueueX,
            1,
            tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            outputQueueY,
            1,
            tileLength * sizeof(DT_X));

        // ascend910b 上 Tanh 使用 float 临时数据：
        // float 输入需要一份，half 输入需要两份。
        constexpr uint32_t TMP_FLOAT_BUFFERS =
            sizeof(DT_X) == sizeof(half) ? 2U : 1U;

        pipe.InitBuffer(
            tmpBuffer,
            tileLength * sizeof(float) * TMP_FLOAT_BUFFERS);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; ++i) {
            CopyIn(i);
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

        AscendC::LocalTensor<uint8_t> tmpLocal =
            tmpBuffer.Get<uint8_t>();

        // 使用库接口计算，half 输入的中间计算使用 float。
        AscendC::Tanh<DT_X, false>(
            yLocal,
            xLocal,
            tmpLocal,
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

    AscendC::TQue<AscendC::QuePosition::VECIN, 1>
        inputQueueX;

    AscendC::TQue<AscendC::QuePosition::VECOUT, 1>
        outputQueueY;

    AscendC::TBuf<AscendC::TPosition::VECCALC>
        tmpBuffer;

    AscendC::GlobalTensor<DT_X> inputGMX;
    AscendC::GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

}  // namespace NsTanh

#endif  // CUSTOM_TANH_KERNEL_H