#include "kernel_operator.h"
#include "div_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename xType, typename yType, typename zType>
class KernelDivCustomTemplate {
public:
    __aicore__ inline KernelDivCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t blockLength, uint32_t tileLength,
        uint32_t loopCount)
    {
        this->blockLength = blockLength;
        this->tileLength = tileLength;
        this->loopCount = loopCount;

        const uint32_t blockOffset = this->blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ xType *)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ yType *)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ zType *)z + blockOffset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(xType));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(yType));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(zType));
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < static_cast<int32_t>(loopCount); i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<xType> xLocal = inQueueX.AllocTensor<xType>();
        AscendC::LocalTensor<yType> yLocal = inQueueY.AllocTensor<yType>();
        const uint32_t offset = progress * tileLength;
        AscendC::DataCopy(xLocal, xGm[offset], tileLength);
        AscendC::DataCopy(yLocal, yGm[offset], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<xType> xLocal = inQueueX.DeQue<xType>();
        AscendC::LocalTensor<yType> yLocal = inQueueY.DeQue<yType>();
        AscendC::LocalTensor<zType> zLocal = outQueueZ.AllocTensor<zType>();
        AscendC::Div(zLocal, xLocal, yLocal, tileLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<zType> zLocal = outQueueZ.DeQue<zType>();
        AscendC::DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<xType> xGm;
    AscendC::GlobalTensor<yType> yGm;
    AscendC::GlobalTensor<zType> zGm;
    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    uint32_t loopCount = 0;
};

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(DivCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelDivCustomTemplate<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.blockLength, tilingData.tileLength, tilingData.loopCount);
    op.Process();
}
