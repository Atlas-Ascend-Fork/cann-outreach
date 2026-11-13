// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length, uint32_t tileNum)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        this->tileNum = (tileNum == 0) ? 1 : tileNum;
        // 多核切分：向上取整，最后一个核处理剩余元素
        this->blockLength = (length + blockNum - 1) / blockNum;
        const uint32_t start = blockIdx * this->blockLength;
        this->localLength = (start >= length) ? 0 : (length - start);
        if (this->localLength > this->blockLength) {
            this->localLength = this->blockLength;
        }
        // 单核内按 tileNum 分块，并限制单次 UB 处理量
        this->tileLength = (this->localLength + this->tileNum - 1) / this->tileNum;
        constexpr uint32_t MAX_TILE_LENGTH = 4096;
        if (this->tileLength > MAX_TILE_LENGTH) {
            this->tileLength = MAX_TILE_LENGTH;
        }
        if (this->tileLength == 0) {
            this->tileLength = 1;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + start, this->localLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + start, this->localLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + start, this->localLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->localLength == 0 || this->tileLength == 0) {
            return;
        }
        const uint32_t loopCount = (this->localLength + this->tileLength - 1) / this->tileLength;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t currentNum = this->tileLength;
            if (i == loopCount - 1) {
                currentNum = this->localLength - i * this->tileLength;
            }
            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t currentNum)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams;
        AscendC::DataCopyPad(xLocal, xGm[progress * this->tileLength], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[progress * this->tileLength], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t currentNum)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, currentNum);
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t currentNum)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPad(zGm[progress * this->tileLength], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength = 0;
    uint32_t localLength = 0;
    uint32_t tileNum = 1;
    uint32_t tileLength = 1;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length, tiling_data.tileNum);
    op.Process();
}
