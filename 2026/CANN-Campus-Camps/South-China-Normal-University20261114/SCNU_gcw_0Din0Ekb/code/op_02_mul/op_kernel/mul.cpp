// Kernel side implementation of Mul (z = x * y)
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 1;           // one buffer per queue: single tile per core
constexpr uint32_t MAX_TILE_LENGTH = 2048;  // max elements handled in one tile

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        this->totalLength = length;
        // 32B alignment: 8 elements for float32, 16 elements for float16
        uint32_t alignElem = 32 / sizeof(DT_X);
        uint32_t blockNum = AscendC::GetBlockNum();
        // elements per core, rounded up to 32B alignment
        uint32_t avgLength = (this->totalLength + blockNum - 1) / blockNum;
        this->blockLength = (avgLength + alignElem - 1) / alignElem * alignElem;
        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        if (offset >= this->totalLength) {
            // no data left for this core
            this->coreLength = 0;
            return;
        }
        this->coreLength = this->totalLength - offset;
        if (this->coreLength > this->blockLength) {
            this->coreLength = this->blockLength;
        }
        this->tileLength = (this->coreLength < MAX_TILE_LENGTH) ? this->coreLength : MAX_TILE_LENGTH;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, this->coreLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, this->coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }
        uint32_t loopCount = (this->coreLength + this->tileLength - 1) / this->tileLength;
        for (uint32_t i = 0; i < loopCount; ++i) {
            CopyIn(static_cast<int32_t>(i));
            Compute(static_cast<int32_t>(i));
            CopyOut(static_cast<int32_t>(i));
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        uint32_t offset = progress * this->tileLength;
        uint32_t curLength = this->coreLength - offset;
        if (curLength > this->tileLength) {
            curLength = this->tileLength;
        }
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], curLength);
        AscendC::DataCopy(yLocal, yGm[offset], curLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        uint32_t offset = progress * this->tileLength;
        uint32_t curLength = this->coreLength - offset;
        if (curLength > this->tileLength) {
            curLength = this->tileLength;
        }
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, curLength);
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        uint32_t offset = progress * this->tileLength;
        uint32_t curLength = this->coreLength - offset;
        if (curLength > this->tileLength) {
            curLength = this->tileLength;
        }
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, curLength);
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
    uint32_t totalLength;  // total elements of the tensor
    uint32_t blockLength;  // elements per core (32B aligned)
    uint32_t coreLength;   // real elements handled by this core
    uint32_t tileLength;   // elements handled per tile
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
