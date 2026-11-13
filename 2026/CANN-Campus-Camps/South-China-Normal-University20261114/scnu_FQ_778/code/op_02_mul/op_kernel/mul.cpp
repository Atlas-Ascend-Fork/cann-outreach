#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t UB_ALIGN_BYTES = 32;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData &tiling)
    {
        this->totalLength = tiling.totalLength;
        this->tileLength = tiling.tileLength;
        this->blockNum = AscendC::GetBlockNum();
        this->blockIdx = AscendC::GetBlockIdx();

        int64_t base = this->totalLength / this->blockNum;
        int64_t rem = this->totalLength % this->blockNum;
        this->ubLength = base + ((this->blockIdx < rem) ? 1 : 0);
        int64_t startOffset = this->blockIdx * base
            + ((this->blockIdx < rem) ? this->blockIdx : rem);

        if (this->tileLength <= 0) {
            this->tileLength = 1;
        }
        if (this->ubLength > 0 && this->tileLength > this->ubLength) {
            this->tileLength = this->ubLength;
        }

        int64_t bufStart = startOffset;
        if (bufStart >= this->totalLength) {
            bufStart = (this->totalLength > 0) ? (this->totalLength - 1) : 0;
        }
        int64_t bufLength = this->ubLength > 0 ? this->ubLength : 1;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + bufStart, bufLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + bufStart, bufLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + bufStart, bufLength);

        int64_t bufBytes = this->tileLength * static_cast<int64_t>(sizeof(DT_X));
        bufBytes = (bufBytes + UB_ALIGN_BYTES - 1) / UB_ALIGN_BYTES * UB_ALIGN_BYTES;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(inQueueY, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, bufBytes);
    }

    __aicore__ inline void Process()
    {
        for (int64_t progress = 0; progress < this->ubLength; progress += this->tileLength) {
            int64_t count = this->ubLength - progress;
            if (count > this->tileLength) {
                count = this->tileLength;
            }
            CopyIn(progress, count);
            Compute(count);
            CopyOut(progress, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t count)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(DT_X)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = static_cast<DT_X>(0);
        AscendC::DataCopyPad(xLocal, xGm[progress], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[progress], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, static_cast<uint32_t>(count));
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t count)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(DT_X)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(zGm[progress], zLocal, copyParams);
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

    int64_t totalLength;
    int64_t blockNum;
    int64_t blockIdx;
    int64_t ubLength;
    int64_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}


