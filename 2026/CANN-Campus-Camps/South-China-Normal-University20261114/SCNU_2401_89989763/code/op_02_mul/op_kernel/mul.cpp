// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(DT_X);

        uint32_t coreNum = GetBlockNum();
        if (coreNum == 0) { coreNum = 1; }

        uint32_t perCore = (totalLength + coreNum - 1) / coreNum;
        perCore = (perCore + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;

        uint32_t start = GetBlockIdx() * perCore;
        if (start >= totalLength) {
            this->blockLength = 0;
            this->tileLength = 0;
            this->tileNum = 0;
            return;
        }
        this->blockLength = (start + perCore <= totalLength) ? perCore : (totalLength - start);

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + start, this->blockLength);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + start, this->blockLength);
        zGm_.SetGlobalBuffer((__gm__ DT_X *)z + start, this->blockLength);

        this->tileLength = 8192 / sizeof(DT_X);
        if (this->tileLength > this->blockLength) { this->tileLength = this->blockLength; }
        this->tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;

        this->pipe_.InitBuffer(inQueueX_, 2, this->tileLength * sizeof(DT_X));
        this->pipe_.InitBuffer(inQueueY_, 2, this->tileLength * sizeof(DT_X));
        this->pipe_.InitBuffer(outQueueZ_, 2, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        for (int32_t i = 0; i < static_cast<int32_t>(this->tileNum); i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        uint32_t off = static_cast<uint32_t>(progress) * this->tileLength;
        uint32_t len = this->blockLength - off;
        if (len > this->tileLength) { len = this->tileLength; }

        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY_.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm_[off], len);
        DataCopy(yLocal, yGm_[off], len);
        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        uint32_t off = static_cast<uint32_t>(progress) * this->tileLength;
        uint32_t len = this->blockLength - off;
        if (len > this->tileLength) { len = this->tileLength; }

        LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY_.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ_.AllocTensor<DT_X>();
        Mul(zLocal, xLocal, yLocal, static_cast<int32_t>(len));
        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        uint32_t off = static_cast<uint32_t>(progress) * this->tileLength;
        uint32_t len = this->blockLength - off;
        if (len > this->tileLength) { len = this->tileLength; }

        LocalTensor<DT_X> zLocal = outQueueZ_.DeQue<DT_X>();
        DataCopy(zGm_[off], zLocal, len);
        outQueueZ_.FreeTensor(zLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inQueueX_;
    TQue<QuePosition::VECIN, 2> inQueueY_;
    TQue<QuePosition::VECOUT, 2> outQueueZ_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    GlobalTensor<DT_X> zGm_;
    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    uint32_t tileNum = 0;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}