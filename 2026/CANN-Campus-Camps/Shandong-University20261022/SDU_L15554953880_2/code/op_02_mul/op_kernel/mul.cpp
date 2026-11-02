// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
        uint32_t length, uint32_t blockLength, uint32_t tileLength) {
        tileLength_ = tileLength;
        uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * blockLength;
        coreLength_ = start < length ? static_cast<uint32_t>(length - start) : 0;
        if (coreLength_ > blockLength) coreLength_ = blockLength;
        if (coreLength_ == 0 || tileLength_ == 0) return;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + start, coreLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + start, coreLength_);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(z) + start, coreLength_);
        pipe_.InitBuffer(xQueue_, 2, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(yQueue_, 2, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(zQueue_, 2, tileLength_ * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (coreLength_ == 0 || tileLength_ == 0) return;
        for (uint32_t offset = 0; offset < coreLength_;) {
            uint32_t count = coreLength_ - offset;
            if (count > tileLength_) count = tileLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        LocalTensor<DT_X> xLocal = xQueue_.template AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = yQueue_.template AllocTensor<DT_X>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> pad{};
        DataCopyPad(xLocal, xGm_[offset], copy, pad);
        DataCopyPad(yLocal, yGm_[offset], copy, pad);
        xQueue_.template EnQue<DT_X>(xLocal);
        yQueue_.template EnQue<DT_X>(yLocal);
    }
    __aicore__ inline void Compute(uint32_t count) {
        LocalTensor<DT_X> xLocal = xQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = yQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = zQueue_.template AllocTensor<DT_X>();
        Mul(zLocal, xLocal, yLocal, count);
        zQueue_.template EnQue<DT_X>(zLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        LocalTensor<DT_X> zLocal = zQueue_.template DeQue<DT_X>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(zGm_[offset], zLocal, copy);
        zQueue_.FreeTensor(zLocal);
    }
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> xQueue_, yQueue_;
    TQue<QuePosition::VECOUT, 2> zQueue_;
    GlobalTensor<DT_X> xGm_, yGm_, zGm_;
    uint32_t coreLength_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}
