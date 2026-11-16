// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        blockLength_ = length / AscendC::GetBlockNum();
        tileLength_ = blockLength_ / BUFFER_NUM;
        const uint32_t blockOffset = AscendC::GetBlockIdx() * blockLength_;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + blockOffset, blockLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + blockOffset, blockLength_);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z) + blockOffset, blockLength_);

        pipe_.InitBuffer(xQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(zQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        for (uint32_t progress = 0; progress < BUFFER_NUM; ++progress) {
            CopyIn(progress);
            Compute();
            CopyOut(progress);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t progress) {
        AscendC::LocalTensor<DT_X> xLocal = xQueue_.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yQueue_.AllocTensor<DT_X>();
        const uint32_t offset = progress * tileLength_;
        AscendC::DataCopy(xLocal, xGm_[offset], tileLength_);
        AscendC::DataCopy(yLocal, yGm_[offset], tileLength_);
        xQueue_.EnQue(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_X> xLocal = xQueue_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yQueue_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = zQueue_.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, tileLength_);
        zQueue_.EnQue(zLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        AscendC::LocalTensor<DT_X> zLocal = zQueue_.DeQue<DT_X>();
        AscendC::DataCopy(zGm_[progress * tileLength_], zLocal, tileLength_);
        zQueue_.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> yQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> zQueue_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::GlobalTensor<DT_X> zGm_;
    uint32_t blockLength_;
    uint32_t tileLength_;

};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
