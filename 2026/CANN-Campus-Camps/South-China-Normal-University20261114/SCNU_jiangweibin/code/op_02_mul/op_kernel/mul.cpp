// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

namespace {
constexpr uint32_t TILE_LENGTH = 256;
constexpr uint32_t QUEUE_DEPTH = 2;
}

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length)
    {
        blockLength_ = length / AscendC::GetBlockNum();
        const uint32_t offset = blockLength_ * AscendC::GetBlockIdx();
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + offset, blockLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + offset, blockLength_);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(z) + offset, blockLength_);
        pipe_.InitBuffer(xQueue_, QUEUE_DEPTH, TILE_LENGTH * sizeof(DT_X));
        pipe_.InitBuffer(yQueue_, QUEUE_DEPTH, TILE_LENGTH * sizeof(DT_X));
        pipe_.InitBuffer(zQueue_, QUEUE_DEPTH, TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength_; offset += TILE_LENGTH) {
            auto x = xQueue_.AllocTensor<DT_X>();
            auto y = yQueue_.AllocTensor<DT_X>();
            AscendC::DataCopy(x, xGm_[offset], TILE_LENGTH);
            AscendC::DataCopy(y, yGm_[offset], TILE_LENGTH);
            xQueue_.EnQue(x);
            yQueue_.EnQue(y);

            x = xQueue_.DeQue<DT_X>();
            y = yQueue_.DeQue<DT_X>();
            auto z = zQueue_.AllocTensor<DT_X>();
            AscendC::Mul(z, x, y, TILE_LENGTH);
            xQueue_.FreeTensor(x);
            yQueue_.FreeTensor(y);
            zQueue_.EnQue(z);

            z = zQueue_.DeQue<DT_X>();
            AscendC::DataCopy(zGm_[offset], z, TILE_LENGTH);
            zQueue_.FreeTensor(z);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> xQueue_, yQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> zQueue_;
    AscendC::GlobalTensor<DT_X> xGm_, yGm_, zGm_;
    uint32_t blockLength_;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
