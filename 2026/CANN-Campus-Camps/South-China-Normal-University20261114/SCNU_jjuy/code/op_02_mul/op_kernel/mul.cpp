#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

namespace {
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 256;
}

template <typename DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               uint32_t length)
    {
        // 8 核均分，每核 2048 个元素，每次处理 256 个。
        blockLength_ = length / AscendC::GetBlockNum();
        const uint32_t offset = AscendC::GetBlockIdx() * blockLength_;

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + offset, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + offset, blockLength_);
        zGm_.SetGlobalBuffer((__gm__ DT_X *)z + offset, blockLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM,
                         TILE_LENGTH * sizeof(DT_X));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM,
                         TILE_LENGTH * sizeof(DT_X));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM,
                         TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        const uint32_t loopCount = blockLength_ / TILE_LENGTH;
        for (uint32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX_.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY_.AllocTensor<DT_X>();

        const uint32_t offset = progress * TILE_LENGTH;
        AscendC::DataCopy(xLocal, xGm_[offset], TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm_[offset], TILE_LENGTH);

        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ_.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, TILE_LENGTH);

        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ_.DeQue<DT_X>();

        AscendC::DataCopy(zGm_[progress * TILE_LENGTH], zLocal,
                          TILE_LENGTH);
        outQueueZ_.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::GlobalTensor<DT_X> zGm_;
    uint32_t blockLength_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                             GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tilingData, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tilingData.length);
    op.Process();
}
