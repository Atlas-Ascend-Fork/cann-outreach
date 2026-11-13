// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z,
        uint32_t length, AscendC::TPipe& pipe)
    {
        length_ = length;

        xGm_.SetGlobalBuffer((__gm__ DT_X*)x, length_);
        yGm_.SetGlobalBuffer((__gm__ DT_X*)y, length_);
        zGm_.SetGlobalBuffer((__gm__ DT_X*)z, length_);

        constexpr uint32_t tileBytes =
            MUL_TILE_ELEMENTS * sizeof(DT_X);

        pipe.InitBuffer(xQueue_, 1, tileBytes);
        pipe.InitBuffer(yQueue_, 1, tileBytes);
        pipe.InitBuffer(zQueue_, 1, tileBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t firstOffset =
            AscendC::GetBlockIdx() * MUL_TILE_ELEMENTS;
        const uint32_t stride =
            AscendC::GetBlockNum() * MUL_TILE_ELEMENTS;

        for (uint32_t offset = firstOffset;
             offset < length_;
             offset += stride) {
      
            auto xLocal = xQueue_.AllocTensor<DT_X>();
            auto yLocal = yQueue_.AllocTensor<DT_X>();

            AscendC::DataCopy(
                xLocal, xGm_[offset], MUL_TILE_ELEMENTS);
            AscendC::DataCopy(
                yLocal, yGm_[offset], MUL_TILE_ELEMENTS);

            xQueue_.EnQue<DT_X>(xLocal);
            yQueue_.EnQue<DT_X>(yLocal);

            auto xReady = xQueue_.DeQue<DT_X>();
            auto yReady = yQueue_.DeQue<DT_X>();
            auto zLocal = zQueue_.AllocTensor<DT_X>();

            AscendC::Mul(
                zLocal, xReady, yReady, MUL_TILE_ELEMENTS);

            zQueue_.EnQue<DT_X>(zLocal);
            xQueue_.FreeTensor(xReady);
            yQueue_.FreeTensor(yReady);

            auto zReady = zQueue_.DeQue<DT_X>();

            AscendC::DataCopy(
                zGm_[offset], zReady, MUL_TILE_ELEMENTS);

            zQueue_.FreeTensor(zReady);
        }
    }

private:
    uint32_t length_;

    AscendC::GlobalTensor<DT_X> xGm_, yGm_, zGm_;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> xQueue_, yQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> zQueue_;
};

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData, tiling_data, tiling);

    AscendC::TPipe pipe;
    KernelMul<DT_X> op;

    op.Init(x, y, z, tiling_data.length, pipe);
    op.Process();
}