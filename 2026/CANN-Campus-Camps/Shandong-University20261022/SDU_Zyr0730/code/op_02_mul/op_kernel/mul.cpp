#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length)
    {
        length_ = length;

        xGm_.SetGlobalBuffer((__gm__ DT_X*)x, length_);
        yGm_.SetGlobalBuffer((__gm__ DT_X*)y, length_);
        zGm_.SetGlobalBuffer((__gm__ DT_X*)z, length_);

        pipe_.InitBuffer(queueX_, 1, TILE_LENGTH * sizeof(DT_X));
        pipe_.InitBuffer(queueY_, 1, TILE_LENGTH * sizeof(DT_X));
        pipe_.InitBuffer(queueZ_, 1, TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        uint64_t offset =
            static_cast<uint64_t>(AscendC::GetBlockIdx()) * TILE_LENGTH;

        uint64_t stride =
            static_cast<uint64_t>(AscendC::GetBlockNum()) * TILE_LENGTH;

        for (; offset < length_; offset += stride) {
            uint64_t remaining = length_ - offset;
            uint32_t count = remaining < TILE_LENGTH
                ? static_cast<uint32_t>(remaining)
                : TILE_LENGTH;

            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        auto xLocal = queueX_.AllocTensor<DT_X>();
        auto yLocal = queueY_.AllocTensor<DT_X>();

        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(count * sizeof(DT_X)),
            0, 0, 0
        };

        AscendC::DataCopyPadExtParams<DT_X> padParams{
            false, 0, 0, static_cast<DT_X>(0)
        };

        AscendC::DataCopyPad(
            xLocal, xGm_[offset], copyParams, padParams);
        AscendC::DataCopyPad(
            yLocal, yGm_[offset], copyParams, padParams);

        queueX_.EnQue(xLocal);
        queueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto xLocal = queueX_.DeQue<DT_X>();
        auto yLocal = queueY_.DeQue<DT_X>();
        auto zLocal = queueZ_.AllocTensor<DT_X>();

        // 逐元素乘法：z = x * y。
        AscendC::Mul(zLocal, xLocal, yLocal, count);

        queueZ_.EnQue(zLocal);
        queueX_.FreeTensor(xLocal);
        queueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        auto zLocal = queueZ_.DeQue<DT_X>();

        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(count * sizeof(DT_X)),
            0, 0, 0
        };

        AscendC::DataCopyPad(zGm_[offset], zLocal, copyParams);
        queueZ_.FreeTensor(zLocal);
    }

private:
    static constexpr uint32_t TILE_LENGTH = 2048;

    AscendC::TPipe pipe_;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> queueX_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> queueY_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> queueZ_;

    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::GlobalTensor<DT_X> zGm_;

    uint32_t length_;
};

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}