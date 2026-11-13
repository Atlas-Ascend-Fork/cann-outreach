#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <typename T>
class KernelMul {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               uint64_t totalLength, uint32_t tileLength)
    {
        totalLength_ = totalLength;
        tileLength_ = tileLength;
        tileCount_ = totalLength / tileLength + (totalLength % tileLength != 0);
        xGm_.SetGlobalBuffer((__gm__ T *)x, totalLength);
        yGm_.SetGlobalBuffer((__gm__ T *)y, totalLength);
        zGm_.SetGlobalBuffer((__gm__ T *)z, totalLength);
        // x and y share one allocation and one queue synchronization point.
        pipe_.InitBuffer(input_, 1, 2 * tileLength_ * sizeof(T));
        pipe_.InitBuffer(output_, 1, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint64_t tile = AscendC::GetBlockIdx(); tile < tileCount_;
             tile += AscendC::GetBlockNum()) {
            const uint64_t offset = tile * tileLength_;
            const uint64_t remaining = totalLength_ - offset;
            const uint32_t count = remaining < tileLength_ ?
                static_cast<uint32_t>(remaining) : tileLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        auto x = input_.AllocTensor<T>();
        auto y = x[tileLength_];
        if (count * sizeof(T) % 32 == 0) {
            AscendC::DataCopy(x, xGm_[offset], count);
            AscendC::DataCopy(y, yGm_[offset], count);
        } else {
            const AscendC::DataCopyExtParams params{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<T> pad{
                true, 0, static_cast<uint8_t>((32 / sizeof(T) - count % (32 / sizeof(T))) %
                                             (32 / sizeof(T))), static_cast<T>(0)};
            AscendC::DataCopyPad(x, xGm_[offset], params, pad);
            AscendC::DataCopyPad(y, yGm_[offset], params, pad);
        }
        input_.EnQue(x);
    }
    __aicore__ inline void Compute(uint32_t count)
    {
        auto x = input_.DeQue<T>();
        auto y = x[tileLength_];
        auto z = output_.AllocTensor<T>();
        AscendC::Mul(z, x, y, count);
        output_.EnQue(z);
        input_.FreeTensor(x);
    }
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        auto z = output_.DeQue<T>();
        if (count * sizeof(T) % 32 == 0) {
            AscendC::DataCopy(zGm_[offset], z, count);
        } else {
            const AscendC::DataCopyExtParams params{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(zGm_[offset], z, params);
        }
        output_.FreeTensor(z);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_;
    AscendC::GlobalTensor<T> xGm_, yGm_, zGm_;
    uint64_t totalLength_, tileCount_;
    uint32_t tileLength_;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tilingData, tiling);
    if (tilingData.totalLength == 0) {
        return;
    }
    KernelMul<DT_X> op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileLength);
    op.Process();
}
