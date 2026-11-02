#include "kernel_operator.h"
#include "sqrt_tiling_data.h"
#include "sqrt_tiling_key.h"

namespace SqrtCustom {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t DMA_BLOCK_BYTES = 32;

template <typename T>
class KernelSqrt {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData& tiling)
    {
        const int64_t offset = static_cast<int64_t>(AscendC::GetBlockIdx()) * tiling.blockFactor;
        if (offset >= tiling.totalNum) {
            return;
        }
        const int64_t remaining = tiling.totalNum - offset;
        blockLength_ = remaining < tiling.blockFactor ? remaining : tiling.blockFactor;
        tileLength_ = static_cast<uint32_t>(tiling.ubFactor);
        xGm_.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (int64_t offset = 0; offset < blockLength_; offset += tileLength_) {
            const int64_t remaining = blockLength_ - offset;
            const uint32_t count = remaining < tileLength_
                ? static_cast<uint32_t>(remaining) : tileLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
        if ((count * sizeof(T)) % DMA_BLOCK_BYTES == 0) {
            AscendC::DataCopy(xLocal, xGm_[offset], count);
        } else {
            AscendC::DataCopyExtParams params{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padding{true, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(xLocal, xGm_[offset], params, padding);
        }
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal = inQueue_.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueue_.AllocTensor<T>();
        AscendC::Sqrt(yLocal, xLocal, count);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> yLocal = outQueue_.DeQue<T>();
        if ((count * sizeof(T)) % DMA_BLOCK_BYTES == 0) {
            AscendC::DataCopy(yGm_[offset], yLocal, count);
        } else {
            AscendC::DataCopyExtParams params{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(yGm_[offset], yLocal, params);
        }
        outQueue_.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    int64_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
};
} // namespace SqrtCustom

template <int DATA_TYPE>
__global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SqrtTilingData);
    GET_TILING_DATA_WITH_STRUCT(SqrtTilingData, tilingData, tiling);
    // An empty tensor must not allocate buffers or form any GM address.
    if (tilingData.totalNum == 0) {
        return;
    }
    static_assert(DATA_TYPE == SQRT_FLOAT16 || DATA_TYPE == SQRT_FLOAT32,
                  "Unsupported Sqrt dispatch ID");
    if constexpr (DATA_TYPE == SQRT_FLOAT32) {
        SqrtCustom::KernelSqrt<float> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else {
        SqrtCustom::KernelSqrt<half> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
}
